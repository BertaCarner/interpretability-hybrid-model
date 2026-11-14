"""Mechanistic analysis tools for the trained HybridLM.

Four modules:
1. A-matrix timescales     — memory timescales per Mamba layer, vs initialisation
2. Linear probing on h_t   — is B's identity decodable from the SSM state?
3. State-dim ablation      — silence targeted state dims, measure accuracy drop
4. Pathway ablation        — switch off the SSM memory and / or the conv in
                             each Mamba layer: which of the two ways a Mamba
                             block can move information between positions does
                             the model actually rely on?
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import HybridLM, HybridConfig, SimpleMambaBlock, mamba_ablation


@torch.no_grad()
def eval_induction(
    model: HybridLM,
    dataset,                     # InductionDataset
    n_examples: int = 200,
    device: str = "cpu",
    batch_size: int = 64,
) -> tuple[float, float]:
    """(accuracy, mean logit(B) - logit(B')) at the trigger position."""
    model.eval()
    n = min(n_examples, len(dataset))
    correct = 0
    ld_sum = 0.0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        seqs = dataset.data[start:end].to(device)
        targets = dataset.targets[start:end].to(device)
        foils = dataset.foils[start:end].to(device)
        logits = model(seqs)[:, -1, :]                       # (B, V)
        correct += (logits.argmax(-1) == targets).sum().item()
        c = logits.gather(1, targets.view(-1, 1)).squeeze(1)
        f = logits.gather(1, foils.view(-1, 1)).squeeze(1)
        ld_sum += (c - f).sum().item()
    return correct / max(n, 1), ld_sum / max(n, 1)


# ---------------------------------------------------------------------------
# 1. A-matrix timescales
# ---------------------------------------------------------------------------

@dataclass
class EigenResult:
    layer: int                       # index in HybridLM.layers (Mamba layers only)
    eigenvalue_magnitudes: list[float]  # |λ| sorted descending; diagonal SSM → real
    decay_timescales: list[float]    # τ = -1/log|λ| in steps (trained A)
    init_timescales: list[float]     # same, for A at initialisation (1..N) and
                                     # the SAME Δ → isolates what training did to A
    mean_delta: float                # Δ used for both
    A_mean: list[float]              # mean A over d_inner per state dim
    A_std: list[float]


@torch.no_grad()
def empirical_mean_delta(model: HybridLM, dataset, n_examples: int = 64,
                         device: str = "cpu") -> dict[int, float]:
    """Mean Δ per Mamba layer measured on real data.

    Δ is input-dependent (selective) — using an arbitrary fixed Δ (e.g. 0.1)
    can distort the implied timescales τ = -1/log|exp(Δ·A)| by an order of
    magnitude. This measures what the trained model actually uses.
    """
    model = model.to(device).eval()
    seqs = dataset.data[:n_examples].to(device)
    _, caches = model.run_with_cache(seqs)
    return {
        li: float(caches[li]["delta"].mean())
        for li, layer in enumerate(model.layers)
        if isinstance(layer, SimpleMambaBlock)
    }


def _timescales(A_bar_diag: np.ndarray) -> list[float]:
    """τ = -1/log|λ| per entry (0 for λ≈0, inf for |λ|≥1)."""
    out = []
    for mag in np.abs(A_bar_diag):
        if mag < 1e-8:
            out.append(0.0)
        elif mag >= 1.0:
            out.append(float("inf"))
        else:
            out.append(float(-1.0 / np.log(mag)))
    return out


def analyze_A_matrices(
    model: HybridLM,
    delta_ref: float = 0.1,
    delta_per_layer: dict[int, float] | None = None,
) -> list[EigenResult]:
    """
    For every SimpleMambaBlock, the memory timescale of each state dim. The
    SSM is diagonal, so the eigenvalues of Ā = exp(Δ·A) are just its diagonal
    entries. If delta_per_layer (from empirical_mean_delta) is given, it
    overrides delta_ref — strongly recommended.

    The timescales of the initial A (1..N per channel) under the same Δ are
    returned alongside: A starts with τ_n ∝ 1/n, so only a departure from
    that curve is something the model learned.

    Returns one EigenResult per Mamba layer (skips Attention layers).
    """
    results = []
    for layer_idx, layer in enumerate(model.layers):
        if not isinstance(layer, SimpleMambaBlock):
            continue

        A = -torch.exp(layer.A_log.detach().float().cpu())   # (D_inner, N), negative
        A_mean = A.mean(dim=0)                          # (N,) — average over channels
        A_std  = A.std(dim=0)                           # (N,)
        A_init = -torch.arange(1, A.shape[1] + 1, dtype=torch.float32)   # (N,)

        dref = (delta_per_layer.get(layer_idx, delta_ref)
                if delta_per_layer else delta_ref)

        A_bar_diag = torch.exp(dref * A_mean).numpy()   # (N,)
        order = np.argsort(-np.abs(A_bar_diag))
        A_bar_init = torch.exp(dref * A_init).numpy()   # (N,), already descending

        results.append(EigenResult(
            layer=layer_idx,
            eigenvalue_magnitudes=np.abs(A_bar_diag)[order].tolist(),
            decay_timescales=_timescales(A_bar_diag[order]),
            init_timescales=_timescales(A_bar_init),
            mean_delta=float(dref),
            A_mean=A_mean.tolist(),
            A_std=A_std.tolist(),
        ))

    return results


# ---------------------------------------------------------------------------
# 2. Linear probing on h_t
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Multiclass decoding of the *content* carried by the SSM state:
    decode the IDENTITY of B (the token to be recalled) — a balanced 64-way
    problem where chance is 1/64 and every example is usable."""
    layer: int              # index in HybridLM.layers (Mamba layers only)
    position: int           # sequence position where h is read
    accuracy: float         # cross-validated multiclass decode accuracy
    chance: float           # 1 / vocab_size
    per_dim_score: list[float]  # ANOVA F-score per state dim (for ranking/ablation)
    n_examples: int


def probe_state_dimensions(
    model: HybridLM,
    dataset,
    probe_positions: list[int],
    n_examples: int = 2000,
    device: str = "cpu",
    n_splits: int = 5,
    batch_size: int = 64,
    decode: str = "target",   # 'target' → decode B; 'trigger' → decode A
    n_pca: int = 64,
) -> list[ProbeResult]:
    """
    For each Mamba layer and each position, fit a cross-validated multinomial
    logistic regression on the FULL state h_t ∈ R^{D_inner·N} to decode the
    identity of B (dataset.targets). The state is standardised and reduced to
    `n_pca` principal components inside each fold (it has far more dimensions
    than there are examples).

    Two built-in controls make the accuracy interpretable:
      - positions *before* the [A,B] bigram must decode at chance;
      - a model known (from patching) to carry B in its state — the pure-SSM
        model — must decode well. Only then does chance-level decoding in
        another model mean 'B is not in the state'.

    Use enough examples for ~30 per class (n_examples ≈ 2000 at vocab 64).

    dataset must be an InductionDataset (uses .data / .targets).
    """
    from sklearn.feature_selection import f_classif

    model = model.to(device)
    model.eval()
    n = min(n_examples, len(dataset))
    vocab_size = dataset.vocab_size

    mamba_layer_indices = [
        i for i, layer in enumerate(model.layers)
        if isinstance(layer, SimpleMambaBlock)
    ]

    labels_all = (dataset.targets[:n] if decode == "target"
                  else dataset.data[:n, -1]).numpy()

    # Collect features in batches: full h at every (layer, pos) requested
    feats: dict[tuple[int, int], list[np.ndarray]] = {
        (li, p): [] for li in mamba_layer_indices for p in probe_positions
    }
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            seqs = dataset.data[start:end].to(device)
            _, caches = model.run_with_cache(seqs)
            for li in mamba_layer_indices:
                h = caches[li]["h"]                     # (B, L, Di, N)
                for p in probe_positions:
                    feats[(li, p)].append(h[:, p].float().cpu().numpy())  # (B, Di, N)

    results = []
    for (li, p), chunks in feats.items():
        H = np.concatenate(chunks, axis=0)              # (n, Di, N)
        X = H.reshape(len(H), -1)                       # (n, Di·N)
        y = labels_all

        n_folds = n_splits if np.bincount(y).min() >= n_splits else 2
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        n_comp = min(n_pca, X.shape[1], int(len(X) * (n_folds - 1) / n_folds) - 1)
        probe = make_pipeline(
            StandardScaler(),
            PCA(n_components=n_comp, random_state=0),
            LogisticRegression(max_iter=1000, C=1.0),
        )
        preds = cross_val_predict(probe, X, y, cv=cv)
        acc = float((preds == y).mean())

        # Per-state-dim score on the channel-averaged state (ranks dims for
        # the state-dim ablation)
        f_scores, _ = f_classif(StandardScaler().fit_transform(H.mean(axis=1)), y)
        f_scores = np.nan_to_num(f_scores, nan=0.0)

        results.append(ProbeResult(
            layer=li,
            position=p,
            accuracy=acc,
            chance=1.0 / vocab_size,
            per_dim_score=f_scores.tolist(),
            n_examples=int(n),
        ))
    return results


# ---------------------------------------------------------------------------
# 3. State-dim ablation
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    layer: int
    ablated_dims: list[int]
    baseline_accuracy: float
    val_accuracy: float
    accuracy_drop: float
    kind: str = "top"       # 'top' (best-scoring dims) or 'random' (control)


def ablate_state_dims(
    model: HybridLM,
    dataset,
    layer: int,
    dims: list[int],
    n_examples: int = 200,
    device: str = "cpu",
    kind: str = "top",
) -> AblationResult:
    """
    Measure induction accuracy when SSM state dimensions `dims` in Mamba
    layer `layer` are silenced.

    Silencing is implemented by:
      1. Setting A_log[:, dims] = +50 → A = -exp(50) → Ā = exp(Δ·A) ≈ 0
         (no persistent memory in those dims)
      2. Zeroing the C_t read weights for those dims in x_proj
         (so those dims never contribute to the SSM output y_t = C_t · h_t)

    A deep copy of the model is made so the original is not modified.
    """
    model = model.to(device)
    model.eval()

    target_layer = model.layers[layer]
    assert isinstance(target_layer, SimpleMambaBlock), (
        f"Layer {layer} is not a SimpleMambaBlock; cannot ablate SSM dims"
    )

    baseline_acc, _ = eval_induction(model, dataset, n_examples, device)

    # Deep-copy and silence dims
    ablated = copy.deepcopy(model)
    tgt = ablated.layers[layer]          # SimpleMambaBlock

    with torch.no_grad():
        # Kill persistent memory in ablated dims: Ā[:, dims] = exp(-Δ·e^50) ≈ 0
        tgt.A_log[:, dims] = 50.0

        # Zero the C_t readout weights for ablated dims
        # x_proj output layout: [dt_rank | d_state (B_t) | d_state (C_t)]
        c_start = tgt.dt_rank + tgt.d_state   # start of C_t rows in x_proj.weight
        tgt.x_proj.weight[c_start + torch.tensor(dims), :] = 0.0

    ablated_acc, _ = eval_induction(ablated, dataset, n_examples, device)

    return AblationResult(
        layer=layer,
        ablated_dims=dims,
        baseline_accuracy=baseline_acc,
        val_accuracy=ablated_acc,
        accuracy_drop=baseline_acc - ablated_acc,
        kind=kind,
    )


# ---------------------------------------------------------------------------
# 4. Pathway ablation: SSM memory vs conv
# ---------------------------------------------------------------------------

PATHWAY_CONDITIONS: dict[str, dict[str, bool]] = {
    "intact":            {"memory": False, "conv": False},
    "no_memory":         {"memory": True,  "conv": False},
    "no_conv":           {"memory": False, "conv": True},
    "no_memory_no_conv": {"memory": True,  "conv": True},
}


@dataclass
class PathwayResult:
    layers: list[int]       # Mamba layers the ablation was applied to
    condition: str          # key of PATHWAY_CONDITIONS
    accuracy: float
    logit_diff: float


def pathway_ablation(
    model: HybridLM,
    dataset,
    n_examples: int = 300,
    device: str = "cpu",
) -> list[PathwayResult]:
    """
    A Mamba block can move information between positions in two ways: the
    recurrent state ('memory') and the depthwise conv, which sees the
    previous d_conv-1 tokens. For each Mamba layer on its own, and for all
    Mamba layers together, switch off one, the other, or both, and measure
    induction accuracy.

    Reading the result for one layer:
      no_memory ≈ intact, no_conv ≪ intact → the layer's job needs only the
          last few tokens (e.g. 'which token came just before me');
      no_memory ≪ intact                  → the layer carries information
          over a long range in its state;
      only no_memory_no_conv ≪ intact     → either pathway can do the job.

    These are zero-ablations of a trained model, so a drop shows the pathway
    is needed by THIS model; it does not show the task is unsolvable without it.
    """
    model = model.to(device)
    model.eval()
    mamba_layers = [i for i, l in enumerate(model.layers)
                    if isinstance(l, SimpleMambaBlock)]
    layer_sets = [[i] for i in mamba_layers]
    if len(mamba_layers) > 1:
        layer_sets.append(mamba_layers)

    results = []
    for layers in layer_sets:
        for cond, flags in PATHWAY_CONDITIONS.items():
            with mamba_ablation(model, layers=layers, **flags):
                acc, ld = eval_induction(model, dataset, n_examples, device)
            results.append(PathwayResult(layers=layers, condition=cond,
                                         accuracy=acc, logit_diff=ld))
    return results


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_analysis(
    eigen: list[EigenResult],
    probes: list[ProbeResult],
    ablations: list[AblationResult],
    pathways: list[PathwayResult],
    path: str | Path,
    layer_types: list[str] | None = None,
) -> None:
    import dataclasses
    out = {
        "layer_types": layer_types,
        "eigen":    [dataclasses.asdict(r) for r in eigen],
        "probe":    [dataclasses.asdict(r) for r in probes],
        "ablation": [dataclasses.asdict(r) for r in ablations],
        "pathway":  [dataclasses.asdict(r) for r in pathways],
    }
    Path(path).write_text(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from train import default_hybrid_config
    from data import InductionDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--layer_pattern", default="MAMA")
    parser.add_argument("--vocab_size", type=int, default=64)
    parser.add_argument("--seq_len",   type=int, default=64)
    parser.add_argument("--n_examples", type=int, default=2000)
    parser.add_argument("--out", default="results/analysis.json")
    args = parser.parse_args()

    cfg = default_hybrid_config(vocab_size=args.vocab_size,
                                layer_pattern=args.layer_pattern)
    model = HybridLM(cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    # Fixed-distance dataset → known critical positions for probing
    d_probe = args.seq_len - 16
    probe_ds = InductionDataset(args.n_examples, seq_len=args.seq_len,
                                vocab_size=args.vocab_size, seed=99,
                                min_dist=d_probe, max_dist=d_probe)
    first_B = int(probe_ds.first_positions[0]) + 1

    print("Timescales (empirical Δ)...")
    deltas = empirical_mean_delta(model, probe_ds)
    eigen = analyze_A_matrices(model, delta_per_layer=deltas)

    positions = [first_B - 2, first_B, (first_B + args.seq_len - 1) // 2,
                 args.seq_len - 2]
    print(f"Decoding B identity from h at positions {positions} "
          f"(pos {first_B - 2} is a pre-bigram negative control)...")
    probes = probe_state_dimensions(model, probe_ds,
                                    probe_positions=positions,
                                    n_examples=args.n_examples)

    mamba_indices = [i for i, l in enumerate(model.layers)
                     if isinstance(l, SimpleMambaBlock)]
    ablations = []
    for li in mamba_indices:
        pr = [p for p in probes if p.layer == li and p.position == args.seq_len - 2]
        if not pr:
            continue
        scores = np.array(pr[0].per_dim_score)
        top2 = np.argsort(-scores)[:2].tolist()
        print(f"Ablating dims {top2} in Mamba layer L{li}...")
        abl = ablate_state_dims(model, probe_ds, layer=li, dims=top2,
                                n_examples=200)
        ablations.append(abl)
        print(f"  drop: {abl.accuracy_drop:.3f}")

    print("Pathway ablation (SSM memory vs conv)...")
    pathways = pathway_ablation(model, probe_ds)
    for r in pathways:
        print(f"  L{r.layers} {r.condition:18s} acc {r.accuracy:.3f}")

    save_analysis(eigen, probes, ablations, pathways, args.out,
                  layer_types=[cfg.layer_type(i) for i in range(cfg.n_layers)])
    print(f"Saved to {args.out}")
