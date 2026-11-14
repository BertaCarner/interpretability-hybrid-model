"""Cross-architecture activation patching for the Hybrid Mamba-Transformer model.

Implements causal tracing over a heterogeneous layer stack:
  - Any layer   : patch the residual stream after the block ('resid') — the
                  one component that means the same thing in every architecture
  - Mamba layers: patch the SSM hidden state h_t at specific positions
  - Attn layers : patch K, V or the attention output at specific positions

Metric:
    logit_diff = logit(B) - logit(B')
    effect(l, t, c)   = patched_LD - corrupted_LD            (logit units)
    recovery(l, t, c) = effect / (clean_LD - corrupted_LD)   (fraction)

    Recovery is a ratio, so it is only meaningful when clean_LD - corrupted_LD
    is well away from zero (PatchGrid.reliable). The effect in logit units is
    always meaningful and is what the formation analysis plots.

Corruption protocols (one token is changed, so recovery can be attributed to
specific positions). An induction circuit has two halves and each corruption
isolates one of them:

    'B' — replace B at first_B_pos with the foil B'.
          Clean and corrupted runs differ only in WHICH token should be
          recalled → patching finds the path that carries the answer
          (the 'move' half: typically V of an induction head, or the SSM state).

    'A' — replace A₁ at first_A_pos with another token A'.
          The answer B is still in the sequence, but nothing marks it as the
          token that followed A → patching finds the path that does the
          MATCHING (the 'previous token was A' half: typically K of an
          induction head, and whatever writes that information next to B).

Patchable components:
    Any layer   : 'resid'     — block output (post-residual)
    Mamba layers: 'h'         — SSM hidden state post-step
    Attn layers : 'K'         — key projection
                  'V'         — value projection
                  'attn_out'  — full attention output

Output:
    PatchGrid — (n_layers, seq_len) tensors per component. Cells that do not
    apply to a layer type (e.g. 'K' on a Mamba layer) are NaN.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import torch

from model import HybridLM, SimpleMambaBlock, StandardAttentionBlock


# ---------------------------------------------------------------------------
# Components, corruptions, logit difference helpers
# ---------------------------------------------------------------------------

ComponentName = Literal["resid", "h", "K", "V", "attn_out"]

SHARED_COMPONENTS: tuple[str, ...] = ("resid",)
MAMBA_COMPONENTS: tuple[str, ...] = ("h",)
ATTN_COMPONENTS: tuple[str, ...] = ("K", "V", "attn_out")
ALL_COMPONENTS: tuple[str, ...] = SHARED_COMPONENTS + MAMBA_COMPONENTS + ATTN_COMPONENTS

CORRUPTIONS: tuple[str, ...] = ("B", "A")

# Below this |clean_LD - corrupted_LD| the recovery ratio is noise.
MIN_RELIABLE_LD_GAP = 0.5


def valid_components(layer_type: str, components: list[str]) -> list[str]:
    """Subset of `components` that can be patched on a 'mamba' / 'attn' layer."""
    ok = SHARED_COMPONENTS + (MAMBA_COMPONENTS if layer_type == "mamba" else ATTN_COMPONENTS)
    return [c for c in components if c in ok]


def logit_diff(
    logits: torch.Tensor,                    # (B, V)
    correct: int | torch.Tensor,             # scalar or (B,) per-example B
    foil: int | torch.Tensor,                # scalar or (B,) per-example B'
) -> float:
    """Mean logit(correct) - logit(foil) over batch (per-example tokens supported)."""
    if isinstance(correct, int):
        correct = torch.full((logits.shape[0],), correct, device=logits.device)
    if isinstance(foil, int):
        foil = torch.full((logits.shape[0],), foil, device=logits.device)
    c = logits.gather(1, correct.view(-1, 1)).squeeze(1)   # (B,)
    f = logits.gather(1, foil.view(-1, 1)).squeeze(1)      # (B,)
    return (c - f).mean().item()


def _last_logits(logits: torch.Tensor) -> torch.Tensor:
    """(B, L, V) → (B, V) at the final position."""
    return logits[:, -1, :]


def alternative_keys(
    A: torch.Tensor,        # (B,) trigger token
    B: torch.Tensor,        # (B,) target token
    B_prime: torch.Tensor,  # (B,) foil token
    vocab_size: int,
) -> torch.Tensor:
    """Per-example A' ∉ {A, B, B'} for the 'A' corruption: the first of
    A+1, A+2, A+3 (mod V) that clashes with neither B nor B'."""
    assert vocab_size >= 4
    alt = (A + 1) % vocab_size
    for _ in range(2):
        clash = (alt == B) | (alt == B_prime)
        alt = torch.where(clash, (alt + 1) % vocab_size, alt)
    return alt


def make_corrupted(
    seqs: torch.Tensor,       # (B, L) clean sequences, A₁ at first_A_pos
    first_A_pos: int,
    foils: torch.Tensor,      # (B,) per-example B'
    corruption: str = "B",
    vocab_size: Optional[int] = None,   # required for corruption='A'
) -> torch.Tensor:
    """Corrupted copy of `seqs` — see the module docstring for the two protocols."""
    assert corruption in CORRUPTIONS, f"corruption must be one of {CORRUPTIONS}"
    corrupted = seqs.clone()
    if corruption == "B":
        corrupted[:, first_A_pos + 1] = foils
    else:
        assert vocab_size is not None, "corruption='A' needs vocab_size"
        corrupted[:, first_A_pos] = alternative_keys(
            seqs[:, first_A_pos], seqs[:, first_A_pos + 1], foils, vocab_size
        )
    return corrupted


# ---------------------------------------------------------------------------
# Patched forward pass
# ---------------------------------------------------------------------------

def _patched_forward(
    model: HybridLM,
    corrupted_ids: torch.Tensor,    # (B, L)
    clean_caches: list[dict],
    patch_layer: int,
    patch_pos: int,
    patch_component: ComponentName,
) -> torch.Tensor:
    """
    Run HybridLM on `corrupted_ids`, replacing one activation component at
    (patch_layer, patch_pos) with the corresponding value from `clean_caches`.

    Returns logits (B, L, V).
    """
    B, L = corrupted_ids.shape
    x = model._embed(corrupted_ids)             # (B, L, D)

    for layer_idx, layer in enumerate(model.layers):

        if layer_idx != patch_layer:
            # Standard forward, no patching
            x, _ = layer(x)
            continue

        # ── Patching this layer ──────────────────────────────────────────
        mask = torch.zeros(L, dtype=torch.bool, device=x.device)
        mask[patch_pos] = True
        clean = clean_caches[layer_idx]

        if patch_component == "resid":
            # Overwrite the block output at patch_pos (any layer type)
            x, _ = layer(x)
            x = x.clone()
            x[:, patch_pos] = clean["resid_out"][:, patch_pos]   # (B, D)

        elif isinstance(layer, SimpleMambaBlock) and patch_component == "h":
            x, _ = layer(x, h_override=clean["h"], h_override_mask=mask)

        elif isinstance(layer, StandardAttentionBlock) and patch_component == "K":
            x, _ = layer(x, K_override=clean["K"], KV_mask=mask)

        elif isinstance(layer, StandardAttentionBlock) and patch_component == "V":
            x, _ = layer(x, V_override=clean["V"], KV_mask=mask)

        elif isinstance(layer, StandardAttentionBlock) and patch_component == "attn_out":
            x, _ = layer(x, attn_out_override=clean["attn_out"], attn_out_mask=mask)

        else:
            raise ValueError(
                f"component '{patch_component}' cannot be patched on layer "
                f"{layer_idx} ({model.config.layer_type(layer_idx)})"
            )

    x = model.norm_f(x)
    return model.lm_head(x)                     # (B, L, V)


# ---------------------------------------------------------------------------
# Full patching scan
# ---------------------------------------------------------------------------

@dataclass
class PatchGrid:
    """Patching results for one model run under one corruption protocol."""
    n_layers: int
    seq_len: int
    layer_types: list[str]           # 'mamba' or 'attn' per layer
    # component → (n_layers, seq_len) recovery fraction
    grids: dict[str, torch.Tensor] = field(default_factory=dict)
    # component → (n_layers, seq_len) effect = patched_LD - corrupted_LD (logits)
    effects: dict[str, torch.Tensor] = field(default_factory=dict)
    clean_ld: float = 0.0
    corrupted_ld: float = 0.0
    corruption: str = "B"

    @property
    def total_effect(self) -> float:
        """clean_LD - corrupted_LD: how much the corruption changes behaviour."""
        return self.clean_ld - self.corrupted_ld

    @property
    def reliable(self) -> bool:
        """False → the recovery denominator is tiny; do not read `grids`."""
        return abs(self.total_effect) >= MIN_RELIABLE_LD_GAP

    def to_dict(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "seq_len": self.seq_len,
            "layer_types": self.layer_types,
            "corruption": self.corruption,
            "clean_ld": self.clean_ld,
            "corrupted_ld": self.corrupted_ld,
            "reliable": self.reliable,
            "grids": {k: v.tolist() for k, v in self.grids.items()},
            "effects": {k: v.tolist() for k, v in self.effects.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PatchGrid":
        pg = cls(
            n_layers=d["n_layers"],
            seq_len=d["seq_len"],
            layer_types=d["layer_types"],
            clean_ld=d["clean_ld"],
            corrupted_ld=d["corrupted_ld"],
            corruption=d.get("corruption", "B"),
        )
        pg.grids = {k: torch.tensor(v, dtype=torch.float) for k, v in d["grids"].items()}
        pg.effects = {k: torch.tensor(v, dtype=torch.float)
                      for k, v in d.get("effects", {}).items()}
        return pg


def _recovery_from_effects(
    effects: dict[str, torch.Tensor], total_effect: float
) -> dict[str, torch.Tensor]:
    """effect / (clean_LD - corrupted_LD), keeping the sign of the denominator.
    All-NaN when the denominator is exactly degenerate."""
    if abs(total_effect) < 1e-8:
        return {c: torch.full_like(e, float("nan")) for c, e in effects.items()}
    return {c: e / total_effect for c, e in effects.items()}


@torch.no_grad()
def patch_scan(
    model: HybridLM,
    clean_ids: torch.Tensor,          # (B, L)
    corrupted_ids: torch.Tensor,      # (B, L)
    correct_token: int | torch.Tensor,   # B (scalar or per-example (B,))
    foil_token: int | torch.Tensor,      # B' (scalar or per-example (B,))
    components: Optional[list[str]] = None,
    positions: Optional[list[int]] = None,   # None → all positions
    corruption: str = "B",                   # label only; ids are already corrupted
) -> PatchGrid:
    """
    Sweep all (layer, position, component) combinations and compute the
    logit-difference effect and recovery.

    Only sweeps components that are valid for each layer type
    (see valid_components).
    """
    if components is None:
        components = list(ALL_COMPONENTS)

    device = next(model.parameters()).device
    clean_ids = clean_ids.to(device)
    corrupted_ids = corrupted_ids.to(device)
    if isinstance(correct_token, torch.Tensor):
        correct_token = correct_token.to(device)
    if isinstance(foil_token, torch.Tensor):
        foil_token = foil_token.to(device)

    # Baseline runs
    clean_logits, clean_caches = model.run_with_cache(clean_ids)
    corrupted_logits, _ = model.run_with_cache(corrupted_ids)

    clean_ld = logit_diff(_last_logits(clean_logits), correct_token, foil_token)
    corr_ld = logit_diff(_last_logits(corrupted_logits), correct_token, foil_token)

    n_layers = len(model.layers)
    L = clean_ids.shape[1]
    layer_types = [model.config.layer_type(i) for i in range(n_layers)]

    # NaN marks (layer, pos, component) cells that were never evaluated
    # (invalid component for that layer type, or position not scanned).
    effects: dict[str, torch.Tensor] = {
        c: torch.full((n_layers, L), float("nan")) for c in components
    }
    scan_positions = list(range(L)) if positions is None else positions

    for layer_idx in range(n_layers):
        for comp in valid_components(layer_types[layer_idx], components):
            for pos in scan_positions:
                patched_logits = _patched_forward(
                    model, corrupted_ids, clean_caches,
                    patch_layer=layer_idx,
                    patch_pos=pos,
                    patch_component=comp,
                )
                patched_ld = logit_diff(_last_logits(patched_logits), correct_token, foil_token)
                effects[comp][layer_idx, pos] = patched_ld - corr_ld

    pg = PatchGrid(
        n_layers=n_layers,
        seq_len=L,
        layer_types=layer_types,
        clean_ld=clean_ld,
        corrupted_ld=corr_ld,
        corruption=corruption,
    )
    pg.effects = effects
    pg.grids = _recovery_from_effects(effects, pg.total_effect)
    return pg


# ---------------------------------------------------------------------------
# Batch experiment: average over many induction examples
# ---------------------------------------------------------------------------

def run_patch_experiment(
    model: HybridLM,
    dataset,                      # InductionDataset with min_dist == max_dist
    n_examples: int = 100,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
    positions: Optional[list[int]] = None,
    corruption: str = "B",
    components: Optional[list[str]] = None,
) -> PatchGrid:
    """
    Batched patch scan over a FIXED-DISTANCE dataset.

    Requiring min_dist == max_dist means first_A_pos, first_B_pos and
    second_A_pos are identical across examples, so patching a given position
    means the same thing for every example and the scan can run in batch
    (one forward per (layer, pos, component) instead of one per example).

    The foil token B' is per-example (dataset.foils), guaranteeing
    B' ≠ B ≠ A. `corruption` selects which token is replaced ('B' or 'A',
    see module docstring).

    Effects (logit units) and the clean / corrupted LDs are averaged over
    chunks weighted by chunk size; recovery is then computed ONCE from those
    averages, so a chunk with a small denominator cannot dominate.
    """
    assert getattr(dataset, "min_dist", None) == getattr(dataset, "max_dist", None), (
        "run_patch_experiment requires a fixed-distance dataset "
        "(InductionDataset(min_dist=d, max_dist=d)) so that critical positions "
        "align across examples and the scan can be batched."
    )
    model = model.to(device)
    model.eval()
    if components is None:
        components = list(ALL_COMPONENTS)

    n = min(n_examples, len(dataset))
    seqs = dataset.data[:n]                        # (n, L)
    targets = dataset.targets[:n]                  # (n,)
    foils = dataset.foils[:n]                      # (n,)
    first_A_pos = int(dataset.first_positions[0])

    n_layers = len(model.layers)
    seq_len = seqs.shape[1]
    layer_types = [model.config.layer_type(i) for i in range(n_layers)]

    accum: dict[str, torch.Tensor] = {}
    sum_clean_ld = sum_corr_ld = 0.0
    total = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = end - start
        clean_ids = seqs[start:end]
        corrupted_ids = make_corrupted(
            clean_ids, first_A_pos, foils[start:end],
            corruption=corruption, vocab_size=dataset.vocab_size,
        )

        pg = patch_scan(
            model, clean_ids, corrupted_ids,
            correct_token=targets[start:end],
            foil_token=foils[start:end],
            components=components,
            positions=positions,
            corruption=corruption,
        )
        for c in components:
            if c not in accum:
                accum[c] = pg.effects[c] * chunk
            else:
                accum[c] += pg.effects[c] * chunk
        sum_clean_ld += pg.clean_ld * chunk
        sum_corr_ld += pg.corrupted_ld * chunk
        total += chunk
        if verbose:
            print(f"  patched {total}/{n} examples (batched)...")

    result = PatchGrid(
        n_layers=n_layers,
        seq_len=seq_len,
        layer_types=layer_types,
        clean_ld=sum_clean_ld / max(total, 1),
        corrupted_ld=sum_corr_ld / max(total, 1),
        corruption=corruption,
    )
    result.effects = {c: g / max(total, 1) for c, g in accum.items()}
    result.grids = _recovery_from_effects(result.effects, result.total_effect)
    if verbose and not result.reliable:
        print("  WARNING: |clean_LD - corrupted_LD| "
              f"= {abs(result.total_effect):.3f} < {MIN_RELIABLE_LD_GAP} — the "
              "recovery denominator is tiny, so recovery values are noise "
              "(can wildly exceed 1). Do not interpret this scan.")
    if verbose:
        print(f"  Done. corruption={corruption} | {total} examples | "
              f"clean_LD={result.clean_ld:.3f} corrupted_LD={result.corrupted_ld:.3f} "
              f"| A1={first_A_pos} B={first_A_pos + 1} A2={seq_len-1}")
    return result


def save_patch_results(pg: PatchGrid, path: str | Path) -> None:
    Path(path).write_text(json.dumps(pg.to_dict(), indent=2))


def load_patch_results(path: str | Path) -> PatchGrid:
    return PatchGrid.from_dict(json.loads(Path(path).read_text()))
