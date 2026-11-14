"""Q3 — Memory limits: at what A₁…A₂ distance does recall fail, and is the SSM
state what carries the answer across that distance?

Behavioural side:
    For each distance d, build a fixed-distance induction set and measure
    accuracy + logit-diff at the trigger. Distances up to the training length
    use the training sequence length; longer ones use seq_len = d+8 and are
    flagged out-of-distribution. Models with learned absolute positions cannot
    be evaluated beyond the training length and skip those points.

Mechanistic side — two direct measurements per distance:

    1. Reliance on SSM memory (causal).
       Accuracy with the recurrent state switched off in every Mamba layer
       (model.mamba_ablation(memory=True)). If accuracy is unchanged, the
       model does not use the SSM state to bridge the distance, whatever the
       state happens to contain.

    2. B-signal in the carried state (observational).
       Run a clean sequence and its B→B' twin, and compare the state that the
       trigger step receives (h at position L-2):

           signal = ‖h_clean − h_corrupted‖   (how much the state depends on B)
           spread = ‖h_clean − mean(h_clean)‖ (how much it varies between
                                               sequences for any other reason)

       signal / spread ≈ 0 means the state the trigger receives says nothing
       about B; ≈ 1 or more means B is a dominant source of variation.

    Also stored, as a description of the state rather than a predictor of
    recall: the decay factor exp(A_n · Σ_t Δ_t) from the write position to the
    trigger, with the empirical (input-dependent) Δ. Decay alone does not
    decide whether B can be read out — a shrunken state is still decodable;
    what breaks recall is later writes drowning out B, which is what
    signal / spread measures.

Output: results/distance_<run_name>.json
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from data import InductionDataset
from model import HybridLM, SimpleMambaBlock, mamba_ablation
from patch import make_corrupted


DEFAULT_DISTANCES = [4, 8, 12, 16, 24, 32, 48, 62, 96, 128, 192, 256]


@torch.no_grad()
def _eval_at_distance(
    model: HybridLM,
    d: int,
    seq_len: int,
    vocab_size: int,
    n_examples: int,
    device: str,
    seed: int,
    batch_size: int = 64,
) -> tuple[float, float]:
    ds = InductionDataset(n_examples, seq_len=seq_len, vocab_size=vocab_size,
                          seed=seed, min_dist=d, max_dist=d)
    correct = total = 0
    ld_sum = 0.0
    for start in range(0, len(ds), batch_size):
        seqs = ds.data[start:start + batch_size].to(device)
        targets = ds.targets[start:start + batch_size].to(device)
        foils = ds.foils[start:start + batch_size].to(device)
        logits = model(seqs)[:, -1, :]
        correct += (logits.argmax(-1) == targets).sum().item()
        total += targets.numel()
        c = logits.gather(1, targets.view(-1, 1)).squeeze(1)
        f = logits.gather(1, foils.view(-1, 1)).squeeze(1)
        ld_sum += (c - f).sum().item()
    return correct / total, ld_sum / total


@torch.no_grad()
def _state_signal(
    model: HybridLM,
    d: int,
    seq_len: int,
    vocab_size: int,
    device: str,
    seed: int,
    n_examples: int = 32,
) -> dict[int, dict]:
    """
    Per Mamba layer: how strongly the state handed to the trigger step
    (h at position L-2) depends on B, relative to how much it varies between
    sequences. See the module docstring.
    """
    ds = InductionDataset(n_examples, seq_len=seq_len, vocab_size=vocab_size,
                          seed=seed + 2, min_dist=d, max_dist=d)
    first_A_pos = int(ds.first_positions[0])
    clean = ds.data.to(device)
    corrupted = make_corrupted(ds.data, first_A_pos, ds.foils, corruption="B").to(device)
    read_pos = seq_len - 2

    _, clean_caches = model.run_with_cache(clean)
    h_clean = {li: c["h"][:, read_pos].flatten(1).float()         # (B, Di·N)
               for li, c in enumerate(clean_caches) if "h" in c}
    del clean_caches
    _, corr_caches = model.run_with_cache(corrupted)

    out: dict[int, dict] = {}
    for li, hc in h_clean.items():
        hk = corr_caches[li]["h"][:, read_pos].flatten(1).float()
        signal = (hc - hk).norm(dim=1).mean()
        spread = (hc - hc.mean(dim=0, keepdim=True)).norm(dim=1).mean()
        out[li] = {
            "signal": float(signal),
            "spread": float(spread),
            "signal_over_spread": float(signal / spread.clamp_min(1e-12)),
        }
    return out


@torch.no_grad()
def _state_decay(
    model: HybridLM,
    d: int,
    seq_len: int,
    vocab_size: int,
    device: str,
    seed: int,
    n_examples: int = 16,
) -> dict[int, dict]:
    """
    Per Mamba layer: decay factor exp(A_n · Σ_{t>p} Δ_t) from the write
    position p = first_B_pos to the trigger, per state dim n, averaged over
    examples and channels, using the empirical (input-dependent) Δ.
    Descriptive only — see the module docstring.
    """
    ds = InductionDataset(n_examples, seq_len=seq_len, vocab_size=vocab_size,
                          seed=seed + 1, min_dist=d, max_dist=d)
    seqs = ds.data.to(device)
    first_B_pos = int(ds.first_positions[0]) + 1

    _, caches = model.run_with_cache(seqs)
    out: dict[int, dict] = {}
    for li, layer in enumerate(model.layers):
        if not isinstance(layer, SimpleMambaBlock):
            continue
        A = -torch.exp(layer.A_log.detach().float())          # (Di, N), negative
        delta = caches[li]["delta"].float()                    # (B, L, Di)

        # Σ Δ_t over the retention window (write → trigger)
        S_emp = delta[:, first_B_pos + 1:, :].sum(dim=1)       # (B, Di)
        log_decay = torch.einsum("bd,dn->bdn", S_emp, A)       # (B, Di, N)
        decay = log_decay.exp().mean(dim=(0, 1))               # (N,)

        out[li] = {
            "decay_empirical": decay.cpu().tolist(),           # per state dim
            "mean_delta": float(delta.mean()),
            "retention_steps": int(delta.shape[1] - (first_B_pos + 1)),
        }
    return out


def run_distance_sweep(
    model: HybridLM,
    vocab_size: int = 64,
    trained_seq_len: int = 64,
    distances: list[int] | None = None,
    n_examples: int = 300,
    device: str = "cpu",
    seed: int = 777,
    out_path: str | Path | None = None,
    run_name: str = "model",
) -> dict:
    model = model.to(device)
    model.eval()
    distances = distances or DEFAULT_DISTANCES

    has_learned_pos = model.pos_emb is not None
    has_mamba = any(isinstance(l, SimpleMambaBlock) for l in model.layers)

    records = []
    for d in distances:
        if d <= trained_seq_len - 2:
            # In-range: evaluate at the TRAINED sequence length, varying only
            # A₁'s position — exactly the training/validation distribution.
            seq_len = trained_seq_len
            in_dist = True
        else:
            # Beyond trained length: sequence length itself is out of
            # distribution. Not possible with learned absolute positions.
            if has_learned_pos:
                print(f"  d={d}: skipped (beyond trained length — learned "
                      "pos-emb model cannot extrapolate)")
                continue
            seq_len = d + 8
            in_dist = False

        acc, ld = _eval_at_distance(model, d, seq_len, vocab_size,
                                    n_examples, device, seed)
        record = {
            "distance": d,
            "seq_len": seq_len,
            "in_distribution": in_dist,
            "accuracy": acc,
            "logit_diff": ld,
        }
        msg = f"  d={d:4d} (L={seq_len:3d}) | acc {acc:.3f} | LD {ld:+.3f}"

        if has_mamba:
            with mamba_ablation(model, memory=True):
                acc_nm, ld_nm = _eval_at_distance(model, d, seq_len, vocab_size,
                                                  n_examples, device, seed)
            signal = _state_signal(model, d, seq_len, vocab_size, device, seed)
            decay = _state_decay(model, d, seq_len, vocab_size, device, seed)
            record.update({
                "accuracy_no_memory": acc_nm,
                "logit_diff_no_memory": ld_nm,
                "state_signal": {str(k): v for k, v in signal.items()},
                "state_decay": {str(k): v for k, v in decay.items()},
            })
            best = max(v["signal_over_spread"] for v in signal.values())
            msg += (f" | no-memory acc {acc_nm:.3f} | "
                    f"B-signal in state {best:.3f}")

        records.append(record)
        print(msg)

    result = {
        "run_name": run_name,
        "layer_pattern": model.config.layer_pattern,
        "trained_seq_len": trained_seq_len,
        "chance_accuracy": 1.0 / vocab_size,
        "records": records,
    }
    if out_path is not None:
        Path(out_path).write_text(json.dumps(result, indent=2))
        print(f"Saved distance sweep to {out_path}")
    return result
