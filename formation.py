"""Q2 — Circuit formation: when and how does the induction circuit crystallise?

For every formation checkpoint saved during training we compute:
  1. Behaviour: validation accuracy and logit-diff on a fixed-distance set.
  2. Mechanism : patching at the three critical positions
     (first_A_pos, first_B_pos, second_A_pos) for every layer/component,
     under BOTH corruptions (see patch.py):
       'B' → the path that carries the answer
       'A' → the path that does the matching

Each record stores the patching EFFECT in logit units
(patched_LD - corrupted_LD) alongside the recovery fraction. The effect is
the quantity to plot over training: early on the model has no logit
difference to recover, so the recovery *fraction* divides by ~0 and can sit
near 1 for a path that carries almost nothing. The effect stays near 0 until
the path carries real signal.

Plotting effect-vs-step next to accuracy-vs-step answers: does the
mechanism appear before, with, or after the behavioural phase transition,
and which half of the circuit moves first?
(In transformers, Olsson et al. 2022 tie the ICL bump to induction-head
formation; here we ask which *side* of the hybrid moves first.)

Output: results/formation_<run_name>.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from data import InductionDataset
from model import HybridLM, HybridConfig
from patch import CORRUPTIONS, PatchGrid, run_patch_experiment


@torch.no_grad()
def _eval_behaviour(model: HybridLM, ds: InductionDataset, device: str,
                    batch_size: int = 128) -> tuple[float, float]:
    """(accuracy, mean logit-diff) at the trigger position."""
    model.eval()
    correct = total = 0
    ld_sum = 0.0
    for start in range(0, len(ds), batch_size):
        seqs = ds.data[start:start + batch_size].to(device)
        targets = ds.targets[start:start + batch_size].to(device)
        foils = ds.foils[start:start + batch_size].to(device)
        logits = model(seqs)[:, -1, :]                       # (B, V)
        correct += (logits.argmax(-1) == targets).sum().item()
        total += targets.numel()
        c = logits.gather(1, targets.view(-1, 1)).squeeze(1)
        f = logits.gather(1, foils.view(-1, 1)).squeeze(1)
        ld_sum += (c - f).sum().item()
    return correct / total, ld_sum / total


def list_formation_checkpoints(ckpt_dir: Path, run_name: str) -> list[tuple[int, Path]]:
    """Return sorted [(step, path)] for run_name's formation checkpoints."""
    fdir = ckpt_dir / "formation"
    if not fdir.exists():
        return []
    out = []
    pat = re.compile(rf"^{re.escape(run_name)}_step(\d+)\.pt$")
    for p in sorted(fdir.iterdir()):
        m = pat.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def _at_key_positions(
    grids: dict[str, torch.Tensor], pg: PatchGrid, key_positions: dict[str, int]
) -> dict:
    """{comp: {layer: {pos_name: value | None}}} — None where not evaluated."""
    out = {}
    for comp, grid in grids.items():
        out[comp] = {}
        for li in range(pg.n_layers):
            out[comp][str(li)] = {
                name: (None if torch.isnan(grid[li, pos]) else float(grid[li, pos]))
                for name, pos in key_positions.items()
            }
    return out


def run_formation_analysis(
    cfg: HybridConfig,
    ckpt_dir: str | Path,
    run_name: str,
    seq_len: int = 64,
    vocab_size: int = 64,
    patch_dist: int | None = None,     # None → seq_len - 16
    n_eval: int = 300,
    n_patch: int = 48,
    device: str = "cpu",
    out_path: str | Path | None = None,
    max_step: int | None = None,       # ignore stale ckpts from previous runs
) -> dict:
    """
    Sweep formation checkpoints; at each, measure behaviour + patching at
    the critical positions under both corruptions. Cheap by design: only 3
    positions are patched.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpts = list_formation_checkpoints(ckpt_dir, run_name)
    if max_step is not None:
        stale = [s for s, _ in ckpts if s > max_step]
        if stale:
            print(f"  (ignoring {len(stale)} stale checkpoint(s) beyond step "
                  f"{max_step}: {stale} — delete {ckpt_dir/'formation'} to clean)")
        ckpts = [(s, p) for s, p in ckpts if s <= max_step]
    if not ckpts:
        raise FileNotFoundError(
            f"No formation checkpoints for '{run_name}' in {ckpt_dir/'formation'}. "
            "Train with ckpt_every > 0 first."
        )

    d = patch_dist if patch_dist is not None else seq_len - 16
    eval_ds = InductionDataset(n_eval, seq_len=seq_len, vocab_size=vocab_size,
                               seed=99, min_dist=d, max_dist=d)
    first_A = int(eval_ds.first_positions[0])
    key_positions = {"first_A": first_A, "first_B": first_A + 1,
                     "second_A": seq_len - 1}

    records = []
    for step, path in ckpts:
        model = HybridLM(cfg)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.to(device).eval()

        acc, ld = _eval_behaviour(model, eval_ds, device)

        patch = {}
        for corruption in CORRUPTIONS:
            pg = run_patch_experiment(
                model, eval_ds, n_examples=n_patch, device=device,
                verbose=False, positions=list(key_positions.values()),
                corruption=corruption,
            )
            patch[corruption] = {
                "clean_ld": pg.clean_ld,
                "corrupted_ld": pg.corrupted_ld,
                "total_effect": pg.total_effect,
                "reliable": pg.reliable,
                "effect": _at_key_positions(pg.effects, pg, key_positions),
                "recovery": _at_key_positions(pg.grids, pg, key_positions),
            }

        records.append({
            "step": step,
            "val_acc": acc,
            "val_logit_diff": ld,
            "patch": patch,
        })
        print(f"  step {step:6d} | acc {acc:.3f} | LD {ld:+.3f} | total effect "
              + " ".join(f"{c}:{patch[c]['total_effect']:+.2f}" for c in CORRUPTIONS))

    result = {
        "run_name": run_name,
        "layer_types": [cfg.layer_type(i) for i in range(cfg.n_layers)],
        "key_positions": key_positions,
        "patch_dist": d,
        "records": records,
    }
    if out_path is not None:
        Path(out_path).write_text(json.dumps(result, indent=2))
        print(f"Saved formation analysis to {out_path}")
    return result
