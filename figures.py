"""Figures for the Hybrid Mamba-Transformer interpretability project.

Per seed (figures/<seed_tag>/):
    figure_patching_heatmap_<arch>.png  Q1  patching, every position, both corruptions
    figure_arch_comparison.png          Q1  where each half of the circuit lives, per arch
    figure_formation_<arch>.png         Q2  behaviour and patching effect over training
    figure_distance.png                 Q3  recall vs distance, with / without SSM memory
    figure_pathway_ablation.png             SSM memory vs conv, per Mamba layer
    figure_probe_decoding.png               is B decodable from the SSM state?
    figure_ablation.png                     state-dim ablation vs random-dims control
    figure_timescales_<arch>.png            SSM memory timescales vs initialisation
    figure_phase_transition_<run>.png       written by train.py

Across seeds (figures/):
    figure_seeds_training.png           validation accuracy vs step, one line per seed

Conventions: one y-axis per panel; colour follows the entity (component,
architecture, ablation condition) and is the same in every figure; every
series also differs by marker or line style, so nothing relies on colour alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap

# ── Ink, surfaces, palette ──────────────────────────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "figure.dpi": 150,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
})

# Fixed categorical slots (validated order): blue, orange, aqua, yellow, magenta
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

COMP_ORDER = ["resid", "h", "K", "V", "attn_out"]
COMP_COLOR = dict(zip(COMP_ORDER, SLOTS))

ARCH_ORDER = ["hybrid_ma", "hybrid_am", "mamba", "attn"]
ARCH_COLOR = dict(zip(ARCH_ORDER, SLOTS))
ARCH_MARKER = {"hybrid_ma": "o", "hybrid_am": "s", "mamba": "^", "attn": "D"}
ARCH_LABEL = {
    "hybrid_ma": "MAMA (hybrid, Mamba first)",
    "hybrid_am": "AMAM (hybrid, attention first)",
    "mamba":     "MMMM (pure Mamba)",
    "attn":      "AAAA (pure attention)",
}

PATHWAY_ORDER = ["intact", "no_memory", "no_conv", "no_memory_no_conv"]
PATHWAY_COLOR = dict(zip(PATHWAY_ORDER, [MUTED] + SLOTS[:3]))
PATHWAY_LABEL = {
    "intact": "intact",
    "no_memory": "SSM memory off",
    "no_conv": "conv off",
    "no_memory_no_conv": "both off",
}

LINE_STYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D", "v", "P"]

# Diverging: red (patch makes it worse) ← neutral gray (no effect) → blue (recovers)
RECOVERY_CMAP = LinearSegmentedColormap.from_list(
    "recovery", ["#9c2424", "#e34948", "#f0efec", "#3987e5", "#104281"]
)
RECOVERY_CMAP.set_bad(SURFACE)

CORRUPTION_TITLE = {
    "B": "Corrupt B → which path carries the answer?",
    "A": "Corrupt A₁ → which path does the matching?",
}
POS_LABEL = {"first_A": "A₁", "first_B": "B", "second_A": "A₂"}


def _arch_label(arch: str) -> str:
    return ARCH_LABEL.get(arch, arch)


def _sorted_archs(archs) -> list[str]:
    known = [a for a in ARCH_ORDER if a in archs]
    return known + sorted(a for a in archs if a not in ARCH_ORDER)


def _layer_tag(li: int, ltype: str) -> str:
    return f"L{li} {'Mamba' if ltype == 'mamba' else 'attn'}"


def _mechanism_rows(layer_types: list[str], with_resid: bool) -> list[tuple[int, str]]:
    """(layer, component) rows from the bottom of the network up."""
    rows = []
    for li, lt in enumerate(layer_types):
        comps = ["h"] if lt == "mamba" else ["K", "V", "attn_out"]
        if with_resid:
            comps.append("resid")
        rows += [(li, c) for c in comps]
    return rows


def _pos_name(pos: int, first_A: int, seq_len: int) -> str:
    """Human name for a sequence position relative to the bigram."""
    if pos == first_A:
        return "A₁"
    if pos == first_A + 1:
        return "B"
    if pos == seq_len - 1:
        return "A₂"
    if first_A + 1 < pos <= first_A + 6:
        return f"B+{pos - first_A - 1}"
    return f"pos {pos}"


def _save(fig: plt.Figure, save_path: Optional[str]) -> None:
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Q1: patching heatmap — every position, both corruptions
# ---------------------------------------------------------------------------

def plot_patching_heatmap(
    pgs: dict[str, "object"],           # corruption → PatchGrid
    arch: str,
    first_A_pos: int,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    One panel per corruption. Rows are the patchable (layer, component)
    pairs from the bottom of the network up; columns are sequence positions.
    A blue cell means: copying that one activation from the clean run into
    the corrupted run restores the clean answer.
    """
    corruptions = list(pgs.keys())
    any_pg = pgs[corruptions[0]]
    rows = _mechanism_rows(any_pg.layer_types, with_resid=True)
    L = any_pg.seq_len

    fig, axes = plt.subplots(
        1, len(corruptions), figsize=(6.4 * len(corruptions), 0.34 * len(rows) + 2.2),
        sharey=True, squeeze=False, constrained_layout=True,
    )
    axes = axes[0]

    im = None
    for ax, corr in zip(axes, corruptions):
        pg = pgs[corr]
        M = np.array([np.asarray(pg.grids[c], dtype=float)[li] for li, c in rows])
        im = ax.imshow(
            np.ma.masked_invalid(M), aspect="auto", vmin=-1, vmax=1,
            cmap=RECOVERY_CMAP, origin="lower", interpolation="nearest",
            extent=[-0.5, L - 0.5, -0.5, len(rows) - 0.5],
        )
        ax.grid(False)
        # Surface-coloured gaps between layers
        for r in range(1, len(rows)):
            if rows[r][0] != rows[r - 1][0]:
                ax.axhline(r - 0.5, color=SURFACE, lw=3)

        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(
            [f"{_layer_tag(li, any_pg.layer_types[li])} · {c}" for li, c in rows]
        )
        ax.set_xlabel("Sequence position of the patch")

        # Name the three critical positions on the top edge
        top = ax.secondary_xaxis("top")
        top.set_xticks([first_A_pos, first_A_pos + 1, L - 1])
        top.set_xticklabels(["A₁", "B", "A₂"])
        top.tick_params(length=4, color=INK_2, labelcolor=INK)
        top.spines["top"].set_visible(False)
        # A₁ and B are adjacent columns: push their labels apart
        lab_A, lab_B, _ = top.get_xticklabels()
        lab_A.set_ha("right")
        lab_B.set_ha("left")

        note = (f"logit diff: clean {pg.clean_ld:+.1f}, "
                f"corrupted {pg.corrupted_ld:+.1f}")
        if not pg.reliable:
            note += "  — gap too small, do NOT interpret"
        ax.set_title(CORRUPTION_TITLE.get(corr, corr), pad=38)
        ax.annotate(note, xy=(0, 1), xycoords="axes fraction", xytext=(0, 24),
                    textcoords="offset points", fontsize=8.5, color=INK_2)

    cbar = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02)
    cbar.set_label("Recovery (1 = patch fully restores the clean answer)")
    cbar.outline.set_visible(False)
    fig.suptitle(f"{_arch_label(arch)} — activation patching, one position at a time",
                 fontweight="bold", x=0.01, ha="left")
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Q1: cross-architecture summary
# ---------------------------------------------------------------------------

def plot_arch_comparison(
    arch_pgs: dict[str, dict[str, "object"]],   # arch → corruption → PatchGrid
    first_A_pos: int,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    For each architecture, the best recovery over positions for every
    mechanism component (h for Mamba layers; K, V, attn_out for attention
    layers), under each corruption. Each cell also names the position where
    that best recovery occurs. The residual stream is left out: patching it
    late in the network recovers everything by construction.
    """
    archs = _sorted_archs(arch_pgs.keys())
    n_rows_max = max(
        len(_mechanism_rows(next(iter(arch_pgs[a].values())).layer_types, False))
        for a in archs
    )
    fig, axes = plt.subplots(
        1, len(archs), figsize=(3.3 * len(archs), 0.42 * n_rows_max + 2.0),
        squeeze=False, constrained_layout=True,
    )
    axes = axes[0]

    im = None
    for ax, arch in zip(axes, archs):
        pgs = arch_pgs[arch]
        corruptions = list(pgs.keys())
        any_pg = pgs[corruptions[0]]
        rows = _mechanism_rows(any_pg.layer_types, with_resid=False)

        M = np.full((len(rows), len(corruptions)), np.nan)
        where = [[None] * len(corruptions) for _ in rows]
        for j, corr in enumerate(corruptions):
            pg = pgs[corr]
            for i, (li, c) in enumerate(rows):
                vals = np.asarray(pg.grids[c], dtype=float)[li]
                if pg.reliable and np.isfinite(vals).any():
                    k = int(np.nanargmax(vals))
                    M[i, j], where[i][j] = vals[k], k

        im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", vmin=-1, vmax=1,
                       cmap=RECOVERY_CMAP, origin="lower")
        ax.grid(False)
        for r in range(1, len(rows)):
            if rows[r][0] != rows[r - 1][0]:
                ax.axhline(r - 0.5, color=SURFACE, lw=3)
        ax.axvline(0.5, color=SURFACE, lw=3)

        for i in range(len(rows)):
            for j in range(len(corruptions)):
                if np.isnan(M[i, j]):
                    ax.text(j, i, "n/a", ha="center", va="center",
                            fontsize=8, color=MUTED)
                elif abs(M[i, j]) >= 0.1:
                    name = _pos_name(where[i][j], first_A_pos, any_pg.seq_len)
                    ax.text(j, i, f"{M[i, j]:.2f} @ {name}", ha="center", va="center",
                            fontsize=8, color="white" if abs(M[i, j]) > 0.55 else INK)

        ax.set_xticks(range(len(corruptions)))
        ax.set_xticklabels(
            ["answer path\n(corrupt B)" if c == "B" else "matching path\n(corrupt A₁)"
             for c in corruptions]
        )
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(
            [f"{_layer_tag(li, any_pg.layer_types[li])} · {c}" for li, c in rows]
        )
        ax.tick_params(length=0)
        ax.set_title(_arch_label(arch))

    cbar = fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.02)
    cbar.set_label("Best recovery over positions")
    cbar.outline.set_visible(False)
    fig.suptitle("Where does each half of the induction circuit live?  "
                 "(value @ position of the best patch)",
                 fontweight="bold", x=0.01, ha="left")
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Q2: circuit formation timeline
# ---------------------------------------------------------------------------

def plot_formation(
    formation: dict,          # output of run_formation_analysis
    arch: Optional[str] = None,
    save_path: Optional[str] = None,
    top_k: int = 4,
) -> plt.Figure:
    """
    Top panel: validation accuracy over training (the behaviour).
    One panel per corruption below it: the patching EFFECT in logit units
    for the top_k mechanism components (ranked at the last checkpoint), with
    the total effect of the corruption (clean − corrupted) as the ceiling.
    A component 'has formed' when its line leaves zero and climbs towards
    the ceiling.
    """
    recs = formation["records"]
    steps = [r["step"] for r in recs]
    ltypes = formation["layer_types"]
    corruptions = list(recs[-1]["patch"].keys())

    fig, axes = plt.subplots(
        1 + len(corruptions), 1, figsize=(8.2, 2.2 + 2.6 * len(corruptions)),
        sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [1] + [1.5] * len(corruptions)},
    )

    axes[0].plot(steps, [r["val_acc"] for r in recs], color=INK, lw=2)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Behaviour: recall accuracy at the trigger")

    for ax, corr in zip(axes[1:], corruptions):
        # Rank mechanism traces (no 'resid') by |effect| at the last checkpoint
        last = recs[-1]["patch"][corr]["effect"]
        traces = []
        for comp, layers in last.items():
            if comp == "resid":
                continue
            for li, poss in layers.items():
                for pname, v in poss.items():
                    if v is not None:
                        traces.append((abs(v), comp, li, pname))
        traces = sorted(traces, reverse=True)[:top_k]

        ax.plot(steps, [r["patch"][corr]["total_effect"] for r in recs],
                color=INK_2, lw=1.4, label="ceiling: clean − corrupted")
        ax.axhline(0, color=AXIS, lw=0.8)
        for k, (_, comp, li, pname) in enumerate(traces):
            ys = [r["patch"][corr]["effect"][comp][li][pname] for r in recs]
            ys = [np.nan if y is None else y for y in ys]
            ax.plot(steps, ys, ls=LINE_STYLES[k % len(LINE_STYLES)],
                    marker=MARKERS[k % len(MARKERS)], ms=4,
                    markevery=max(len(steps) // 12, 1),
                    color=COMP_COLOR.get(comp, MUTED),
                    label=f"{comp} in {_layer_tag(int(li), ltypes[int(li)])}, "
                          f"patched at {POS_LABEL.get(pname, pname)}")
        ax.set_ylabel("Effect of patch (logits)")
        ax.set_title(CORRUPTION_TITLE.get(corr, corr))
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    axes[-1].set_xlabel("Training step")
    fig.suptitle(f"{_arch_label(arch) if arch else formation['run_name']} — "
                 "when does each part of the circuit form?",
                 fontweight="bold", x=0.01, ha="left")
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Q3: distance sweep
# ---------------------------------------------------------------------------

def plot_distance(
    sweeps: dict[str, dict],       # arch → run_distance_sweep output
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Left  : recall accuracy vs induction distance.
    Middle: the same with the SSM memory switched off in every Mamba layer —
            a model whose line does not move does not use the state to recall.
    Right : how strongly the state handed to the trigger depends on B.
    Hollow markers = sequence longer than anything seen in training.
    """
    archs = _sorted_archs(sweeps.keys())
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4.0),
                                        constrained_layout=True)

    def _line(ax, arch, key, getter=None):
        recs = [r for r in sweeps[arch]["records"] if key in r]
        if not recs:
            return
        ds = [r["distance"] for r in recs]
        ys = [getter(r) if getter else r[key] for r in recs]
        c, m = ARCH_COLOR.get(arch, MUTED), ARCH_MARKER.get(arch, "o")
        ax.plot(ds, ys, color=c, lw=1.8, label=_arch_label(arch))
        for r, y in zip(recs, ys):
            ax.plot(r["distance"], y, marker=m, ms=6, color=c, mew=1.6,
                    mfc=c if r["in_distribution"] else SURFACE)

    for arch in archs:
        _line(ax1, arch, "accuracy")
        _line(ax2, arch, "accuracy_no_memory")
        _line(ax3, arch, "state_signal",
              lambda r: max(v["signal_over_spread"] for v in r["state_signal"].values()))

    chance = next((sw.get("chance_accuracy") for sw in sweeps.values()), None)
    trained_len = next((sw.get("trained_seq_len") for sw in sweeps.values()), None)
    for ax in (ax1, ax2, ax3):
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Induction distance A₁ → A₂ (tokens)")
        if trained_len:
            ax.axvline(trained_len, color=AXIS, lw=1)
            ax.text(trained_len, 1.0, " training length", transform=ax.get_xaxis_transform(),
                    fontsize=8, color=MUTED, va="top")
    for ax in (ax1, ax2):
        ax.set_ylim(-0.05, 1.08)
        ax.set_ylabel("Recall accuracy")
        if chance:
            ax.axhline(chance, color=AXIS, lw=1)
            # x in axes coordinates → safe on a log axis
            ax.text(0.01, chance + 0.015, "chance", transform=ax.get_yaxis_transform(),
                    fontsize=8, color=MUTED)

    ax1.set_title("Recall vs distance")
    handles, labels = ax1.get_legend_handles_labels()
    ax3.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax2.set_title("…with SSM memory switched off")
    ax3.set_title("Does the carried SSM state depend on B?")
    ax3.set_ylabel("B-signal / between-sequence spread")
    ax3.set_ylim(bottom=0)

    fig.suptitle("Memory limits of in-context recall  "
                 "(hollow marker = longer than any training sequence)",
                 fontweight="bold", x=0.01, ha="left")
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Analysis figures (one panel per architecture with a Mamba layer)
# ---------------------------------------------------------------------------

def _arch_panels(analyses: dict[str, dict], width: float, height: float):
    archs = _sorted_archs(analyses.keys())
    fig, axes = plt.subplots(1, len(archs), figsize=(width * len(archs), height),
                             sharey=True, squeeze=False, constrained_layout=True)
    return fig, axes[0], archs


def plot_pathway_ablation(
    analyses: dict[str, dict],     # arch → analysis dict (needs 'pathway')
    chance: Optional[float] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Accuracy with the SSM memory and / or the conv switched off, per Mamba
    layer and for all Mamba layers together."""
    fig, axes, archs = _arch_panels(analyses, 4.3, 3.8)
    for ax, arch in zip(axes, archs):
        res = analyses[arch]["pathway"]
        layer_sets = []
        for r in res:
            if r["layers"] not in layer_sets:
                layer_sets.append(r["layers"])
        w = 0.8 / len(PATHWAY_ORDER)
        for k, cond in enumerate(PATHWAY_ORDER):
            ys = [next((r["accuracy"] for r in res
                        if r["layers"] == ls and r["condition"] == cond), np.nan)
                  for ls in layer_sets]
            ax.bar(np.arange(len(layer_sets)) + (k - 1.5) * w, ys, w * 0.86,
                   color=PATHWAY_COLOR[cond], label=PATHWAY_LABEL[cond])
        if chance:
            ax.axhline(chance, color=AXIS, lw=1)
            ax.text(0.01, chance + 0.015, "chance", transform=ax.get_yaxis_transform(),
                    fontsize=8, color=MUTED)
        ax.set_xticks(range(len(layer_sets)))
        ax.set_xticklabels(
            [f"L{ls[0]} only" if len(ls) == 1 else "all Mamba\nlayers" for ls in layer_sets]
        )
        ax.set_ylim(0, 1.08)
        ax.grid(axis="x", visible=False)
        ax.set_title(_arch_label(arch))
    axes[0].set_ylabel("Recall accuracy")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                    title="Ablated in the named layer(s)")
    fig.suptitle("Which cross-position pathway of the Mamba block does recall need?",
                 fontweight="bold", x=0.01, ha="left")
    _save(fig, save_path)
    return fig


def plot_probe_decoding(
    analyses: dict[str, dict],     # arch → analysis dict (needs 'probe')
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Cross-validated accuracy of decoding B's identity from the full SSM
    state, at four read positions, one line per Mamba layer."""
    fig, axes, archs = _arch_panels(analyses, 4.3, 3.8)
    names = ["before A₁\n(control)", "at B", "midway", "just\nbefore A₂"]
    for ax, arch in zip(axes, archs):
        probes = analyses[arch]["probe"]
        positions = sorted({p["position"] for p in probes})
        layers = sorted({p["layer"] for p in probes})
        xs = np.arange(len(positions))
        for k, li in enumerate(layers):
            acc = {p["position"]: p["accuracy"] for p in probes if p["layer"] == li}
            ax.plot(xs, [acc.get(p, np.nan) for p in positions],
                    color=SLOTS[k % len(SLOTS)], marker=MARKERS[k % len(MARKERS)],
                    ls=LINE_STYLES[k % len(LINE_STYLES)], label=f"Mamba L{li}")
        chance = probes[0]["chance"] if probes else None
        if chance:
            ax.axhline(chance, color=AXIS, lw=1)
            ax.text(0.01, chance + 0.015, "chance", transform=ax.get_yaxis_transform(),
                    fontsize=8, color=MUTED)
        ax.set_xticks(xs)
        ax.set_xticklabels(names if len(positions) == len(names)
                           else [f"pos {p}" for p in positions])
        ax.set_xlabel("Position where the state is read")
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(_arch_label(arch))
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Decoding accuracy of B (64-way)")
    fig.suptitle("Can B's identity be read out of the SSM state?",
                 fontweight="bold", x=0.01, ha="left")
    _save(fig, save_path)
    return fig


def plot_ablation(
    analyses: dict[str, dict],     # arch → analysis dict (needs 'ablation')
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Accuracy after silencing the two state dims that best separate B
    ('top') vs two random dims (control)."""
    fig, axes, archs = _arch_panels(analyses, 4.3, 3.8)
    color = {"top": SLOTS[0], "random": MUTED}
    for ax, arch in zip(axes, archs):
        res = analyses[arch]["ablation"]
        xs = np.arange(len(res))
        ax.bar(xs, [r["val_accuracy"] for r in res], 0.6,
               color=[color.get(r.get("kind", "top"), SLOTS[0]) for r in res])
        if res:
            base = res[0]["baseline_accuracy"]
            ax.axhline(base, color=INK_2, lw=1)
            ax.text(0.99, base + 0.015, "intact model", ha="right",
                    transform=ax.get_yaxis_transform(), fontsize=8, color=INK_2)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [f"L{r['layer']}\n{r.get('kind', 'top')}\n{r['ablated_dims']}" for r in res],
            fontsize=8,
        )
        ax.set_ylim(0, 1.12)
        ax.grid(axis="x", visible=False)
        ax.set_title(_arch_label(arch))
    axes[0].set_ylabel("Recall accuracy after ablation")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color["top"]),
               plt.Rectangle((0, 0), 1, 1, color=color["random"])]
    axes[-1].legend(handles, ["top-scoring state dims", "random state dims (control)"],
                    loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("Does silencing two SSM state dims break recall?",
                 fontweight="bold", x=0.01, ha="left")
    _save(fig, save_path)
    return fig


def plot_timescales(
    eigen: list[dict],      # EigenResult dicts for one architecture
    arch: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Memory timescale τ of each state dim after training vs at
    initialisation (same Δ). Points on the grey curve were not changed by
    training."""
    n = len(eigen)
    if n == 0:
        return plt.figure()
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 3.6), sharey=True,
                             squeeze=False, constrained_layout=True)
    for ax, er in zip(axes[0], eigen):
        dims = np.arange(len(er["decay_timescales"]))
        cap = 1e4      # display cap for τ = ∞
        tr = np.minimum(np.array(er["decay_timescales"], dtype=float), cap)
        init = np.minimum(np.array(er["init_timescales"], dtype=float), cap)
        ax.plot(dims, init, color=MUTED, lw=1.2, marker="o", ms=5, mfc=SURFACE,
                label="at initialisation")
        ax.plot(dims, tr, color=SLOTS[0], lw=0, marker="o", ms=6, label="after training")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("State dim (longest memory first)")
        ax.set_title(f"Mamba L{er['layer']}  (mean Δ = {er['mean_delta']:.3f})")
        ax.legend(loc="upper right")
    axes[0][0].set_ylabel("Memory timescale τ (steps)")
    title = "SSM memory timescales: did training change them?"
    fig.suptitle(f"{_arch_label(arch)} — {title}" if arch else title,
                 fontweight="bold", x=0.01, ha="left")
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Across seeds
# ---------------------------------------------------------------------------

def plot_seed_training(
    logs: dict[str, dict[str, list[dict]]],    # arch → seed_tag → train log
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Validation accuracy vs training step: one line per (architecture,
    seed). The spread between same-coloured lines is the seed-to-seed
    variation in when induction appears."""
    fig, ax = plt.subplots(figsize=(8.2, 4.0), constrained_layout=True)
    for arch in _sorted_archs(logs.keys()):
        for k, (tag, log) in enumerate(sorted(logs[arch].items())):
            ax.plot([e["step"] for e in log], [e["val_acc"] for e in log],
                    color=ARCH_COLOR.get(arch, MUTED), lw=1.5, alpha=0.85,
                    marker=ARCH_MARKER.get(arch, "o"), ms=4,
                    markevery=max(len(log) // 10, 1),
                    label=f"{_arch_label(arch)} — {len(logs[arch])} seed(s)"
                          if k == 0 else None)
    ax.set_xscale("log")
    ax.set_xlabel("Training step (log scale)")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title("When does induction appear?  One line per seed")
    _save(fig, save_path)
    return fig


def make_seed_summary(results_dir: str = "results", figures_dir: str = "figures") -> None:
    """figure_seeds_training.png from every results/<arch>_<seed_tag>_train_log.json."""
    rdir, fdir = Path(results_dir), Path(figures_dir)
    fdir.mkdir(parents=True, exist_ok=True)
    logs: dict[str, dict[str, list[dict]]] = {}
    for pth in sorted(rdir.glob("*_train_log.json")):
        run_name = pth.stem.replace("_train_log", "")
        arch, _, tag = run_name.rpartition("_")
        logs.setdefault(arch, {})[tag] = json.loads(pth.read_text())["log"]
    if logs:
        fig = plot_seed_training(logs, save_path=str(fdir / "figure_seeds_training.png"))
        plt.close(fig)
        print("Saved figure_seeds_training.png")


# ---------------------------------------------------------------------------
# Convenience: generate all per-seed figures from saved results
# ---------------------------------------------------------------------------

def make_all_figures(results_dir: str = "results", figures_dir: str = "figures") -> None:
    rdir = Path(results_dir)
    fdir = Path(figures_dir)
    fdir.mkdir(parents=True, exist_ok=True)
    from patch import CORRUPTIONS, load_patch_results

    # ── Q1: per-architecture patching heatmaps + cross-arch comparison ──
    meta_path = rdir / "patch_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    arch_pgs: dict[str, dict] = {}
    for pth in sorted(rdir.glob("patch_*_*.json")):
        arch, _, corr = pth.stem.replace("patch_", "", 1).rpartition("_")
        if corr in CORRUPTIONS:
            arch_pgs.setdefault(arch, {})[corr] = load_patch_results(pth)
    for arch in _sorted_archs(arch_pgs.keys()):
        pgs = {c: arch_pgs[arch][c] for c in CORRUPTIONS if c in arch_pgs[arch]}
        arch_pgs[arch] = pgs
        any_pg = next(iter(pgs.values()))
        first_A = meta.get("first_A_pos", any_pg.seq_len - 1 - (any_pg.seq_len - 16))
        fig = plot_patching_heatmap(
            pgs, arch, first_A_pos=first_A,
            save_path=str(fdir / f"figure_patching_heatmap_{arch}.png"),
        )
        plt.close(fig)
        print(f"Saved figure_patching_heatmap_{arch}.png")

    if arch_pgs and meta:
        fig = plot_arch_comparison(
            arch_pgs, first_A_pos=meta["first_A_pos"],
            save_path=str(fdir / "figure_arch_comparison.png"),
        )
        plt.close(fig)
        print("Saved figure_arch_comparison.png")

    # ── Q2: formation timelines ──
    for pth in sorted(rdir.glob("formation_*.json")):
        arch = pth.stem.replace("formation_", "")
        formation = json.loads(pth.read_text())
        fig = plot_formation(formation, arch=arch,
                             save_path=str(fdir / f"figure_formation_{arch}.png"))
        plt.close(fig)
        print(f"Saved figure_formation_{arch}.png")

    # ── Q3: distance sweep ──
    sweeps = {}
    for pth in sorted(rdir.glob("distance_*.json")):
        sweeps[pth.stem.replace("distance_", "")] = json.loads(pth.read_text())
    if sweeps:
        fig = plot_distance(sweeps, save_path=str(fdir / "figure_distance.png"))
        plt.close(fig)
        print("Saved figure_distance.png")

    # ── Timescales / probes / ablations from analysis_<arch>.json ──
    analyses = {}
    for pth in sorted(rdir.glob("analysis_*.json")):
        analyses[pth.stem.replace("analysis_", "")] = json.loads(pth.read_text())
    if analyses:
        chance = next((sw.get("chance_accuracy") for sw in sweeps.values()), None)
        for name, plot, kwargs in [
            ("figure_pathway_ablation.png", plot_pathway_ablation, {"chance": chance}),
            ("figure_probe_decoding.png", plot_probe_decoding, {}),
            ("figure_ablation.png", plot_ablation, {}),
        ]:
            fig = plot(analyses, save_path=str(fdir / name), **kwargs)
            plt.close(fig)
            print(f"Saved {name}")
        for arch, data in analyses.items():
            if data.get("eigen"):
                fig = plot_timescales(data["eigen"], arch=arch,
                                      save_path=str(fdir / f"figure_timescales_{arch}.png"))
                plt.close(fig)
                print(f"Saved figure_timescales_{arch}.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/s42")
    parser.add_argument("--figures_dir", default="figures/s42")
    args = parser.parse_args()
    make_all_figures(args.results_dir, args.figures_dir)
