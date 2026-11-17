# Division of Labour in Hybrid Mamba–Transformer In-Context Recall

How do SSM and attention layers divide the labour of
in-context recall (induction) in a hybrid model?

The task: a sequence contains a bigram `A B` once, and ends with `A` again.
The model must predict `B`. Solving it takes two steps, and the project asks
which layer type does each one:

- **matching**: find the place where `A` occurred before (know that "the token before `B` was `A`");
- **moving the answer**: carry `B` from there to the final position.

| Q | Question | Phase | Output |
|---|----------|-------|--------|
| Q1 | **Localisation**: where does each half of the circuit live? | `patch` | `figure_patching_heatmap_*.png`, `figure_arch_comparison.png` |
| Q2 | **Formation**: when does each half form during training? | `formation` | `figure_formation_*.png` |
| Q3 | **Memory limits**: at what A₁…A₂ distance does recall fail, and is the SSM state what bridges the distance? | `distance` | `figure_distance.png` |
| | **Pathways**: does a Mamba layer contribute through its recurrent state or its convolution? | `analyze` | `figure_pathway_ablation.png`, `figure_probe_decoding.png`, `figure_ablation.png`, `figure_timescales_*.png` |

Four architectures, trained identically (d_model=128, 4 layers, same data):
`MAMA` (hybrid, Mamba-first), `AMAM` (order control), `MMMM` (pure SSM),
`AAAA` (pure attention).

All four are built from the same two blocks, so they differ only in layer
order. Attention layers get position from rotary embeddings (RoPE) in every
architecture. An attention block has ~132k parameters and a Mamba block ~117k.

## Run

```bash
pip install -r requirements.txt
python run_all.py --phase all --seeds 42            # one seed
python run_all.py --phase all --seeds 42 43 44      # three seeds
```

Phases can be run separately, in this order:
`train → patch → formation → distance → analyze → figures`.
`--arch hybrid_ma` restricts a phase to one architecture, and `--n_steps`
overrides the 20,000 training steps.

Outputs are kept per seed, so seeds never overwrite each other:

```
results/checkpoints/            model checkpoints (final, best, formation/)
results/<arch>_s42_train_log.json
results/s42/                    patch, formation, distance, analysis JSON
figures/s42/                    all figures for that seed
figures/figure_seeds_training.png   accuracy vs step, every seed overlaid
```

Every analysis loads the **final** checkpoint, so all architectures are
analysed after the same number of training steps.

Work already on disk is skipped, so running the same command again continues
an interrupted run. `--force` redoes everything.

### On Kaggle

A GPU is required: on CPU a full run takes more than a day. In the notebook
settings choose *Accelerator → GPU* and turn *Internet* on, then:

```python
!git clone https://github.com/BertaCarner/interpretability-hybrid-model.git
%cd interpretability-hybrid-model
!python run_all.py --phase all --seeds 42
!zip -qr /kaggle/working/outputs.zip results figures -x "results/checkpoints/*"
```

Check the first lines of the output before leaving it running:

- `PHASE 1 — Training hybrid_ma (MAMA) | seed 42 | cuda` must say `cuda`,
  not `cpu`;
- after 50 steps each training run prints how many minutes it still needs.

Use *Save Version → Save & Run All* so the session keeps running and
`outputs.zip` is stored with the notebook version. A version that hits
Kaggle's 12-hour limit is stopped before the zip step, so do not count on
getting its output.

If one session is not enough, run one architecture per notebook version by
adding `--arch hybrid_ma` (then `hybrid_am`, `mamba`, `attn`) to the
`run_all.py` line. Unzip the four `outputs.zip` files into the same folder
and build the cross-architecture figures there, which needs no GPU:

```bash
python run_all.py --phase figures --seeds 42
```

## How to read the main results

**Patching (Q1).** A clean sequence and a corrupted copy differ in one token.
One activation from the clean run is copied into the corrupted run, and
*recovery* is the fraction of the clean answer that comes back (1 = fully
restored, 0 = no effect). Two corruptions are used because each one exposes a
different half of the circuit:

- *Corrupt B* (replace `B` with another token): recovery shows the path that
  carries the answer.
- *Corrupt A₁* (replace the first `A`): the answer is still in the sequence
  but nothing marks it, so recovery shows the path that does the matching.

Patchable components: `resid` (block output, any layer), `h` (SSM state,
Mamba layers), `K`, `V`, `attn_out` (attention layers).

**Formation (Q2).** The plots show the patching effect in logits, not the
recovery fraction. Before the model has learned anything there is almost no
logit difference to recover, so a fraction would divide by nearly zero.

**Pathway ablation.** A Mamba block can move information between positions
in two ways: its recurrent state (long range) and its depthwise convolution
(the previous 3 tokens). Each is switched off in turn. If recall survives
without the state but not without the convolution, that layer is only doing
short-range work such as "which token came just before me".

**Distance (Q3).** Three panels: recall accuracy vs distance; the same with
the SSM state switched off; and how strongly the state that reaches the
final position depends on `B`.

## Files

- `model.py`: HybridLM with arbitrary M/A layer patterns; explicit SSM scan so h_t is patchable; switches to ablate SSM memory / conv.
- `data.py`: induction task with controllable A₁…A₂ distance.
- `train.py`: training + formation checkpoints + phase-transition detection.
- `patch.py`: batched cross-architecture activation patching, two corruption protocols.
- `formation.py`: Q2: patching effect at critical positions across checkpoints.
- `distance.py`: Q3: distance sweep, with and without SSM memory.
- `analyze.py`: memory timescales, decoding probes, state-dim and pathway ablations.
- `figures.py`: all figures.
- `val_profile.py`: per-distance accuracy table for one checkpoint.

## Limitations

- Small synthetic task and 4-layer models; nothing here is tested on a
  pretrained hybrid.
- Ablations set a pathway to zero in a trained model. A drop shows that this
  model needs the pathway, not that the task cannot be solved without it.
- Attention components are patched per layer, not per head.
- How fast induction appears depends on the position scheme. In a single-seed
  CPU test the pure-attention model reached 50% accuracy around step 5,800
  with RoPE and around step 4,300 with learned absolute positions
  (`pos_encoding="learned"` in `default_hybrid_config`). Compare formation
  speed across architectures with that in mind, and over several seeds.
- The PNGs currently in `results/` come from an earlier version of the
  pipeline (B-corruption only, learned positions for `AAAA` only) and do not
  match this code. Regenerated figures go to `figures/<seed>/`.
