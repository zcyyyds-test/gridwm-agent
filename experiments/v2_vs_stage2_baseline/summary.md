# V2 vs Stage2 Baseline

## Core Question

Does the V2 latent recurrent dynamics model add measurable world-model
behavior over the v0.1 Stage2 MPNN baseline on the same CloudPSS IEEE39
validation split?

## Experiment Groups

| Group | Run | Model | Purpose |
|---|---|---|---|
| zero | built into eval | zero-delta baseline | sanity floor for fault-window dynamics |
| linear | built into eval | one-step linear extrapolation | strong smooth-trajectory reference |
| v0.1-stage2-baseline | `outputs/run_91ace18288` (not bundled) | MPNN one-step dynamics | old baseline tagged as `v0.1-stage2-baseline` |
| V2 | `outputs/demo_run` (was `run_3acec7d14d` internally) | latent recurrent dynamics | action-conditioned recurrent latent rollout |

## Fixed Controls

- Dataset: same collected CloudPSS IEEE39 HDF5 set.
- Split: same `data/splits.json`.
- Fault-window eval: same current `scripts/eval.py` metrics.
- Device: Intel single GPU, `CUDA_VISIBLE_DEVICES=0`.

## Success Frame

V2 does not need to win every one-step smooth-trajectory metric. The
behavioural target is:

- fault-window model/zero ratio below 1.0 across all channels;
- better multi-step rollout correlation and boundedness;
- clear action sensitivity under counterfactual fault perturbations.

## Result

| Run | Model | Mean fault/zero | Mean direction | Mean fault Pearson | Mean rollout Pearson@10 | H2 | H3 |
|---|---|---:|---:|---:|---:|---|---|
| Stage2-MPNN | `mpnn` | 1.013 | 0.5929 | 0.3833 | 0.9988 | no | yes |
| V2-LatentGraphWM | `latent_graph_wm` | 0.529 | 0.8091 | 0.805 | 0.9994 | yes | yes |

Interpretation:

- V2 cuts mean fault-window MAE versus zero baseline from roughly parity
  (`1.013`) to about half-zero (`0.529`).
- V2 improves mean direction accuracy from `0.5929` to `0.8091`.
- V2 improves mean fault-window waveform correlation from `0.3833` to `0.805`.
- Both models have high rollout Pearson because short-step EMT traces are
  smooth, so the stronger story is the combination of fault-window improvement,
  bounded rollout, and action sensitivity.

Conclusion: V2 is a meaningful world-model upgrade over the v0.1 Stage2
baseline — the gain is concentrated in fault-window dynamics, not in the
already-easy steady-state rollout.
