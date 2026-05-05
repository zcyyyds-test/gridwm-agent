# gridwm-agent Eval Comparison

| Run | Model | Ckpt epoch | Fault pairs | Mean fault/zero | wr | LA | VT | IT | Mean direction | Mean fault Pearson | Mean rollout Pearson@10 | H2 | H3 | Source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| Stage2-MPNN | `mpnn` | 36 | 304 | 1.013 | 1.219 | 1.017 | 0.7807 | 1.035 | 0.5929 | 0.3833 | 0.9988 | no | yes | `run_91ace18288` |
| V2-LatentGraphWM | `latent_graph_wm` | 49 | 304 | 0.529 | 0.5541 | 0.7026 | 0.3203 | 0.5389 | 0.8091 | 0.805 | 0.9994 | yes | yes | `run_3acec7d14d` |

Notes:
- Fault/zero below 1.0 means the model beats the zero-delta baseline on fault-window pairs.
- H2 is rollout boundedness; H3 is action perturbation sensitivity.
- Linear extrapolation is intentionally not the headline baseline because short-step EMT trajectories are very smooth, which makes one-step linear extrapolation deceptively strong on rollout Pearson without learning any fault-window dynamics.
