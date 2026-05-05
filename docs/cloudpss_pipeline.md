# CloudPSS Pipeline

This document records the public-safe CloudPSS practice used by wm-agent. It is
written for reproducibility and project packaging. Do not place CloudPSS
passwords, tokens, browser cookies, private model links, or account material in
this repository.

## Role In wm-agent

CloudPSS is the EMT simulation oracle. wm-agent uses it to create trajectories
from an IEEE 39-bus system, then trains a neural latent graph world model to
approximate short-horizon future dynamics under fault actions.

The resulting story is:

```text
CloudPSS EMT simulator -> HDF5 trajectory dataset -> latent graph world model
-> multi-step imagination rollout -> fault-window and action-sensitivity eval
```

## SDK Call Chain

The stable live workflow is:

1. `cloudpss.Model.fetch(rid)`
2. `model.updateComponent(fault_key, args=...)`
3. `set_emt_params(model, end_time, step_time, n_cpu)`
4. `cloudpss.Model.update(model)`
5. Find the EMT job with `"emtp"` in `job["rid"]`
6. `runner = model.run(emt_job, model.configs[0])`
7. `runner.result.waitFor()`
8. `runner.result.getPlots()`
9. `runner.result.getLogs()` for event sanity checks
10. `runner.close()` in best-effort cleanup

Two practical points matter:

- `Model.update()` is required after changing component or job arguments.
  Running directly after a local mutation can use stale server-side state.
- `result.waitFor()` must finish before reading plots or logs because EMT
  results are delivered asynchronously.

wm-agent implements this chain in:

- `wmagent/data/cloudpss_helpers.py`
- `wmagent/data/collector.py`
- `scripts/collect_data.py`

## Authentication Boundary

Supported inputs:

- `CLOUDPSS_TOKEN`
- `CLOUDPSS_USERNAME`
- `CLOUDPSS_PASSWORD`
- `CLOUDPSS_API_URL`
- `GRIDWM_IEEE39_RID`

`ensure_authenticated()` first tries `CLOUDPSS_TOKEN`; otherwise it logs in via
environment variables. Local wrappers may source a gitignored env file, but that
file must never be committed.

## Fault Action Space

V2 path-A uses the single fault element available in the IEEE39 model. The
collector samples:

| Field | Meaning | Range |
|---|---|---|
| `fs` | fault start time | uniform `[1.0, 5.0]` seconds |
| `duration` | `fe - fs` | uniform `[0.05, 0.20]` seconds |
| `fe` | fault clear time | `fs + duration` |
| `ft` | CloudPSS fault type code | one of `{1, 3, 7}` |
| `chg_ohm` | fault resistance | log-uniform `[0.01, 10.0]` ohm |

CloudPSS `ft` codes used here:

- `1`: A-ground fault
- `3`: AB fault
- `7`: ABC fault

The model sees this as a 12-d action vector:

```text
8-d ft one-hot + fs_norm + duration_norm + log10(chg)_norm + tau
```

`tau = (t - fs) / duration` is recomputed for every rollout step and clipped to
`[-2, 3]`. This gives the world model explicit phase information:

- `tau < 0`: pre-fault
- `0 <= tau <= 1`: active fault
- `tau > 1`: post-clear recovery

## EMT Settings

Default wm-agent collection settings are in `configs/data.yaml`:

| Setting | Value | Why |
|---|---:|---|
| `sim_end_time_s` | `15.0` | covers pre-fault, active fault, and recovery |
| `sim_step_time_s` | `5.0e-5` | 50 us EMT internal step |
| `output_dt_s` | `0.001` | 1 ms training/eval output cadence |
| `expected_n_samples` | `15001` | 15 s inclusive output at 1 ms |
| `sim_n_cpu` | `1` | IEEE39 is small; multi-core did not improve this job class |
| `concurrency` | `3` | conservative against update races and quota pressure |

Historical practice showed that batch concurrency can improve throughput, but
the live platform has hourly quota and server-side `Model.update` races under
aggressive parallelism. wm-agent therefore keeps collection reliable first and
uses retry/backoff for transient errors such as duplicate resource tags, quota
blips, timeouts, and temporary network failures.

## Extracted Channels

CloudPSS plots are localized, so the helper classifies by stable channel tags.
wm-agent keeps:

| Output | Shape | Meaning |
|---|---:|---|
| `state/wr` | `(T, 10)` | generator speed |
| `state/LA` | `(T, 10)` | load angle in radians |
| `state/VT` | `(T, 10)` | terminal voltage |
| `state/IT` | `(T, 10)` | terminal current |
| `bus_obs` | `(T, 3)` | bus-37 three-phase current observation |

`RA[Deg]` is intentionally dropped because it is redundant with `LA[Rad]`.

## HDF5 Schema

Each trajectory is one HDF5 group:

```text
sample_<uid>/
  meta/
    attrs:
      case
      seed
      cloudpss_rid
      cloudpss_version
      data_version
      output_dt_seconds
      topology_static_during_fault
      collected_at
  action/
    attrs:
      fs
      fe
      ft
      chg_ohm
  state/
    wr: (T, 10) float32
    LA: (T, 10) float32
    VT: (T, 10) float32
    IT: (T, 10) float32
  bus_obs: (T, 3) float32
```

The schema is implemented in `wmagent/data/schema.py` and validated before a
sample is written.

## Collection Command

On a machine with the CloudPSS SDK and secrets already configured:

```bash
bash scripts/run_collection_wsl.sh
```

Direct Python entry:

```bash
export GRIDWM_IEEE39_RID="..."
export CLOUDPSS_TOKEN="..."
python scripts/collect_data.py
```

Use `CLOUDPSS_USERNAME` and `CLOUDPSS_PASSWORD` only when a token is not
available. Do not paste real values into logs, docs, commits, issues, or model
prompts.

## Why Path-A Is Enough For V2

Path-A deliberately keeps topology changes fixed and varies the fault action on
the known fault component. That makes V2 a clean action-conditioned world-model
problem:

- same grid prior
- diverse fault timing, type, and resistance
- high-fidelity EMT state response
- stable train/val/test splits
- direct comparison against zero, linear, and MPNN baselines

Future work can add topology extraction, multiple fault locations, stochastic
latent uncertainty, and safety heads. V2 focuses on a deterministic recurrent
latent model that can be trained, evaluated, and presented quickly.

## Reused Practice From Earlier CloudPSS Work

Earlier CloudPSS experiments in the surrounding workspace contain two ideas
that are useful for wm-agent, but they should remain extensions instead of
blocking the V2 path:

- **Energy-method validation**: a classical dissipated-energy heuristic can
  localize faults from voltage, angle, active-power, and reactive-power traces.
  For wm-agent V2.1, this can become a downstream sanity check: if imagined
  rollouts preserve the same energy-based fault signature as CloudPSS, the
  latent world model is not merely matching pointwise MAE.
- **Noise and missing-data augmentation**: older dataset builders injected row
  noise and missing samples before training classifiers. For wm-agent, the better
  use is a robustness eval split, not the first V2 training run. A clean EMT
  rollout model should be established first, then measured under sensor noise
  and sparse observation stress.

## Known Failure Modes

- Missing `Model.update()` after `updateComponent()` can produce stale runs.
- Reading plots before `result.waitFor()` can produce incomplete outputs.
- Over-aggressive concurrency can trigger server-side duplicate resource races
  or quota throttling.
- A missing or changed plot title will fail channel extraction; this is
  intentional because silent channel drift would corrupt the dataset.
- Windows server training should keep DataLoader workers low (`0` or `2`) to
  avoid process and commit-limit pressure.

## Packaging Notes

For resume and project presentation, describe the CloudPSS side as:

> A high-fidelity EMT oracle that generates counterfactual fault rollouts for
> training and evaluating a latent graph world model.

Avoid claiming real-time deployment or protection-system certification. The
strong claim is simulation-grounded latent rollout under counterfactual fault
actions, which is exactly what the current project implements.
