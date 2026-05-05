# gridwm-agent Agent Benchmark

| Run | Pearson | Spearman | Discover hit | Safe hit | Risk lift | Risk reduction | Agent ms | Exhaustive ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V4.2-canonical-val | 0.6376 | 0.6175 | 0.6875 | 0.6875 | 0.7665 | 0.35 | 0.2537 | 634.9 |
| V4.2-canonical-test | 0.6704 | 0.6201 | 0.75 | 0.5 | 1.006 | 0.3585 | 0.1708 | 639.2 |
| V4.2-FT7-candidate-holdout | 0.2348 | 0.2998 | 0.5 | 0.3125 | 0.2771 | 0.2808 | 0.2202 | 500 |

`Agent ms` is the actor-critic forward pass per state. `Exhaustive ms` is
running the **same neural world model** for all 162 candidates per anchor;
this is *not* the cost of running a CloudPSS EMT simulation, which is
orders of magnitude slower. So `Exhaustive ms` is the cost the agent
saves at inference, not the cost of "ground truth physics".

The "FT-7 candidate-space holdout" row trains the agent's actor and
critic with FT-7 candidate events excluded; the underlying world model
is trained on a random 80/10/10 trajectory split that **does** include
FT-7 fault-type trajectories. So the world model is in-distribution on
FT-7 dynamics; only the agent's candidate-space coverage is held out.
This is not end-to-end OOD generalisation.

`n_anchors = 16` per row; 1000-resample bootstrap CIs on the headline
hit-rate metrics are stored alongside as
`bootstrap_ci_{val,test}.json`.

## Risk continuum vs reference points

This section previously contained a "discover gain over heuristic" table
computed against an earlier biased strawman (a `_ft7_heuristic` that
added +10 to fault_type==7, which made the strawman almost always pick
FT-7). The strawman has been replaced with an unbiased
`_physics_heuristic` (`fault_type + duration/180 + 0.05/resistance`,
no per-fault-type bonus); the table has been removed because the old
numbers no longer reflect the current code path.

The replacement heuristic is implemented in
`wmagent/agent/metrics.py::_physics_heuristic_indices`. Re-running
`scripts/eval_agent.py` against the bundled `outputs/demo_run` agent
will produce the corrected `physics_heuristic_mean_risk` field; until
those numbers are re-collected, please cite only the headline hit-rate
table at the top of this file.
