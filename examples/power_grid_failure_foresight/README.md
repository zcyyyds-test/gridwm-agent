# Power Grid Failure Foresight

This example packages wm-agent as a system-level world model case study:

```text
grid state + fault event
-> latent graph world model
-> imagined future
-> risk signal
-> scenario / portfolio ranking
```

The point is not another one-step prediction benchmark. The point is a
decision-facing loop that mirrors 2025-2026 industrial world-model workflows:
generate candidate events, imagine futures, rank long-tail risk, and summarize
which event families dominate the high-risk frontier.

## System API

```python
from pathlib import Path

from wmagent.world.power_grid import PowerGridWorldModelSystem

world = PowerGridWorldModelSystem.from_run_dir(Path("outputs/run_3acec7d14d"))
state = world.anchor_state(split="val", anchor_index=0, horizon=10)
events = world.candidate_events(horizon=10)

future = world.imagine(state, events[0])
risk = world.score(future)
top = world.search(state, events, top_k=10)
```

## Reproduce Scenario Search

```bash
python scripts/rank_scenarios.py outputs/run_3acec7d14d \
    --top-k 10 \
    --out experiments/scenario_search/top_scenarios.json
```

Current artifact:

- [`top_scenarios.md`](../../experiments/scenario_search/top_scenarios.md)

## Reproduce Portfolio Sweep

```bash
python scripts/sweep_scenario_portfolio.py outputs/run_3acec7d14d \
    --n-anchors 8 \
    --top-k 15 \
    --out experiments/scenario_portfolio/portfolio_sweep.json
```

Current artifact:

- [`portfolio_sweep.md`](../../experiments/scenario_portfolio/portfolio_sweep.md)

## Employment Narrative

This example can be described as:

> Built an event-conditioned latent graph world model system that imagines
> counterfactual futures for graph-structured physical infrastructure and uses
> those futures for risk-aware scenario search and portfolio-level long-tail
> event discovery.
