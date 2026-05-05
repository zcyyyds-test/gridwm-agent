# Power Grid Failure Foresight

This example packages gridwm-agent as a system-level world model case study:

```text
grid state + fault event
-> latent recurrent dynamics model
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

world = PowerGridWorldModelSystem.from_run_dir(Path("outputs/demo_run"))
state = world.anchor_state(split="val", anchor_index=0, horizon=10)
events = world.candidate_events(horizon=10)

future = world.imagine(state, events[0])
risk = world.score(future)
top = world.search(state, events, top_k=10)
```

## Reproduce Scenario Search

```bash
python scripts/rank_scenarios.py outputs/demo_run \
    --top-k 10 \
    --out experiments/scenario_search/top_scenarios.json
```

Current artifact:

- [`top_scenarios.md`](../../experiments/scenario_search/top_scenarios.md)

## Reproduce Portfolio Sweep

```bash
python scripts/sweep_scenario_portfolio.py outputs/demo_run \
    --n-anchors 8 \
    --top-k 15 \
    --out experiments/scenario_portfolio/portfolio_sweep.json
```

Current artifact:

- [`portfolio_sweep.md`](../../experiments/scenario_portfolio/portfolio_sweep.md)

