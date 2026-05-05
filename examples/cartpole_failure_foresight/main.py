"""CartPole failure foresight smoke example.

Mirrors :mod:`examples.power_grid_failure_foresight.main` shape, but runs on
the CartPole second-domain adapter -- the same ``WorldModelSystem`` contract,
zero learned ML model, zero CloudPSS, no GPU. Demonstrates that the
``imagine / score / search`` loop is domain-general.
"""
from __future__ import annotations

from wmagent.world.cartpole import CartPoleWorldModelSystem


def main() -> None:
    horizon = 5
    system = CartPoleWorldModelSystem.from_defaults(horizon=horizon)
    state = system.anchor_state(seed=2026, anchor_index=3)
    events = system.candidate_events()

    print(
        f"domain={system.domain}  "
        f"candidates={len(events)}  "
        f"horizon={horizon}"
    )
    print(f"anchor state (cart_pos, cart_vel, pole_angle, pole_vel) = "
          f"{state.tensor.tolist()}")
    print()
    print("Top-3 risky 5-step action sequences (1 = push right, 0 = push left):")
    print()
    print(f"{'rank':>4}  {'code':>5}  {'risk':>6}  {'band':>4}  "
          f"{'max_angle_deg':>13}  {'max_pos_m':>9}  {'term':>5}")
    print("-" * 64)
    for result in system.search(state, events, top_k=3):
        meta = result.future.event.metadata
        feat = result.risk.features
        angle_deg = feat["max_angle_rad"] * 180.0 / 3.141592653589793
        term = "yes" if feat["terminated"] else "no"
        print(
            f"{result.rank:>4}  {meta['code']:>5}  "
            f"{result.risk.value:>6.2f}  {result.risk.band:>4}  "
            f"{angle_deg:>13.2f}  {feat['max_pos_m']:>9.3f}  {term:>5}"
        )


if __name__ == "__main__":
    main()
