"""CartPole adapter for the wm-agent world-model system API.

This second-domain adapter exists to demonstrate that the
``WorldState / WorldEvent / ImaginedFuture / RiskSignal`` contract from
:mod:`wmagent.world.base` is domain-general. The same ``imagine / score /
search`` loop that runs on the trained power-grid V2 latent graph world
model also runs here, with zero changes to ``wmagent.agent`` or
``wmagent.eval``.

The "world model" used here is the CartPole-v1 dynamics itself (Euler
integration, parameters lifted from gymnasium's reference implementation).
This is intentional: the point of the adapter is contract conformance, not
modeling. Running on a perfect-information oracle keeps the example fast
(no GPU, no checkpoint, no learned model) and lets the contract test
compare an imagined trajectory against the analytic ground truth.

Pure Python on top of numpy / torch -- no gymnasium dependency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from wmagent.world.base import (
    ImaginedFuture,
    RiskSignal,
    SearchResult,
    WorldEvent,
    WorldState,
)


@dataclass(frozen=True)
class CartPoleParams:
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5
    force_mag: float = 10.0
    tau: float = 0.02
    theta_threshold_radians: float = 12.0 * 2.0 * math.pi / 360.0
    x_threshold: float = 2.4

    @property
    def total_mass(self) -> float:
        return self.masscart + self.masspole

    @property
    def polemass_length(self) -> float:
        return self.masspole * self.length


def _cartpole_step(state: np.ndarray, action: int, p: CartPoleParams) -> np.ndarray:
    x, x_dot, theta, theta_dot = state
    force = p.force_mag if action == 1 else -p.force_mag
    costheta = math.cos(float(theta))
    sintheta = math.sin(float(theta))
    temp = (force + p.polemass_length * theta_dot ** 2 * sintheta) / p.total_mass
    thetaacc = (p.gravity * sintheta - costheta * temp) / (
        p.length * (4.0 / 3.0 - p.masspole * costheta ** 2 / p.total_mass)
    )
    xacc = temp - p.polemass_length * thetaacc * costheta / p.total_mass
    return np.array(
        [
            x + p.tau * x_dot,
            x_dot + p.tau * xacc,
            theta + p.tau * theta_dot,
            theta_dot + p.tau * thetaacc,
        ],
        dtype=np.float32,
    )


def _ensure_domain(obj: WorldState | WorldEvent, *, expected: str) -> None:
    if obj.domain != expected:
        raise ValueError(f"expected domain={expected!r}, got {obj.domain!r}")


class CartPoleWorldModelSystem:
    """Second-domain adapter that proves the WorldModelSystem contract is reusable.

    Same ``imagine / score / search`` interface as
    :class:`wmagent.world.power_grid.PowerGridWorldModelSystem`, completely
    different domain. No learned model, no graph, no normalizer.
    """

    domain = "cartpole"

    def __init__(
        self,
        params: CartPoleParams | None = None,
        *,
        horizon: int = 5,
    ) -> None:
        self.params = params or CartPoleParams()
        self.horizon = int(horizon)

    @classmethod
    def from_defaults(cls, *, horizon: int = 5) -> "CartPoleWorldModelSystem":
        return cls(horizon=horizon)

    def anchor_state(self, *, seed: int = 2026, anchor_index: int = 0) -> WorldState:
        rng = np.random.default_rng(seed + anchor_index)
        obs = rng.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        return WorldState(
            tensor=torch.from_numpy(obs),
            domain=self.domain,
            metadata={"seed": int(seed + anchor_index), "anchor_index": int(anchor_index)},
        )

    def candidate_events(self, *, horizon: int | None = None) -> list[WorldEvent]:
        h = self.horizon if horizon is None else int(horizon)
        events: list[WorldEvent] = []
        for code in range(2 ** h):
            actions = [(code >> step) & 1 for step in range(h)]
            tensor = torch.tensor(actions, dtype=torch.long)
            events.append(
                WorldEvent(
                    tensor=tensor,
                    domain=self.domain,
                    metadata={
                        "code": format(code, f"0{h}b"),
                        "horizon": h,
                        "n_pushes_right": int(sum(actions)),
                    },
                )
            )
        return events

    def imagine(self, state: WorldState, event: WorldEvent) -> ImaginedFuture:
        return self.imagine_many(state, [event])[0]

    def imagine_many(
        self,
        state: WorldState,
        events: list[WorldEvent],
    ) -> list[ImaginedFuture]:
        _ensure_domain(state, expected=self.domain)
        for event in events:
            _ensure_domain(event, expected=self.domain)
        anchor = state.tensor.detach().cpu().numpy().astype(np.float32)
        futures: list[ImaginedFuture] = []
        for event in events:
            actions = event.tensor.detach().cpu().tolist()
            traj = [anchor.copy()]
            current = anchor.copy()
            for action in actions:
                current = _cartpole_step(current, int(action), self.params)
                traj.append(current.copy())
            rollout = torch.from_numpy(np.stack(traj, axis=0))
            futures.append(ImaginedFuture(rollout=rollout, state=state, event=event))
        return futures

    def score(self, future: ImaginedFuture) -> RiskSignal:
        rollout_np = future.rollout.detach().cpu().numpy()
        max_angle = float(np.max(np.abs(rollout_np[:, 2])))
        max_pos = float(np.max(np.abs(rollout_np[:, 0])))
        angle_ratio = max_angle / self.params.theta_threshold_radians
        pos_ratio = max_pos / self.params.x_threshold
        raw = float(max(angle_ratio, pos_ratio))
        violations = (
            (np.abs(rollout_np[:, 2]) > self.params.theta_threshold_radians)
            | (np.abs(rollout_np[:, 0]) > self.params.x_threshold)
        )
        terminated = bool(violations.any())
        first_violation = int(np.argmax(violations)) if terminated else -1
        if raw >= 1.0:
            band = "high"
        elif raw >= 0.5:
            band = "med"
        else:
            band = "low"
        score = int(min(100, max(0, round(raw * 100))))
        return RiskSignal(
            score=score,
            band=band,
            value=raw,
            features={
                "max_angle_rad": max_angle,
                "max_pos_m": max_pos,
                "angle_ratio": float(angle_ratio),
                "pos_ratio": float(pos_ratio),
                "terminated": terminated,
                "first_violation_step": first_violation,
            },
        )

    def search(
        self,
        state: WorldState,
        events: list[WorldEvent],
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not events:
            return []
        futures = self.imagine_many(state, events)
        scored = [(future, self.score(future)) for future in futures]
        scored.sort(key=lambda fs: fs[1].value, reverse=True)
        return [
            SearchResult(rank=rank, future=future, risk=risk)
            for rank, (future, risk) in enumerate(scored[:top_k], start=1)
        ]
