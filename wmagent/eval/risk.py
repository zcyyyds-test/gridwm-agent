"""Risk scoring utilities for world-model rollouts.

These functions intentionally sit outside the neural model. The V2 checkpoint
stays a pure latent dynamics model, while this layer turns imagined futures into
decision-facing signals for scenario search and portfolio demos.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from wmagent.data.schema import CHANNELS


@dataclass(frozen=True)
class RolloutRiskConfig:
    """Weights for converting an imagined trajectory into a scalar risk value."""

    max_delta_weight: float = 1.0
    mean_delta_weight: float = 0.35
    final_delta_weight: float = 0.25
    ramp_weight: float = 0.15
    eps: float = 1e-12


def rollout_risk_value(
    rollout: torch.Tensor,
    *,
    cfg: RolloutRiskConfig | None = None,
) -> torch.Tensor:
    """Return an unnormalized scalar risk value per rollout.

    Args:
        rollout: Tensor shaped ``(B, H+1, N, C)`` in either normalized or
            physical units. The score is scale-aware but ranking-oriented, so
            callers usually normalize it within a scenario batch.
    """
    if rollout.ndim != 4:
        raise ValueError(f"rollout must have shape (B,H+1,N,C); got {tuple(rollout.shape)}")
    cfg = cfg or RolloutRiskConfig()
    delta = rollout[:, 1:] - rollout[:, :1]
    abs_delta = delta.abs()
    max_delta = abs_delta.amax(dim=(1, 2, 3))
    mean_delta = abs_delta.mean(dim=(1, 2, 3))
    final_delta = (rollout[:, -1] - rollout[:, 0]).abs().mean(dim=(1, 2))
    final_delta = final_delta.mean(dim=-1)
    if rollout.shape[1] > 2:
        step_delta = rollout[:, 1:] - rollout[:, :-1]
        ramp = step_delta.abs().mean(dim=(1, 2, 3))
    else:
        ramp = torch.zeros_like(max_delta)
    return (
        cfg.max_delta_weight * max_delta
        + cfg.mean_delta_weight * mean_delta
        + cfg.final_delta_weight * final_delta
        + cfg.ramp_weight * ramp
    )


def normalize_risk_scores(
    values: torch.Tensor,
    *,
    floor: int = 35,
    ceil: int = 100,
) -> torch.Tensor:
    """Map raw risk values into an integer-like 0-100 display range."""
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D; got {tuple(values.shape)}")
    lo = values.min()
    hi = values.max()
    span = (hi - lo).clamp_min(1e-12)
    scaled = floor + (values - lo) / span * (ceil - floor)
    return scaled.round().clamp(0, 100)


def risk_band(score: float) -> str:
    if score >= 82:
        return "CRITICAL"
    if score >= 64:
        return "ELEVATED"
    return "WATCH"


def rollout_risk_features(
    rollout: torch.Tensor,
    *,
    scores: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    """Return decision-facing features for each rollout in a batch."""
    if rollout.ndim != 4:
        raise ValueError(f"rollout must have shape (B,H+1,N,C); got {tuple(rollout.shape)}")
    delta = rollout[:, 1:] - rollout[:, :1]
    abs_delta = delta.abs()
    raw_values = rollout_risk_value(rollout)
    scores = normalize_risk_scores(raw_values) if scores is None else scores
    out: list[dict[str, Any]] = []
    for b in range(rollout.shape[0]):
        flat_idx = int(abs_delta[b].argmax().item())
        _step, node_idx, ch_idx = torch.unravel_index(
            torch.tensor(flat_idx, device=rollout.device),
            abs_delta[b].shape,
        )
        score = float(scores[b].item())
        out.append(
            {
                "score": int(round(score)),
                "band": risk_band(score),
                "risk_value": float(raw_values[b].item()),
                "max_abs_delta": float(abs_delta[b].amax().item()),
                "mean_abs_delta": float(abs_delta[b].mean().item()),
                "final_abs_delta": float((rollout[b, -1] - rollout[b, 0]).abs().mean().item()),
                "dominant_step": int(_step.item()) + 1,
                "dominant_node": int(node_idx.item()),
                "dominant_channel": CHANNELS[int(ch_idx.item())],
            }
        )
    return out
