"""Hard-gate sanity checks: H2 (rollout boundedness) and H3 (action conditioning)."""
from __future__ import annotations

from typing import Callable

import torch


def rollout(
    model_step: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    state_t: torch.Tensor,
    action_global: torch.Tensor,
    *,
    k: int,
) -> torch.Tensor:
    """Autoregressive k-step rollout. Returns (B, k+1, N, C) including initial state."""
    states = [state_t]
    s = state_t
    for _ in range(k):
        s = model_step(s, action_global)
        states.append(s)
    return torch.stack(states, dim=1)


def rollout_boundedness_ratio(
    rollout_states: torch.Tensor, train_envelope_inf: float,
) -> float:
    last = rollout_states[:, -1]
    inf_norm = last.abs().amax(dim=(1, 2)).mean().item()
    return float(inf_norm / train_envelope_inf)


def action_conditioning_diff_ratio(
    delta_a: torch.Tensor, delta_b: torch.Tensor,
) -> torch.Tensor:
    a = delta_a.flatten(1)
    b = delta_b.flatten(1)
    diff = (a - b).norm(dim=1)
    base = a.norm(dim=1) + 1e-8
    return diff / base
