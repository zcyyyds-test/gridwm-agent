"""Actor/critic policy that plans over imagined wm-agent futures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


AgentMode = Literal["discover", "safe"]


@dataclass(frozen=True)
class AgentOutput:
    discover_logits: torch.Tensor
    safe_logits: torch.Tensor
    values: torch.Tensor


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class DreamerStyleAgent(nn.Module):
    """True agent layer: actor heads choose events; critic predicts imagined risk."""

    def __init__(
        self,
        *,
        in_channels: int,
        action_dim: int,
        horizon: int,
        n_candidates: int | None = None,
        n_nodes: int | None = None,
        hidden: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.n_candidates = None if n_candidates is None else int(n_candidates)
        self.n_nodes = None if n_nodes is None else int(n_nodes)
        state_dim = self.in_channels * 4
        event_dim = self.action_dim * self.horizon
        self.state_encoder = _mlp(state_dim, hidden, hidden, dropout)
        self.full_state_encoder = (
            _mlp(self.n_nodes * self.in_channels, hidden, hidden, dropout)
            if self.n_nodes is not None
            else None
        )
        self.context_encoder = _mlp(self.action_dim, hidden, hidden, dropout)
        self.event_encoder = _mlp(event_dim, hidden, hidden, dropout)
        joint_dim = hidden * 3
        self.actor_discover = _mlp(joint_dim, hidden, 1, dropout)
        self.actor_safe = _mlp(joint_dim, hidden, 1, dropout)
        self.critic = _mlp(joint_dim, hidden, 1, dropout)
        if self.n_candidates is None:
            self.register_parameter("discover_prior", None)
            self.register_parameter("safe_prior", None)
            self.register_parameter("value_prior", None)
        else:
            # The benchmark action space is a fixed candidate vocabulary. These
            # priors let the agent learn event-level risk tendencies, while the
            # state/event encoders still provide anchor-specific corrections.
            self.discover_prior = nn.Parameter(torch.zeros(self.n_candidates))
            self.safe_prior = nn.Parameter(torch.zeros(self.n_candidates))
            self.value_prior = nn.Parameter(torch.zeros(self.n_candidates))

    def state_features(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3:
            raise ValueError(f"state must have shape (B,N,C); got {tuple(state.shape)}")
        mean = state.mean(dim=1)
        std = state.std(dim=1, unbiased=False)
        min_v = state.amin(dim=1)
        max_v = state.amax(dim=1)
        return torch.cat([mean, std, min_v, max_v], dim=-1)

    def forward(
        self,
        state: torch.Tensor,
        action_sequences: torch.Tensor,
        *,
        context_actions: torch.Tensor | None = None,
    ) -> AgentOutput:
        if action_sequences.ndim == 3:
            action_sequences = action_sequences.unsqueeze(0).expand(state.shape[0], -1, -1, -1)
        if action_sequences.ndim != 4:
            raise ValueError(
                "action_sequences must have shape (K,H,A) or (B,K,H,A); "
                f"got {tuple(action_sequences.shape)}"
            )
        if action_sequences.shape[0] != state.shape[0]:
            raise ValueError("state batch and action batch must match")
        bsz, n_candidates, horizon, action_dim = action_sequences.shape
        if self.n_candidates is not None and n_candidates != self.n_candidates:
            raise ValueError(
                f"expected {self.n_candidates} candidates; got {n_candidates}"
            )
        if horizon != self.horizon or action_dim != self.action_dim:
            raise ValueError(
                f"expected actions (*,{self.horizon},{self.action_dim}); "
                f"got (*,{horizon},{action_dim})"
            )
        state_emb = self.state_encoder(self.state_features(state))
        if self.full_state_encoder is not None:
            if state.shape[1] != self.n_nodes:
                raise ValueError(f"expected {self.n_nodes} nodes; got {state.shape[1]}")
            state_emb = state_emb + self.full_state_encoder(state.reshape(bsz, -1))
        if context_actions is not None:
            if context_actions.shape != (bsz, self.action_dim):
                raise ValueError(
                    f"context_actions must have shape {(bsz, self.action_dim)}; "
                    f"got {tuple(context_actions.shape)}"
                )
            state_emb = state_emb + self.context_encoder(context_actions)
        event_flat = action_sequences.reshape(bsz * n_candidates, horizon * action_dim)
        event_emb = self.event_encoder(event_flat).reshape(bsz, n_candidates, -1)
        state_expanded = state_emb.unsqueeze(1).expand(-1, n_candidates, -1)
        joint = torch.cat(
            [state_expanded, event_emb, state_expanded * event_emb],
            dim=-1,
        )
        flat = joint.reshape(bsz * n_candidates, -1)
        discover_logits = self.actor_discover(flat).reshape(bsz, n_candidates)
        safe_logits = self.actor_safe(flat).reshape(bsz, n_candidates)
        values = self.critic(flat).reshape(bsz, n_candidates)
        if self.n_candidates is not None:
            discover_logits = discover_logits + self.discover_prior.view(1, -1)
            safe_logits = safe_logits + self.safe_prior.view(1, -1)
            values = values + self.value_prior.view(1, -1)
        return AgentOutput(
            discover_logits=discover_logits,
            safe_logits=safe_logits,
            values=values,
        )

    @torch.no_grad()
    def select(
        self,
        state: torch.Tensor,
        action_sequences: torch.Tensor,
        *,
        mode: AgentMode,
        context_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.forward(state, action_sequences, context_actions=context_actions)
        if mode == "discover":
            return (output.discover_logits + output.values).argmax(dim=1)
        if mode == "safe":
            return (output.safe_logits - output.values).argmax(dim=1)
        raise ValueError(f"unknown agent mode: {mode}")
