"""Abstract WorldModel interface (path-A: single fault element, no per-edge flag)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class StaticGraph:
    edge_index: torch.Tensor  # (2, E) long
    edge_attr: torch.Tensor   # (E, F_e)
    node_attr: torch.Tensor   # (N, F_n)


class WorldModel(nn.Module):
    """Abstract base. Subclasses implement `predict_delta` returning Δ in
    normalized space. `forward` returns next-step normalized state via residual.
    """

    def predict_delta(
        self,
        state: torch.Tensor,
        *,
        action_global: torch.Tensor,
        graph: StaticGraph,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        state: torch.Tensor,
        *,
        action_global: torch.Tensor,
        graph: StaticGraph,
    ) -> torch.Tensor:
        return state + self.predict_delta(state, action_global=action_global, graph=graph)
