"""Phase 1 graph dynamics module: PyG MPNN with residual prediction (path-A)."""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINEConv

from wmagent.models.base import StaticGraph, WorldModel


class _GINEBlock(nn.Module):
    def __init__(self, hidden: int, edge_dim: int) -> None:
        super().__init__()
        mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.conv = GINEConv(mlp, edge_dim=edge_dim)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, edge_index, edge_attr):
        h = self.conv(x, edge_index, edge_attr)
        return self.norm(x + h)


class MPNNDynamics(WorldModel):
    def __init__(
        self,
        *,
        in_channels: int,
        hidden: int,
        n_layers: int,
        action_dim: int,
        node_attr_dim: int,
        edge_attr_dim: int,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.in_channels = in_channels

        self.node_in = nn.Linear(in_channels + node_attr_dim, hidden)
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
        )
        self.edge_in = nn.Linear(edge_attr_dim, hidden)

        self.blocks = nn.ModuleList(
            [_GINEBlock(hidden=hidden, edge_dim=hidden) for _ in range(n_layers)]
        )
        self.residual_head = nn.Linear(hidden, in_channels)

    def predict_delta(
        self,
        state: torch.Tensor,
        *,
        action_global: torch.Tensor,
        graph: StaticGraph,
    ) -> torch.Tensor:
        B, N, C = state.shape
        device = state.device

        node_attr = graph.node_attr.to(device).unsqueeze(0).expand(B, -1, -1)
        node_in = torch.cat([state, node_attr], dim=-1)
        h = self.node_in(node_in)                         # (B, N, H)

        a = self.action_mlp(action_global).unsqueeze(1).expand(-1, N, -1)
        h = h + a

        edge_attr = graph.edge_attr.to(device)            # (E, F_e)
        edge_h_single = self.edge_in(edge_attr)           # (E, H)
        edge_index_single = graph.edge_index.to(device)   # (2, E)

        # Batched graph: stack B disjoint copies of the static graph along the
        # node axis. Edge indices for batch b are offset by b * N.
        E = edge_index_single.shape[1]
        offsets = torch.arange(B, device=device).repeat_interleave(E) * N
        edge_index_batched = (
            edge_index_single.unsqueeze(0).expand(B, -1, -1).reshape(2, B * E)
            + offsets.unsqueeze(0)
        )
        edge_h_batched = edge_h_single.unsqueeze(0).expand(B, -1, -1).reshape(B * E, -1)

        x = h.reshape(B * N, -1)
        for blk in self.blocks:
            x = blk(x, edge_index_batched, edge_h_batched)
        h_out = x.reshape(B, N, -1)

        return self.residual_head(h_out)
