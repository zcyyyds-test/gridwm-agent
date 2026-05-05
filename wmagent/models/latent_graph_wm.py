"""Latent graph world model for action-conditioned EMT dynamics."""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINEConv

from wmagent.models.base import StaticGraph, WorldModel


def _batched_graph(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    *,
    batch_size: int,
    n_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.arange(batch_size, device=edge_index.device) * n_nodes
    edge_index_b = edge_index.unsqueeze(0) + offsets.view(batch_size, 1, 1)
    edge_attr_b = edge_attr.unsqueeze(0).expand(batch_size, -1, -1)
    return (
        edge_index_b.permute(1, 0, 2).reshape(2, -1),
        edge_attr_b.reshape(-1, edge_attr.shape[-1]),
    )


class _LatentGINEBlock(nn.Module):
    def __init__(self, hidden: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.conv = GINEConv(mlp, edge_dim=edge_dim)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        h = self.conv(x, edge_index, edge_attr)
        return self.norm(x + h)


class ActionConditioner(nn.Module):
    """Map action vectors into latent FiLM parameters."""

    def __init__(self, action_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.film = nn.Linear(hidden, hidden * 2)

    def forward(
        self, action_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.net(action_global)
        gamma, beta = self.film(emb).chunk(2, dim=-1)
        return emb, gamma, beta


class GraphStateEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        node_attr_dim: int,
        edge_attr_dim: int,
        hidden: int,
        latent_dim: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.node_in = nn.Linear(in_channels + node_attr_dim, hidden)
        self.edge_in = nn.Linear(edge_attr_dim, hidden)
        self.blocks = nn.ModuleList([
            _LatentGINEBlock(hidden=hidden, edge_dim=hidden, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.to_latent = nn.Linear(hidden, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        state: torch.Tensor,
        *,
        graph: StaticGraph,
        action_gamma: torch.Tensor,
        action_beta: torch.Tensor,
    ) -> torch.Tensor:
        B, _, _ = state.shape
        node_attr = graph.node_attr.to(state.device).unsqueeze(0).expand(B, -1, -1)
        h = self.node_in(torch.cat([state, node_attr], dim=-1))
        h = h * (1.0 + action_gamma.unsqueeze(1)) + action_beta.unsqueeze(1)

        edge_h = self.edge_in(graph.edge_attr.to(state.device))
        edge_index, edge_h = _batched_graph(
            graph.edge_index.to(state.device),
            edge_h,
            batch_size=B,
            n_nodes=state.shape[1],
        )
        x = h.reshape(B * state.shape[1], -1)
        for block in self.blocks:
            x = block(x, edge_index, edge_h)
        latent = self.to_latent(x.reshape(B, state.shape[1], -1))
        return self.norm(latent)


class LatentGraphDynamics(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        edge_attr_dim: int,
        action_dim: int,
        n_layers: int,
        attn_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.edge_in = nn.Linear(edge_attr_dim, latent_dim)
        self.action_to_latent = nn.Linear(action_dim, latent_dim)
        self.memory = nn.GRUCell(latent_dim, latent_dim)
        self.blocks = nn.ModuleList(
            [
                _LatentGINEBlock(hidden=latent_dim, edge_dim=latent_dim, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        latent: torch.Tensor,
        memory: torch.Tensor,
        *,
        action_emb: torch.Tensor,
        graph: StaticGraph,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D = latent.shape
        memory_next = self.memory(latent.reshape(B * N, D), memory.reshape(B * N, D))
        memory_next = memory_next.reshape(B, N, D)

        action_latent = self.action_to_latent(action_emb).unsqueeze(1)
        h = memory_next + action_latent
        edge_h = self.edge_in(graph.edge_attr.to(latent.device))
        edge_index, edge_h = _batched_graph(
            graph.edge_index.to(latent.device),
            edge_h,
            batch_size=B,
            n_nodes=N,
        )
        x = h.reshape(B * N, -1)
        for block in self.blocks:
            x = block(x, edge_index, edge_h)
        h = x.reshape(B, N, -1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        latent_next = self.norm(h + attn_out)
        return latent_next, memory_next


class LatentGraphWorldModel(WorldModel):
    """Deterministic latent graph world model with recurrent imagination."""

    def __init__(
        self,
        *,
        in_channels: int,
        hidden: int,
        latent_dim: int,
        action_dim: int,
        node_attr_dim: int,
        edge_attr_dim: int,
        encoder_layers: int = 2,
        dynamics_layers: int = 2,
        attn_heads: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.action = ActionConditioner(action_dim=action_dim, hidden=hidden, dropout=dropout)
        self.encoder = GraphStateEncoder(
            in_channels=in_channels,
            node_attr_dim=node_attr_dim,
            edge_attr_dim=edge_attr_dim,
            hidden=hidden,
            latent_dim=latent_dim,
            n_layers=encoder_layers,
            dropout=dropout,
        )
        self.dynamics = LatentGraphDynamics(
            latent_dim=latent_dim,
            edge_attr_dim=edge_attr_dim,
            action_dim=hidden,
            n_layers=dynamics_layers,
            attn_heads=attn_heads,
            dropout=dropout,
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, in_channels),
        )

    def initial_memory(self, state: torch.Tensor) -> torch.Tensor:
        return state.new_zeros(state.shape[0], state.shape[1], self.latent_dim)

    def encode(
        self,
        state: torch.Tensor,
        *,
        action_global: torch.Tensor,
        graph: StaticGraph,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_emb, gamma, beta = self.action(action_global)
        latent = self.encoder(state, graph=graph, action_gamma=gamma, action_beta=beta)
        return latent, action_emb

    def transition(
        self,
        latent: torch.Tensor,
        memory: torch.Tensor,
        *,
        action_emb: torch.Tensor,
        graph: StaticGraph,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dynamics(latent, memory, action_emb=action_emb, graph=graph)

    def decode_delta(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def step(
        self,
        state: torch.Tensor,
        *,
        action_global: torch.Tensor,
        graph: StaticGraph,
        memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory = self.initial_memory(state) if memory is None else memory
        latent, action_emb = self.encode(state, action_global=action_global, graph=graph)
        latent_next, memory_next = self.transition(
            latent, memory, action_emb=action_emb, graph=graph
        )
        next_state = state + self.decode_delta(latent_next)
        return next_state, memory_next, latent_next

    def predict_delta(
        self,
        state: torch.Tensor,
        *,
        action_global: torch.Tensor,
        graph: StaticGraph,
    ) -> torch.Tensor:
        next_state, _, _ = self.step(state, action_global=action_global, graph=graph)
        return next_state - state

    def rollout(
        self,
        state_t: torch.Tensor,
        *,
        action_sequence: torch.Tensor,
        graph: StaticGraph,
        teacher_sequence: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        states = [state_t]
        state = state_t
        memory = self.initial_memory(state_t)
        for step in range(action_sequence.shape[1]):
            next_state, memory, _ = self.step(
                state,
                action_global=action_sequence[:, step],
                graph=graph,
                memory=memory,
            )
            states.append(next_state)
            if teacher_sequence is not None and teacher_forcing_ratio > 0.0:
                use_teacher = torch.rand(
                    state.shape[0], 1, 1, device=state.device
                ) < teacher_forcing_ratio
                state = torch.where(use_teacher, teacher_sequence[:, step + 1], next_state)
            else:
                state = next_state
        return torch.stack(states, dim=1)
