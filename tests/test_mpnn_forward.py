import pytest
import torch

from wmagent.models.base import StaticGraph
from wmagent.models.mpnn import MPNNDynamics


@pytest.fixture
def fake_graph():
    n_nodes = 10
    n_edges = 24
    return StaticGraph(
        edge_index=torch.randint(0, n_nodes, (2, n_edges), dtype=torch.long),
        edge_attr=torch.randn(n_edges, 3),
        node_attr=torch.randn(n_nodes, 5),
    )


def test_forward_shape(fake_graph):
    B, N, C, A = 4, 10, 4, 11
    state = torch.randn(B, N, C)
    action_global = torch.randn(B, A)

    model = MPNNDynamics(in_channels=C, hidden=32, n_layers=2, action_dim=A,
                        node_attr_dim=5, edge_attr_dim=3)
    out = model(state, action_global=action_global, graph=fake_graph)
    assert out.shape == (B, N, C)


def test_residual_zero_action_zero_delta(fake_graph):
    B, N, C, A = 2, 10, 4, 11
    state = torch.randn(B, N, C)
    action_global = torch.randn(B, A)

    model = MPNNDynamics(in_channels=C, hidden=32, n_layers=2, action_dim=A,
                        node_attr_dim=5, edge_attr_dim=3)
    with torch.no_grad():
        for p in model.residual_head.parameters():
            p.zero_()
    delta_norm = model.predict_delta(state, action_global=action_global, graph=fake_graph)
    assert torch.allclose(delta_norm, torch.zeros_like(delta_norm), atol=1e-6)
