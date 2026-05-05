import pytest
import torch

pytest.importorskip("torch_geometric")

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


def test_batched_forward_independent_per_sample(fake_graph):
    """Regression: the batched MPNN should produce the same per-sample output
    as a single-sample forward, since each batch element is a disjoint copy
    of the static graph and there is no cross-sample message passing."""
    torch.manual_seed(0)
    C, A = 4, 11
    model = MPNNDynamics(in_channels=C, hidden=32, n_layers=2, action_dim=A,
                        node_attr_dim=5, edge_attr_dim=3)
    model.eval()

    state_a = torch.randn(1, 10, C)
    state_b = torch.randn(1, 10, C)
    action_a = torch.randn(1, A)
    action_b = torch.randn(1, A)

    with torch.no_grad():
        out_a = model.predict_delta(state_a, action_global=action_a, graph=fake_graph)
        out_b = model.predict_delta(state_b, action_global=action_b, graph=fake_graph)
        out_batched = model.predict_delta(
            torch.cat([state_a, state_b], dim=0),
            action_global=torch.cat([action_a, action_b], dim=0),
            graph=fake_graph,
        )

    assert torch.allclose(out_batched[0:1], out_a, atol=1e-5)
    assert torch.allclose(out_batched[1:2], out_b, atol=1e-5)
