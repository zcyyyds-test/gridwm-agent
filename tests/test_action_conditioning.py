import torch

from wmagent.models.base import StaticGraph
from wmagent.models.mpnn import MPNNDynamics


def test_different_actions_produce_different_deltas():
    torch.manual_seed(0)
    n_nodes, n_edges, A, C = 10, 24, 11, 4
    graph = StaticGraph(
        edge_index=torch.randint(0, n_nodes, (2, n_edges), dtype=torch.long),
        edge_attr=torch.randn(n_edges, 3),
        node_attr=torch.randn(n_nodes, 5),
    )
    model = MPNNDynamics(in_channels=C, hidden=32, n_layers=2, action_dim=A,
                        node_attr_dim=5, edge_attr_dim=3)
    state = torch.randn(1, n_nodes, C)

    action_a = torch.zeros(1, A)
    action_a[0, 7] = 1.0    # ft=ABC three-phase
    action_a[0, 8] = 0.5    # fs_norm
    action_a[0, 9] = 0.5    # duration_norm
    action_a[0, 10] = 0.5   # chg_log_norm

    action_b = torch.zeros(1, A)
    action_b[0, 1] = 1.0    # ft=A-g
    action_b[0, 8] = 0.2
    action_b[0, 9] = 0.1
    action_b[0, 10] = 0.9

    delta_a = model.predict_delta(state, action_global=action_a, graph=graph)
    delta_b = model.predict_delta(state, action_global=action_b, graph=graph)

    diff = (delta_a - delta_b).norm() / (delta_a.norm() + 1e-8)
    assert diff > 0.0
