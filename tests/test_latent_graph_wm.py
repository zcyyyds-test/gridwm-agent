import torch

from wmagent.models.base import StaticGraph
from wmagent.models.latent_graph_wm import LatentGraphWorldModel


def fake_graph():
    n_nodes = 10
    n_edges = 24
    return StaticGraph(
        edge_index=torch.randint(0, n_nodes, (2, n_edges), dtype=torch.long),
        edge_attr=torch.randn(n_edges, 3),
        node_attr=torch.randn(n_nodes, 5),
    )


def test_latent_world_model_predict_delta_shape():
    graph = fake_graph()
    model = LatentGraphWorldModel(
        in_channels=4,
        hidden=32,
        latent_dim=24,
        action_dim=12,
        node_attr_dim=5,
        edge_attr_dim=3,
        encoder_layers=1,
        dynamics_layers=1,
        attn_heads=2,
    )
    state = torch.randn(3, 10, 4)
    action = torch.randn(3, 12)
    delta = model.predict_delta(state, action_global=action, graph=graph)
    assert delta.shape == state.shape


def test_latent_world_model_rollout_shape():
    graph = fake_graph()
    model = LatentGraphWorldModel(
        in_channels=4,
        hidden=32,
        latent_dim=24,
        action_dim=12,
        node_attr_dim=5,
        edge_attr_dim=3,
        encoder_layers=1,
        dynamics_layers=1,
        attn_heads=2,
    )
    state = torch.randn(2, 10, 4)
    actions = torch.randn(2, 5, 12)
    rollout = model.rollout(state, action_sequence=actions, graph=graph)
    assert rollout.shape == (2, 6, 10, 4)
