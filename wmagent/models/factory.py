"""Model factory for baseline and latent world-model variants."""
from __future__ import annotations

from typing import Any

from wmagent.models.base import StaticGraph, WorldModel
from wmagent.models.latent_graph_wm import LatentGraphWorldModel
from wmagent.models.mpnn import MPNNDynamics


def build_world_model(model_cfg: dict[str, Any], graph: StaticGraph) -> WorldModel:
    cfg = dict(model_cfg)
    model_type = cfg.pop("model_type", "mpnn")
    cfg["node_attr_dim"] = int(graph.node_attr.shape[1])
    cfg["edge_attr_dim"] = int(graph.edge_attr.shape[1])
    if model_type == "mpnn":
        return MPNNDynamics(**cfg)
    if model_type == "latent_graph_wm":
        return LatentGraphWorldModel(**cfg)
    raise ValueError(f"unknown model_type: {model_type}")
