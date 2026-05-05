"""Phase 1: build a placeholder fully-connected 10-generator graph and
save to data/graph_ieee39.h5. No CloudPSS API call needed.

Phase 2 will replace this with real Kron-reduced impedance extraction
from CloudPSS. The h5 schema is already what train/eval expect:
edge_index (2,E), edge_attr (E,F_e), node_attr (N,F_n).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


N_GENERATORS = 10
NODE_ATTR_DIM = 5    # placeholder; Phase 2 fills with H, D, X'd, X'q, T'd0
EDGE_ATTR_DIM = 3    # placeholder; Phase 2 fills with R, X, B
OUT = Path("data") / "graph_ieee39.h5"


def fully_connected_edge_index(n: int) -> np.ndarray:
    """Return edge_index (2, E) with both directions of every undirected edge."""
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    return np.array([src, dst], dtype="int64")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    edge_index = fully_connected_edge_index(N_GENERATORS)  # (2, 90)
    edge_attr = np.zeros((edge_index.shape[1], EDGE_ATTR_DIM), dtype="float32")
    node_attr = np.zeros((N_GENERATORS, NODE_ATTR_DIM), dtype="float32")

    with h5py.File(OUT, "w") as f:
        f.create_dataset("edge_index", data=edge_index, dtype="int64")
        f.create_dataset("edge_attr", data=edge_attr, dtype="float32")
        f.create_dataset("node_attr", data=node_attr, dtype="float32")
        f.attrs["graph_view"] = "phase1-placeholder fully-connected 10-gen"
        f.attrs["n_nodes"] = N_GENERATORS
        f.attrs["n_edges"] = edge_index.shape[1]

    print(
        f"wrote {OUT}: nodes={node_attr.shape}, edges={edge_index.shape[1]} "
        f"(fully-connected directed)"
    )


if __name__ == "__main__":
    main()
