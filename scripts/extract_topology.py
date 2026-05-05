"""Day-1: extract IEEE39 10-machine reduced-graph topology to data/graph_ieee39.h5.

Pulls node_attr (per-generator H, D, X'd, Xq, ...) and edge_attr (per-edge
equivalent impedance/susceptance over the reduced subgraph) from the
CloudPSS IEEE39 model and persists them. Also computes the mapping from
"physical line_id (0..45)" to "reduced-graph edge_id" used by the fault
edge flag (spec §3.7).

Run once. Output committed to data/graph_ieee39.h5 (data/ is gitignored
but graph_ieee39.h5 is small and we'll force-add it).

The three private helpers (_extract_generator_params, _extract_reduced_edges,
_build_line_to_edge_mapping) raise NotImplementedError. Fill them in per
the SDK attribute names documented in
/mnt/d/SCU/cloudpss-simulation/cloudpss-emt-guide.md.
"""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
from cloudpss import Model


RID = os.environ["GRIDWM_IEEE39_RID"]
OUT = Path("data") / "graph_ieee39.h5"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = Model.fetch(RID)

    gen_params = _extract_generator_params(model)
    edges, edge_attr = _extract_reduced_edges(model)
    line_to_edge = _build_line_to_edge_mapping(model)

    with h5py.File(OUT, "w") as f:
        f.create_dataset("edge_index", data=edges, dtype="int64")
        f.create_dataset("edge_attr", data=edge_attr, dtype="float32")
        f.create_dataset("node_attr", data=gen_params, dtype="float32")
        f.create_dataset("line_to_edge", data=line_to_edge, dtype="int64")
        f.attrs["rid"] = RID
        f.attrs["graph_view"] = "10-machine reduced subgraph"

    print(
        f"wrote {OUT}: nodes={gen_params.shape}, "
        f"edges={edges.shape[1]}, line_to_edge={line_to_edge.shape}"
    )


def _extract_generator_params(model) -> np.ndarray:
    """Read the 10 generator components from `model.components` and stack
    parameters [H, D, X'd, X'q, T'd0, T'q0, ...] per cloudpss-emt-guide.md.

    Returns an (10, F_n) float32 array.
    """
    raise NotImplementedError(
        "Fill in per /mnt/d/SCU/cloudpss-simulation/cloudpss-emt-guide.md "
        "generator schema. Look up the per-generator attribute names "
        "(H, D, transient reactances) on the CloudPSS Model component API."
    )


def _extract_reduced_edges(model) -> tuple[np.ndarray, np.ndarray]:
    """Build the inter-machine graph by reducing the 39-bus passive network
    via Kron reduction, or by inheriting CloudPSS's exposed equivalent
    subgraph for reduced-order plotting.

    Returns (edge_index[2,E], edge_attr[E,F_e]).
    """
    raise NotImplementedError(
        "Fill in per /mnt/d/SCU/cloudpss-simulation/cloudpss-emt-guide.md "
        "reduced-graph schema. Two options: (a) compute Kron reduction "
        "yourself from the 39-bus admittance matrix, (b) read CloudPSS's "
        "pre-computed reduced-order topology if exposed by the SDK."
    )


def _build_line_to_edge_mapping(model) -> np.ndarray:
    """For each of the 46 physical lines, find which reduced-graph edge(s)
    it appears in. Phase 1 assumption: each physical line maps to exactly
    one reduced-graph edge (Kron-reduced lumped impedance).

    Returns an (46,) int64 array; entry `i` is the reduced-graph edge id
    that physical line `i` contributes to.
    """
    raise NotImplementedError(
        "Fill in per /mnt/d/SCU/cloudpss-simulation/cloudpss-emt-guide.md "
        "line→edge mapping. Verify the 1-to-1 assumption on the actual "
        "model. If a physical line appears in multiple reduced edges, "
        "raise here so the assumption is corrected before downstream code."
    )


if __name__ == "__main__":
    main()
