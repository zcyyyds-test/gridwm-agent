from pathlib import Path

import h5py
import numpy as np
import pytest

from wmagent.data.schema import (
    FaultAction,
    SampleGroup,
    write_sample,
    read_sample,
    validate_sample,
)


def test_fault_action_validates_ranges():
    # Valid construction
    FaultAction(fs=2.0, fe=2.1, ft=7, chg_ohm=1.0)

    # fs out of [1.0, 5.0]
    with pytest.raises(ValueError):
        FaultAction(fs=0.5, fe=0.6, ft=7, chg_ohm=1.0)

    # duration fe-fs out of [0.05, 0.20]
    with pytest.raises(ValueError):
        FaultAction(fs=2.0, fe=2.3, ft=7, chg_ohm=1.0)  # duration 0.30 > 0.20

    # ft not in {1, 3, 7}
    with pytest.raises(ValueError):
        FaultAction(fs=2.0, fe=2.1, ft=5, chg_ohm=1.0)

    # chg_ohm out of [0.01, 10.0]
    with pytest.raises(ValueError):
        FaultAction(fs=2.0, fe=2.1, ft=7, chg_ohm=0.001)


def test_round_trip_sample(tmp_path: Path):
    n_samples = 15001
    state = np.random.randn(n_samples, 10, 4).astype("float32")
    bus_obs = np.random.randn(n_samples, 3).astype("float32")
    action = FaultAction(fs=2.0, fe=2.1, ft=7, chg_ohm=1.0)
    sg = SampleGroup(
        uid="test_0001",
        action=action,
        state=state,
        bus_obs=bus_obs,
        meta={
            "case": "ieee39",
            "seed": 42,
            "cloudpss_rid": "rid_test",
            "cloudpss_version": "4.5.111",
            "data_version": "phase1.v1",
            "output_dt_seconds": 0.001,
            "topology_static_during_fault": True,
            "collected_at": "2026-04-29T12:00:00",
        },
    )
    path = tmp_path / "out.h5"
    with h5py.File(path, "w") as f:
        write_sample(f, sg)
    with h5py.File(path, "r") as f:
        read = read_sample(f, "test_0001")
    assert read.action == action
    assert np.allclose(read.state, state)
    assert np.allclose(read.bus_obs, bus_obs)
    assert read.meta["case"] == "ieee39"


def test_validate_sample_rejects_wrong_shape():
    sg = SampleGroup(
        uid="bad",
        action=FaultAction(fs=2.0, fe=2.1, ft=7, chg_ohm=1.0),
        state=np.zeros((100, 10, 4), dtype="float32"),  # T mismatch
        bus_obs=np.zeros((100, 3), dtype="float32"),
        meta={
            "case": "ieee39", "seed": 1, "cloudpss_rid": "r",
            "cloudpss_version": "4.5.111", "data_version": "phase1.v1",
            "output_dt_seconds": 0.001, "topology_static_during_fault": True,
            "collected_at": "2026-04-29T12:00:00",
        },
    )
    with pytest.raises(ValueError, match="state shape"):
        validate_sample(sg, expected_n_samples=15001)
