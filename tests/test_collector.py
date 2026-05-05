from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

from wmagent.data.collector import (
    CollectorConfig,
    collect_one_trajectory,
    sample_fault_action,
)
from wmagent.data.schema import FaultAction


def test_sample_fault_action_in_bounds():
    rng = np.random.default_rng(0)
    a = sample_fault_action(rng)
    assert isinstance(a, FaultAction)
    assert 1.0 <= a.fs <= 5.0
    assert 0.05 <= a.fe - a.fs <= 0.20
    assert a.ft in {1, 3, 7}
    assert 0.01 <= a.chg_ohm <= 10.0


def _fake_plots_list(n_samples: int) -> list:
    """List-of-plot-dicts shape that extract_state_arrays expects."""
    plots = []
    for title, base in [("wr", 1.0), ("LA", 0.0), ("VT", 1.0), ("IT", 0.0)]:
        traces = [
            {
                "name": f"#Gen{30 + g:02d}.{title}:0",
                "x": np.arange(n_samples).tolist(),
                "y": (np.zeros(n_samples) + base + 0.01 * g).tolist(),
            }
            for g in range(10)
        ]
        plots.append({"data": {"title": title, "traces": traces}})
    plots.append(
        {
            "data": {
                "title": "Ib37",
                "traces": [
                    {
                        "name": f"#Bus37.Ib:{phase}",
                        "x": np.arange(n_samples).tolist(),
                        "y": np.zeros(n_samples).tolist(),
                    }
                    for phase in ("a", "b", "c")
                ],
            }
        }
    )
    return plots


@patch("wmagent.data.collector.cloudpss")
def test_collect_one_trajectory_writes_valid_sample(mock_cloudpss, tmp_path: Path):
    runner = MagicMock()
    runner.result.waitFor.return_value = None
    runner.result.getPlots.return_value = _fake_plots_list(15001)
    runner.result.getLogs.return_value = []

    fake_model = MagicMock()
    fake_model.jobs = [{"rid": "model/zcyyyds-test/job_emtp", "args": {}}]
    fake_model.configs = [{"name": "default"}]
    fake_model.run.return_value = runner
    mock_cloudpss.Model.fetch.return_value = fake_model

    cfg = CollectorConfig(
        rid="rid_test",
        out_dir=tmp_path,
        expected_n_samples=15001,
        output_dt_s=0.001,
        cloudpss_version="4.5.111",
    )
    uid = collect_one_trajectory(cfg, rng=np.random.default_rng(1))
    h5_path = tmp_path / f"{uid}.h5"
    assert h5_path.exists()
    with h5py.File(h5_path, "r") as f:
        g = f[f"sample_{uid}"]
        assert g["state"]["wr"].shape == (15001, 10)
        assert g["bus_obs"].shape == (15001, 3)
        assert g["meta"].attrs["topology_static_during_fault"]
        # action attrs reflect the new schema
        assert "fs" in g["action"].attrs
        assert "fe" in g["action"].attrs
        assert "ft" in g["action"].attrs
        assert "chg_ohm" in g["action"].attrs
