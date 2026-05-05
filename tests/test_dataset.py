from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from wmagent.data.dataset import ACTION_DIM, wm-agentDataset
from wmagent.data.normalizer import NormStats, save_stats
from wmagent.data.schema import FaultAction, SampleGroup, write_sample
from wmagent.data.splits import TrajectoryRandomSplit, save_split


@pytest.fixture
def fixture_dataset(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    n_t = 3000
    rng = np.random.default_rng(0)
    uids = [f"u{i:04d}" for i in range(8)]
    for i, uid in enumerate(uids):
        state = rng.standard_normal((n_t, 10, 4)).astype("float32")
        bus_obs = rng.standard_normal((n_t, 3)).astype("float32")
        sg = SampleGroup(
            uid=uid,
            action=FaultAction(fs=1.02, fe=1.08, ft=7, chg_ohm=1.0),
            state=state,
            bus_obs=bus_obs,
            meta={
                "case": "ieee39", "seed": i, "cloudpss_rid": "r",
                "cloudpss_version": "4.5.111", "data_version": "phase1.v2-pathA",
                "output_dt_seconds": 0.001, "topology_static_during_fault": True,
                "collected_at": "2026-04-29T12:00:00",
            },
        )
        with h5py.File(raw / f"{uid}.h5", "w") as f:
            write_sample(f, sg)

    split = TrajectoryRandomSplit(
        train_uids=tuple(uids[:6]),
        val_uids=tuple(uids[6:7]),
        test_uids=tuple(uids[7:8]),
        seed=0,
    )
    save_split(split, tmp_path / "splits.json")
    stats = NormStats(
        mean={"wr": 0.0, "LA": 0.0, "VT": 0.0, "IT": 0.0},
        std={"wr": 1.0, "LA": 1.0, "VT": 1.0, "IT": 1.0},
        delta_std={"wr": 1.0, "LA": 1.0, "VT": 1.0, "IT": 1.0},
    )
    save_stats(stats, tmp_path / "norm_stats.json")
    return tmp_path, split


def test_dataset_split_filtering_by_uid(fixture_dataset):
    base, split = fixture_dataset
    ds = wm-agentDataset(
        raw_dir=base / "raw",
        splits_path=base / "splits.json",
        norm_stats_path=base / "norm_stats.json",
        split="train",
        pairs_per_traj_per_epoch=4,
    )
    assert len(ds.entries) == len(split.train_uids)


def test_dataset_returns_normalized_pair_shapes(fixture_dataset):
    base, _ = fixture_dataset
    ds = wm-agentDataset(
        raw_dir=base / "raw",
        splits_path=base / "splits.json",
        norm_stats_path=base / "norm_stats.json",
        split="train",
        pairs_per_traj_per_epoch=4,
    )
    item = ds[0]
    assert item["state_t"].shape == (10, 4)
    assert item["state_tp1"].shape == (10, 4)
    assert item["delta_norm"].shape == (10, 4)
    assert item["action_global"].shape == (ACTION_DIM,)
    # ft=7 → index 7 of one-hot is 1
    assert item["action_global"][7].item() == pytest.approx(1.0)


def test_dataset_returns_rollout_sequence_shapes(fixture_dataset):
    base, _ = fixture_dataset
    ds = wm-agentDataset(
        raw_dir=base / "raw",
        splits_path=base / "splits.json",
        norm_stats_path=base / "norm_stats.json",
        split="train",
        pairs_per_traj_per_epoch=4,
        rollout_horizon=5,
        fault_window_frac=1.0,
        post_clear_window_s=0.0,
    )
    item = ds[0]
    assert item["state_tm1"].shape == (10, 4)
    assert item["state_sequence"].shape == (6, 10, 4)
    assert item["delta_sequence"].shape == (5, 10, 4)
    assert item["action_sequence"].shape == (5, ACTION_DIM)
    assert item["fault_window_sequence"].shape == (5,)
    assert item["action_sequence"][1, 11] != item["action_sequence"][0, 11]
