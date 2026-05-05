from pathlib import Path

import numpy as np
import pytest

from wmagent.data.normalizer import (
    NormStats,
    compute_norm_stats,
    save_stats,
    load_stats,
    apply_norm,
    invert_norm,
)


def test_compute_returns_per_channel_stats():
    state = np.random.randn(100, 15001, 10, 4).astype("float32") * 2.0 + 1.0
    delta = np.diff(state, axis=1)
    stats = compute_norm_stats(state, delta)
    assert set(stats.mean.keys()) == {"wr", "LA", "VT", "IT"}
    assert all(s > 0 for s in stats.std.values())
    assert all(s > 0 for s in stats.delta_std.values())


def test_round_trip_stats(tmp_path: Path):
    stats = NormStats(
        mean={"wr": 1.0, "LA": 0.0, "VT": 1.0, "IT": 0.0},
        std={"wr": 0.5, "LA": 0.3, "VT": 0.1, "IT": 0.2},
        delta_std={"wr": 0.01, "LA": 0.005, "VT": 0.001, "IT": 0.002},
    )
    p = tmp_path / "stats.json"
    save_stats(stats, p)
    loaded = load_stats(p)
    assert loaded == stats


def test_apply_then_invert_is_identity():
    rng = np.random.default_rng(0)
    state = rng.standard_normal((1000, 10, 4)).astype("float32")
    stats = NormStats(
        mean={"wr": 0.1, "LA": 0.0, "VT": 1.0, "IT": -0.1},
        std={"wr": 0.5, "LA": 0.3, "VT": 0.1, "IT": 0.2},
        delta_std={"wr": 0.01, "LA": 0.005, "VT": 0.001, "IT": 0.002},
    )
    s_norm = apply_norm(state, stats)
    s_back = invert_norm(s_norm, stats)
    np.testing.assert_allclose(s_back, state, atol=1e-5)
