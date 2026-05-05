"""Regression tests on the bundled demo_run checkpoint.

These guard against accidental corruption / silent regression of the
demo artifact that the README quickstart depends on. They do not load
the underlying neural model (torch_geometric is optional in this test
suite) — they just verify the on-disk shape and that the headline
numbers are within their published bootstrap CIs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

DEMO = Path(__file__).resolve().parents[1] / "outputs" / "demo_run"


def _has_demo() -> bool:
    return DEMO.exists() and (DEMO / "best.pt").exists() and (DEMO / "agent.pt").exists()


pytestmark = pytest.mark.skipif(not _has_demo(), reason="bundled demo_run not present")


def test_demo_checkpoints_load_with_weights_only_true():
    best = torch.load(DEMO / "best.pt", map_location="cpu", weights_only=True)
    agent = torch.load(DEMO / "agent.pt", map_location="cpu", weights_only=True)
    assert "model" in best and "cfg" in best
    assert "model" in agent and "agent_cfg" in agent
    assert best["epoch"] >= 0
    assert agent["epoch"] >= 0


def test_demo_eval_within_published_bootstrap_ci():
    """demo_run eval point estimates must fall inside the bootstrap CIs we
    publish in README. If a future re-collection or retrain shifts these
    numbers outside the band, the README must be updated alongside."""
    val_ci = json.loads((DEMO / "bootstrap_ci_val.json").read_text())
    test_ci = json.loads((DEMO / "bootstrap_ci_test.json").read_text())

    for ci in (val_ci, test_ci):
        m = ci["metrics"]
        for key in ("discover_top10pct_hit_rate", "safe_bottom10pct_hit_rate"):
            point = m[key]["point"]
            lo, hi = m[key]["ci_low"], m[key]["ci_high"]
            assert lo <= point <= hi, f"{key} point {point} outside CI [{lo},{hi}]"


def test_demo_norm_stats_present():
    stats = Path(__file__).resolve().parents[1] / "data" / "norm_stats.json"
    assert stats.exists(), "data/norm_stats.json must be committed for quickstart"
    d = json.loads(stats.read_text())
    assert set(d["mean"].keys()) == {"wr", "LA", "VT", "IT"}
    assert all(d["std"][k] > 0 for k in d["std"])


def test_demo_anchor_cache_covers_quickstart_defaults():
    """The bundled anchor cache must contain the (split, seed, horizon,
    anchor_index) tuple that the README quickstart and the FastAPI
    default request use, otherwise fresh-clone users hit a cache miss
    and `anchor_state` falls back to data/raw which is gitignored."""
    cache_path = DEMO / "anchors.pt"
    assert cache_path.exists(), "outputs/demo_run/anchors.pt missing"
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    assert payload["version"] == 1
    keys = {(a["split"], a["seed"], a["horizon"], a["anchor_index"])
            for a in payload["anchors"]}
    # README rank_scenarios.py default + FastAPI SearchRequest default:
    assert ("val", 2026, 10, 0) in keys
    # Test split is also exposed via the `--split test` quickstart arm:
    assert ("test", 2026, 10, 0) in keys
    # Each cached anchor carries the tensors anchor_state needs:
    for a in payload["anchors"][:3]:
        assert a["state_t"].ndim == 2  # (N, C)
        assert a["action_global"].ndim == 1  # (action_dim,)
