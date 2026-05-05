"""Smoke tests for the wm-agent FastAPI surface.

The world model itself is faked via a dependency override so the suite
runs without GPU, without a real V2 checkpoint, and without CloudPSS data.
We only check that routing + schema + the search_result_to_record contract
all line up. Real-checkpoint serving is verified manually by running
``python scripts/serve_api.py`` and ``curl localhost:8000/health``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from wmagent.serve.api import app, get_system
from wmagent.world.base import (
    ImaginedFuture,
    RiskSignal,
    SearchResult,
    WorldEvent,
    WorldState,
)


class _FakeSystem:
    """Minimal stand-in for PowerGridWorldModelSystem -- enough for schema."""

    domain = "power_grid"
    run_id = "fake_run"
    model_type = "fake_model"
    checkpoint_epoch = 7
    device = "cpu"

    def anchor_state(self, *, split, seed, horizon, anchor_index):
        return WorldState(
            tensor=None,
            domain=self.domain,
            metadata={
                "split": split,
                "seed": seed,
                "horizon": horizon,
                "anchor_index": anchor_index,
                "fault_window_active": 1.0,
            },
        )

    def candidate_events(self, *, horizon):
        return [
            WorldEvent(
                tensor=None,
                domain=self.domain,
                metadata={"event_code": f"FT-{ft}", "fault_type": ft,
                          "duration_ms": 120.0, "resistance_ohm": 0.1,
                          "start_s": 2.0, "tau0": 0.1},
            )
            for ft in (1, 3, 7)
        ]

    def search(self, state, events, *, top_k=10):
        results = []
        for rank, event in enumerate(events[:top_k], start=1):
            risk = RiskSignal(
                score=80 + rank,
                band="high" if rank == 1 else "medium",
                value=0.5 - 0.1 * rank,
                features={"dominant_channel": "IT", "dominant_node": "G5"},
            )
            results.append(SearchResult(
                rank=rank,
                future=ImaginedFuture(rollout=None, state=state, event=event),
                risk=risk,
            ))
        return results


@pytest.fixture
def client():
    app.dependency_overrides[get_system] = lambda: _FakeSystem()
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_health_returns_world_model_metadata(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["domain"] == "power_grid"
    assert body["run_id"] == "fake_run"
    assert body["checkpoint_epoch"] == 7


def test_search_returns_ranked_records_with_scenario_and_risk(client):
    response = client.post("/search", json={"top_k": 2, "horizon": 10})
    assert response.status_code == 200
    body = response.json()
    assert body["horizon"] == 10
    assert body["top_k"] == 2
    assert body["n_candidates"] == 3
    assert len(body["results"]) == 2
    first = body["results"][0]
    assert first["rank"] == 1
    assert "event_code" in first["scenario"]
    assert "score" in first["risk"]
    assert "band" in first["risk"]
    assert "dominant_channel" in first["risk"]


def test_search_validates_input(client):
    response = client.post("/search", json={"top_k": 0})
    assert response.status_code == 422


def test_health_runs_without_real_checkpoint(client):
    """Sanity: the dependency override means tests can run on any machine."""
    response = client.get("/health")
    assert response.status_code == 200
