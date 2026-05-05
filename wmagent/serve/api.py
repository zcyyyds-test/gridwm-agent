"""Localhost FastAPI surface for the wm-agent world-model system.

Exposes the same ``imagine / score / search`` contract from
:mod:`wmagent.world.base` over HTTP. The endpoints return the same shapes
that :func:`wmagent.world.power_grid.search_result_to_record` already emits
into the offline experiment artifacts, so the static playground JSON and
the live API speak the same vocabulary.

Localhost only, no auth, no HTTPS, no Docker. The point is "the world
model is reachable from a service boundary," not production hardening.
Boot the world model once at app startup and reuse the singleton across
requests.

Run with::

    uvicorn wmagent.serve.api:app --host 127.0.0.1 --port 8000

or via :mod:`scripts.serve_api`.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from wmagent.world.power_grid import (
    PowerGridWorldModelSystem,
    search_result_to_record,
)


_DEFAULT_RUN_DIR = Path(os.environ.get("GRIDWM_RUN_DIR", "outputs/run_3acec7d14d"))


class SearchRequest(BaseModel):
    horizon: int = Field(default=10, ge=1, le=40)
    top_k: int = Field(default=10, ge=1, le=162)
    split: str = Field(default="val")
    anchor_index: int = Field(default=0, ge=0)
    seed: int = Field(default=2026)


class HealthResponse(BaseModel):
    status: str
    domain: str
    run_id: str
    model_type: str
    checkpoint_epoch: int | None
    device: str


class SearchResponseRow(BaseModel):
    rank: int
    scenario: dict[str, Any]
    risk: dict[str, Any]


class SearchResponse(BaseModel):
    horizon: int
    n_candidates: int
    top_k: int
    anchor: dict[str, Any]
    results: list[SearchResponseRow]


@lru_cache(maxsize=1)
def get_system() -> PowerGridWorldModelSystem:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return PowerGridWorldModelSystem.from_run_dir(_DEFAULT_RUN_DIR, device=device)


app = FastAPI(
    title="wm-agent World Model API",
    summary="Imagine futures, score risk, and search a candidate event space.",
    version="0.4.0",
)


@app.get("/health", response_model=HealthResponse)
def health(system: PowerGridWorldModelSystem = Depends(get_system)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        domain=system.domain,
        run_id=system.run_id,
        model_type=system.model_type,
        checkpoint_epoch=system.checkpoint_epoch,
        device=str(system.device),
    )


@app.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    system: PowerGridWorldModelSystem = Depends(get_system),
) -> SearchResponse:
    try:
        state = system.anchor_state(
            split=request.split,
            seed=request.seed,
            horizon=request.horizon,
            anchor_index=request.anchor_index,
        )
        events = system.candidate_events(horizon=request.horizon)
        results = system.search(state, events, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SearchResponse(
        horizon=request.horizon,
        n_candidates=len(events),
        top_k=request.top_k,
        anchor={
            "split": request.split,
            "anchor_index": request.anchor_index,
            "seed": request.seed,
            **state.metadata,
        },
        results=[
            SearchResponseRow(**search_result_to_record(result))
            for result in results
        ],
    )
