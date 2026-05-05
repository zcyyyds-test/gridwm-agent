"""Single-trajectory and batched CloudPSS collection driver (path-A: single fault element)."""
from __future__ import annotations

import datetime as _dt
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

import cloudpss
from wmagent.data.cloudpss_helpers import (
    apply_fault,
    ensure_authenticated,
    extract_state_arrays,
    set_emt_params,
    topology_static_during_fault,
)
from wmagent.data.schema import FaultAction, SampleGroup, validate_sample, write_sample


@dataclass
class CollectorConfig:
    rid: str
    out_dir: Path
    expected_n_samples: int
    output_dt_s: float
    cloudpss_version: str
    sim_end_time_s: float = 15.0
    sim_step_time_s: float = 5.0e-5
    sim_n_cpu: int = 1
    max_retries: int = 3
    retry_backoff_base_s: float = 2.0


_FT_CHOICES = (1, 3, 7)  # Phase 1 fault type codes


def _is_transient_error(exc: BaseException) -> bool:
    """Heuristic: CloudPSS server-side races (concurrent Model.update against the
    `tag_resource_unique` index) and rate-limit / network blips are recoverable."""
    msg = str(exc).lower()
    return any(
        s in msg
        for s in (
            "duplicate entry",
            "tag_resource",
            "rate limit",
            "quota",
            "timed out",
            "timeout",
            "connection reset",
            "temporarily unavailable",
        )
    )


def sample_fault_action(rng: np.random.Generator) -> FaultAction:
    """Sample a Phase-1 fault action: fs uniform[1,5]s, duration uniform[0.05,0.20]s,
    ft uniform over {1,3,7}, chg log-uniform on [0.01, 10] Ω.
    """
    fs = float(rng.uniform(1.0, 5.0))
    duration = float(rng.uniform(0.05, 0.20))
    fe = fs + duration
    ft = int(rng.choice(_FT_CHOICES))
    log_chg = float(rng.uniform(math.log10(0.01), math.log10(10.0)))
    chg_ohm = float(10.0 ** log_chg)
    return FaultAction(fs=fs, fe=fe, ft=ft, chg_ohm=chg_ohm)


def _collect_once(cfg: CollectorConfig, *, rng: np.random.Generator) -> str:
    """One attempt — no retry. Raises on any failure."""
    action = sample_fault_action(rng)
    model = cloudpss.Model.fetch(cfg.rid)
    apply_fault(model, fs=action.fs, fe=action.fe, ft=action.ft, chg_ohm=action.chg_ohm)
    set_emt_params(
        model,
        end_time=cfg.sim_end_time_s,
        step_time=cfg.sim_step_time_s,
        n_cpu=cfg.sim_n_cpu,
    )
    cloudpss.Model.update(model)

    emt_job = next((j for j in model.jobs if "emtp" in j["rid"]), None)
    if emt_job is None:
        raise RuntimeError(f"model {cfg.rid} has no EMT job")
    runner = model.run(emt_job, model.configs[0])
    runner.result.waitFor()
    plots = list(runner.result.getPlots())
    state, bus_obs = extract_state_arrays(plots)

    uid = uuid.uuid4().hex[:12]
    sg = SampleGroup(
        uid=uid,
        action=action,
        state=state,
        bus_obs=bus_obs,
        meta={
            "case": "ieee39",
            "seed": int(rng.bit_generator.state["state"]["state"] & 0xFFFFFFFF),
            "cloudpss_rid": cfg.rid,
            "cloudpss_version": cfg.cloudpss_version,
            "data_version": "phase1.v2-pathA",
            "output_dt_seconds": cfg.output_dt_s,
            "topology_static_during_fault": topology_static_during_fault(model, runner),
            "collected_at": _dt.datetime.now(_dt.UTC).isoformat(),
        },
    )
    validate_sample(sg, expected_n_samples=cfg.expected_n_samples)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = cfg.out_dir / f"{uid}.h5"
    with h5py.File(h5_path, "w") as f:
        write_sample(f, sg)

    try:
        runner.close()
    except Exception:
        pass
    return uid


def collect_one_trajectory(cfg: CollectorConfig, *, rng: np.random.Generator) -> str:
    """Run one CloudPSS EMT trajectory with transient-error retry; persist to
    HDF5 in `cfg.out_dir`; return uid.

    Caller must have called ensure_authenticated() once before starting batched
    collection — this function calls cloudpss.Model.fetch directly.

    Retries on transient CloudPSS-side races (e.g. tag_resource_unique
    duplicate-entry from concurrent Model.update) and quota/timeout blips.
    Each retry uses the same rng so the action is reproducible across attempts.
    """
    last_exc: BaseException | None = None
    for attempt in range(cfg.max_retries + 1):
        try:
            return _collect_once(cfg, rng=rng)
        except Exception as exc:
            last_exc = exc
            if attempt >= cfg.max_retries or not _is_transient_error(exc):
                raise
            sleep_s = cfg.retry_backoff_base_s * (2 ** attempt)
            time.sleep(sleep_s)
    # unreachable
    raise RuntimeError(f"unreachable: retries exhausted, last={last_exc!r}")
