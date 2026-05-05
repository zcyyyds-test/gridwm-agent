"""CloudPSS SDK call-chain helpers used by collector and verification scripts.

Mirrors /mnt/d/SCU/cloudpss-simulation/run_simulation.py — that script is
the working precedent against the live CloudPSS IEEE39 model.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

import numpy as np


# Channel order on packed state arrays. Must match wmagent.data.schema.CHANNELS.
# Real CloudPSS plot titles are localised, e.g. '功角-LA[Rad]' /
# '机端电压-VT[p.u.]' / '37号电机三相电流-Ib37'. We classify by substring
# match on the channel tag, with `Ib37` checked first so the trailing
# 'IT' substring of '机端电流-IT[p.u.]' doesn't bleed into bus_obs.
_BUS_OBS_TAG = "Ib37"
_PLOT_TAG_TO_CHANNEL_INDEX = (
    ("wr", 0),
    ("LA", 1),
    ("VT", 2),
    ("IT", 3),
)
_DEFAULT_FAULT_KEY = "canvas_0_965"


def _classify_plot_title(title: str) -> tuple[str, int] | None:
    """Return ('bus_obs', -1) for the bus-37 obs plot, or ('state', channel_idx)
    for a state channel, or None if the title doesn't match a known channel
    (e.g. the redundant RA[Deg] plot that we intentionally drop)."""
    if _BUS_OBS_TAG in title:
        return ("bus_obs", -1)
    if "RA" in title and "Deg" in title:
        return None  # explicitly drop RA[Deg] — derivable from LA[Rad]
    for tag, idx in _PLOT_TAG_TO_CHANNEL_INDEX:
        if tag in title:
            return ("state", idx)
    return None


def login(username: str, password: str, api_url: str) -> str:
    """GraphQL accountChallenge → createAccountToken → JWT (31-day validity).

    Mirrors run_simulation.py:login().
    """
    gql_url = api_url.rstrip("/") + "/graphql"
    headers = {"Content-Type": "application/json"}

    def _post(query: str, variables: dict | None = None) -> dict:
        payload = json.dumps({"query": query, "variables": variables or {}}).encode()
        req = urllib.request.Request(gql_url, payload, headers)
        return json.loads(urllib.request.urlopen(req).read())

    r1 = _post(
        'query ($input: String!) { accountChallenge(input:{name: $input, scopes:["*browser"]}){ id, groups{ items { id, type } } } }',
        {"input": username},
    )
    ch = r1["data"]["accountChallenge"]
    pwd_item = next(
        i for g in ch["groups"] for i in g["items"] if i["type"] == "PASSWORD"
    )
    r2 = _post(
        "mutation ($input: CreateAccountTokenInput!){ createAccountToken(input: $input){ token } }",
        {
            "input": {
                "id": ch["id"],
                "answers": [{"id": pwd_item["id"], "answer": {"password": password}}],
            }
        },
    )
    return r2["data"]["createAccountToken"]["token"]


def ensure_authenticated() -> str:
    """Return a CloudPSS token. Uses CLOUDPSS_TOKEN if set; else logs in via env vars."""
    import cloudpss

    token = os.environ.get("CLOUDPSS_TOKEN")
    if not token:
        username = os.environ["CLOUDPSS_USERNAME"]
        password = os.environ["CLOUDPSS_PASSWORD"]
        api_url = os.environ.get("CLOUDPSS_API_URL", "http://load.ddns.cloudpss.net/")
        os.environ["CLOUDPSS_API_URL"] = api_url
        token = login(username, password, api_url)
    cloudpss.setToken(token)
    return token


def apply_fault(
    model: Any,
    *,
    fs: float,
    fe: float,
    ft: int,
    chg_ohm: float,
    fault_key: str = _DEFAULT_FAULT_KEY,
) -> None:
    """Configure the IEEE39 model's single fault element on `fault_key`.

    fs / fe in seconds; ft is CloudPSS code (1=A-g, 3=AB, 7=ABC); chg_ohm Ω.
    """
    fault = model.getComponentByKey(fault_key)
    model.updateComponent(
        fault_key,
        args={
            **fault.args,
            "fs": {"source": str(fs), "ɵexp": ""},
            "fe": {"source": str(fe), "ɵexp": ""},
            "ft": str(ft),
            "chg": {"source": str(chg_ohm), "ɵexp": ""},
        },
    )


def set_emt_params(
    model: Any,
    *,
    end_time: float | None = None,
    step_time: float | None = None,
    n_cpu: int | None = None,
) -> None:
    """Update the EMT job's args (model.jobs filtered by 'emtp' substring)."""
    emt_job = next((j for j in model.jobs if "emtp" in j["rid"]), None)
    if emt_job is None:
        raise RuntimeError("model has no EMT job")
    if end_time is not None:
        emt_job["args"]["end_time"] = end_time
    if step_time is not None:
        emt_job["args"]["step_time"] = step_time
    if n_cpu is not None:
        emt_job["args"]["n_cpu"] = n_cpu


def extract_state_arrays(plots_list: list) -> tuple[np.ndarray, np.ndarray]:
    """Pack a CloudPSS plots list into (state[T,N=10,C=4], bus_obs[T,3]).

    `plots_list` is the return of `result.getPlots()` — a list of plot dicts;
    each has `data.title` (localised, e.g. '功角-LA[Rad]') and `data.traces`
    = list of {name, x, y}. We classify each plot by `_classify_plot_title`
    and pull out the 10 generator traces (sorted by trace name for
    determinism) for state channels and 3 phase traces from Ib37 for
    bus_obs. The redundant RA[Deg] plot is dropped — derivable from LA[Rad].

    Channel order on the output state: [wr, LA, VT, IT].
    """
    state_traces: list[list[dict] | None] = [None, None, None, None]
    bus_traces: list[dict] | None = None
    for p in plots_list:
        data = p.get("data", {}) if isinstance(p, dict) else {}
        title = data.get("title")
        if not isinstance(title, str):
            continue
        cls = _classify_plot_title(title)
        if cls is None:
            continue
        kind, idx = cls
        traces = data.get("traces", [])
        if kind == "state":
            state_traces[idx] = traces
        elif kind == "bus_obs":
            bus_traces = traces

    n_samples: int | None = None
    state_channels: list[np.ndarray] = [None, None, None, None]  # type: ignore
    for ch_idx, (tag, _) in enumerate(_PLOT_TAG_TO_CHANNEL_INDEX):
        traces = state_traces[ch_idx]
        if not traces:
            raise RuntimeError(f"state channel '{tag}' missing from CloudPSS result")
        traces_sorted = sorted(traces, key=lambda t: t.get("name", ""))
        if len(traces_sorted) < 10:
            raise RuntimeError(
                f"channel '{tag}' has only {len(traces_sorted)} traces, expected 10"
            )
        col_arrays = [np.asarray(tr.get("y", []), dtype="float32") for tr in traces_sorted[:10]]
        stacked = np.stack(col_arrays, axis=1)  # (T, 10)
        if n_samples is None:
            n_samples = stacked.shape[0]
        elif stacked.shape[0] != n_samples:
            raise RuntimeError(
                f"channel '{tag}' length {stacked.shape[0]} != prior {n_samples}"
            )
        state_channels[ch_idx] = stacked

    state = np.stack(state_channels, axis=-1)  # (T, 10, 4)

    if bus_traces is None or len(bus_traces) < 3:
        n_b = 0 if bus_traces is None else len(bus_traces)
        raise RuntimeError(f"bus_obs plot '{_BUS_OBS_TAG}' has only {n_b} traces, expected 3")
    bus_traces_sorted = sorted(bus_traces, key=lambda t: t.get("name", ""))
    bus_cols = [np.asarray(tr.get("y", []), dtype="float32") for tr in bus_traces_sorted[:3]]
    bus_obs = np.stack(bus_cols, axis=1).astype("float32", copy=False)  # (T, 3)

    return state, bus_obs


def topology_static_during_fault(model: Any, runner: Any) -> bool:
    """For Phase 1 with the single fault resistor element and CloudPSS's default
    fault model (timed insertion of a small resistor, then removal), no permanent
    line-open events are expected. We inspect the runner's logs for any line-open
    event marked permanent and return False if any exist; True otherwise.
    """
    try:
        logs = runner.result.getLogs() if hasattr(runner, "result") else []
    except Exception:
        logs = getattr(runner, "event_log", []) or []
    for entry in logs:
        d = entry.get("data", entry) if isinstance(entry, dict) else {}
        content = (d.get("content") or "") if isinstance(d, dict) else ""
        if "line_open" in content.lower() and "permanent" in content.lower():
            return False
    return True
