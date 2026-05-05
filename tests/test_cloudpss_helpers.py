"""Tests for cloudpss_helpers using mocked CloudPSS SDK objects.

Network-dependent tests (login, ensure_authenticated) are skipped in CI.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest

from wmagent.data.cloudpss_helpers import (
    apply_fault,
    extract_state_arrays,
    topology_static_during_fault,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plots_list(n_samples: int) -> list:
    """Build a list of plot dicts matching the structure of result.getPlots()."""
    gen_names = [f"#Gen{30 + i}.wr:{i}" for i in range(10)]
    bus_names = ["Ib37:A", "Ib37:B", "Ib37:C"]

    def _traces(names, n):
        return [{"name": nm, "x": list(range(n)), "y": [float(j) for j in range(n)]}
                for nm in names]

    plots = []
    for title in ("wr", "LA", "VT", "IT"):
        plots.append({"data": {"title": title, "traces": _traces(gen_names, n_samples)}})
    # Ib37 with 3 bus-phase traces
    plots.append({"data": {"title": "Ib37", "traces": _traces(bus_names, n_samples)}})
    return plots


# ---------------------------------------------------------------------------
# apply_fault
# ---------------------------------------------------------------------------

def test_apply_fault_calls_getComponentByKey_and_updateComponent():
    model = MagicMock()
    fault_component = MagicMock()
    fault_component.args = {"existing_key": "existing_value"}
    model.getComponentByKey.return_value = fault_component

    apply_fault(model, fs=1.0, fe=1.1, ft=7, chg_ohm=0.5)

    model.getComponentByKey.assert_called_once_with("canvas_0_965")
    model.updateComponent.assert_called_once()
    call_args = model.updateComponent.call_args
    assert call_args.args[0] == "canvas_0_965"
    passed_args = call_args.kwargs["args"]
    assert passed_args["fs"] == {"source": "1.0", "ɵexp": ""}
    assert passed_args["fe"] == {"source": "1.1", "ɵexp": ""}
    assert passed_args["ft"] == "7"
    assert passed_args["chg"] == {"source": "0.5", "ɵexp": ""}
    # Existing args should be preserved via **fault.args spread
    assert passed_args["existing_key"] == "existing_value"


def test_apply_fault_accepts_custom_fault_key():
    model = MagicMock()
    model.getComponentByKey.return_value = MagicMock(args={})

    apply_fault(model, fs=2.0, fe=2.15, ft=1, chg_ohm=0.01, fault_key="custom_key")

    model.getComponentByKey.assert_called_once_with("custom_key")
    assert model.updateComponent.call_args.args[0] == "custom_key"


# ---------------------------------------------------------------------------
# extract_state_arrays
# ---------------------------------------------------------------------------

def test_extract_state_arrays_returns_correct_shape():
    plots = _make_plots_list(n_samples=15001)
    state, bus_obs = extract_state_arrays(plots)
    assert state.shape == (15001, 10, 4)
    assert bus_obs.shape == (15001, 3)
    assert state.dtype == np.float32
    assert bus_obs.dtype == np.float32


def test_extract_state_arrays_small():
    plots = _make_plots_list(n_samples=50)
    state, bus_obs = extract_state_arrays(plots)
    assert state.shape == (50, 10, 4)
    assert bus_obs.shape == (50, 3)


def test_extract_state_arrays_raises_on_missing_plot():
    plots = _make_plots_list(n_samples=100)
    # Remove the 'IT' plot entry
    plots = [p for p in plots if p["data"]["title"] != "IT"]
    with pytest.raises(RuntimeError, match="'IT'"):
        extract_state_arrays(plots)


def test_extract_state_arrays_raises_on_too_few_traces():
    plots = _make_plots_list(n_samples=100)
    # Truncate traces in the 'wr' plot to fewer than 10
    for p in plots:
        if p["data"]["title"] == "wr":
            p["data"]["traces"] = p["data"]["traces"][:5]
    with pytest.raises(RuntimeError, match="only 5 traces"):
        extract_state_arrays(plots)


# ---------------------------------------------------------------------------
# topology_static_during_fault
# ---------------------------------------------------------------------------

def test_topology_static_returns_true_for_empty_logs():
    model = MagicMock()
    runner = MagicMock()
    # runner.result.getLogs() returns empty list
    runner.result.getLogs.return_value = []
    assert topology_static_during_fault(model, runner) is True


def test_topology_static_returns_true_for_irrelevant_logs():
    model = MagicMock()
    runner = MagicMock()
    runner.result.getLogs.return_value = [
        {"data": {"content": "Simulation started"}},
        {"data": {"content": "Convergence achieved"}},
    ]
    assert topology_static_during_fault(model, runner) is True


def test_topology_static_returns_false_for_permanent_line_open():
    model = MagicMock()
    runner = MagicMock()
    runner.result.getLogs.return_value = [
        {"data": {"content": "LINE_OPEN event: permanent breaker trip"}},
    ]
    assert topology_static_during_fault(model, runner) is False


def test_topology_static_falls_back_to_event_log():
    """When runner has no .result attribute, fall back to runner.event_log."""
    model = MagicMock()
    runner = MagicMock(spec=[])  # no attributes at all
    runner.event_log = []
    # spec=[] means hasattr(runner, 'result') is False → falls back to event_log
    assert topology_static_during_fault(model, runner) is True


# ---------------------------------------------------------------------------
# Skipped network test (documents expected login behavior)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="requires live CloudPSS network — not run in CI")
def test_login_returns_jwt_token():
    """login() should call accountChallenge + createAccountToken via GraphQL
    and return a JWT string.

    To run manually:
        CLOUDPSS_USERNAME=xxx CLOUDPSS_PASSWORD=yyy python -m pytest -k test_login -s
    """
    import os
    from wmagent.data.cloudpss_helpers import login

    token = login(
        username=os.environ["CLOUDPSS_USERNAME"],
        password=os.environ["CLOUDPSS_PASSWORD"],
        api_url=os.environ.get("CLOUDPSS_API_URL", "http://load.ddns.cloudpss.net/"),
    )
    assert isinstance(token, str)
    assert len(token) > 20
