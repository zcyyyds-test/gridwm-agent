"""Contract conformance tests for the CartPole second-domain adapter.

These tests prove that ``CartPoleWorldModelSystem`` satisfies the same
``WorldModelSystem`` Protocol as ``PowerGridWorldModelSystem`` -- the whole
point of A3 is that the system contract is domain-general.
"""
from __future__ import annotations

import torch

from wmagent.world.base import WorldEvent, WorldModelSystem, WorldState
from wmagent.world.cartpole import CartPoleWorldModelSystem


def test_cartpole_satisfies_world_model_system_protocol():
    system = CartPoleWorldModelSystem.from_defaults(horizon=5)
    _: WorldModelSystem = system
    assert system.domain == "cartpole"


def test_anchor_state_and_candidate_events_have_consistent_domain():
    system = CartPoleWorldModelSystem.from_defaults(horizon=4)
    state = system.anchor_state(seed=2026, anchor_index=0)
    events = system.candidate_events()
    assert state.domain == system.domain
    assert state.tensor.shape == (4,)
    assert len(events) == 2 ** 4
    assert all(event.domain == system.domain for event in events)
    assert all(event.tensor.shape == (4,) for event in events)


def test_imagine_returns_horizon_plus_one_states():
    system = CartPoleWorldModelSystem.from_defaults(horizon=5)
    state = system.anchor_state(seed=2026)
    event = system.candidate_events()[0]
    future = system.imagine(state, event)
    assert future.rollout.shape == (5 + 1, 4)
    assert future.state is state
    assert future.event is event


def test_imagine_many_preserves_event_order():
    system = CartPoleWorldModelSystem.from_defaults(horizon=3)
    state = system.anchor_state(seed=2026)
    events = system.candidate_events()
    futures = system.imagine_many(state, events)
    assert len(futures) == len(events)
    for future, event in zip(futures, events):
        assert future.event is event
        assert future.rollout.shape == (3 + 1, 4)


def test_score_returns_risk_signal_with_features():
    system = CartPoleWorldModelSystem.from_defaults(horizon=5)
    state = WorldState(tensor=torch.zeros(4), domain=system.domain)
    event = system.candidate_events()[0]
    risk = system.score(system.imagine(state, event))
    assert isinstance(risk.score, int)
    assert risk.band in {"low", "med", "high"}
    assert risk.value >= 0.0
    assert {"max_angle_rad", "terminated", "first_violation_step"} <= set(risk.features)


def test_higher_initial_pole_angle_yields_higher_risk():
    system = CartPoleWorldModelSystem.from_defaults(horizon=5)
    safe_state = WorldState(
        tensor=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        domain=system.domain,
    )
    tipping_state = WorldState(
        tensor=torch.tensor([0.0, 0.0, 0.18, 1.0]),
        domain=system.domain,
    )
    event = system.candidate_events()[0]
    safe_risk = system.score(system.imagine(safe_state, event))
    tipping_risk = system.score(system.imagine(tipping_state, event))
    assert tipping_risk.value > safe_risk.value


def test_search_ranks_by_descending_risk_and_caps_top_k():
    system = CartPoleWorldModelSystem.from_defaults(horizon=5)
    state = system.anchor_state(seed=2026)
    events = system.candidate_events()
    results = system.search(state, events, top_k=3)
    assert len(results) == 3
    risks = [r.risk.value for r in results]
    assert risks == sorted(risks, reverse=True)
    for rank, result in enumerate(results, start=1):
        assert result.rank == rank


def test_imagine_rejects_wrong_domain():
    system = CartPoleWorldModelSystem.from_defaults(horizon=3)
    bad_state = WorldState(tensor=torch.zeros(4), domain="power_grid")
    event = system.candidate_events()[0]
    try:
        system.imagine(bad_state, event)
    except ValueError as exc:
        assert "domain" in str(exc)
    else:
        raise AssertionError("expected ValueError on cross-domain state")
