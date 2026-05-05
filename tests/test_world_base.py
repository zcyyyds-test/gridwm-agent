from wmagent.world.base import ImaginedFuture, RiskSignal, SearchResult, WorldEvent, WorldState


def test_world_model_system_records_keep_domain_metadata():
    state = WorldState(tensor="state", domain="toy", metadata={"anchor": 1})
    event = WorldEvent(tensor="event", domain="toy", metadata={"kind": "fault"})
    future = ImaginedFuture(rollout="future", state=state, event=event)
    risk = RiskSignal(score=88, band="CRITICAL", value=1.5, features={"node": 3})
    result = SearchResult(rank=1, future=future, risk=risk)
    assert result.future.state.metadata["anchor"] == 1
    assert result.future.event.metadata["kind"] == "fault"
    assert result.risk.features["node"] == 3
