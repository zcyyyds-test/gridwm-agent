from wmagent.world.power_grid import aggregate_risk_records


def test_aggregate_summarizes_fault_and_dominant_channel():
    records = [
        {
            "scenario": {"event_code": "FT-1"},
            "risk": {
                "score": 60,
                "risk_value": 1.0,
                "dominant_channel": "wr",
                "dominant_node": 2,
            },
        },
        {
            "scenario": {"event_code": "FT-1"},
            "risk": {
                "score": 90,
                "risk_value": 2.0,
                "dominant_channel": "wr",
                "dominant_node": 2,
            },
        },
        {
            "scenario": {"event_code": "FT-7"},
            "risk": {
                "score": 95,
                "risk_value": 3.0,
                "dominant_channel": "IT",
                "dominant_node": 5,
            },
        },
    ]
    agg = aggregate_risk_records(records)
    assert agg["by_fault"][0]["key"] == "FT-7"
    assert agg["by_fault"][1]["key"] == "FT-1"
    assert agg["by_dominant_channel"][0]["key"] == "IT"
    assert agg["by_dominant_node"][0]["key"] == "5"
