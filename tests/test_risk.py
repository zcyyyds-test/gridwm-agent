import torch

from wmagent.eval.risk import normalize_risk_scores, rollout_risk_features, rollout_risk_value


def test_rollout_risk_value_orders_larger_future_deviation_higher():
    quiet = torch.zeros(1, 4, 10, 4)
    risky = torch.zeros(1, 4, 10, 4)
    risky[:, 1:, 3, 2] = torch.tensor([0.2, 0.8, 1.4])
    values = rollout_risk_value(torch.cat([quiet, risky], dim=0))
    assert values[1] > values[0]


def test_rollout_risk_features_report_dominant_node_channel_and_band():
    rollout = torch.zeros(1, 5, 10, 4)
    rollout[0, 3, 7, 1] = 2.0
    features = rollout_risk_features(rollout, scores=torch.tensor([91.0]))
    assert features[0]["score"] == 91
    assert features[0]["band"] == "CRITICAL"
    assert features[0]["dominant_node"] == 7
    assert features[0]["dominant_channel"] == "LA"
    assert features[0]["dominant_step"] == 3


def test_normalize_risk_scores_maps_batch_to_display_range():
    scores = normalize_risk_scores(torch.tensor([0.0, 5.0, 10.0]), floor=40, ceil=90)
    assert scores.tolist() == [40.0, 65.0, 90.0]
