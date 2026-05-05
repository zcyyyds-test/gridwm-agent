import torch

from wmagent.eval.metrics import per_channel_mae, zero_prediction_baseline_mae


def test_per_channel_mae_returns_dict_per_channel():
    pred = torch.zeros(8, 10, 4)
    gt = torch.ones(8, 10, 4)
    mae = per_channel_mae(pred, gt)
    assert set(mae.keys()) == {"wr", "LA", "VT", "IT"}
    for v in mae.values():
        assert abs(v - 1.0) < 1e-6


def test_zero_prediction_baseline_mae_uses_state_diff():
    s_t = torch.zeros(4, 10, 4)
    s_tp1 = torch.ones(4, 10, 4)
    baseline = zero_prediction_baseline_mae(s_t, s_tp1)
    for v in baseline.values():
        assert abs(v - 1.0) < 1e-6
