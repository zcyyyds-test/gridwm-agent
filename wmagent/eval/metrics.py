"""Eval metrics: per-channel MAE/RMSE, zero-prediction baseline, rollout."""
from __future__ import annotations

import torch

from wmagent.data.schema import CHANNELS


def per_channel_mae(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    out = {}
    for i, ch in enumerate(CHANNELS):
        out[ch] = float((pred[..., i] - gt[..., i]).abs().mean().item())
    return out


def per_channel_rmse(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    out = {}
    for i, ch in enumerate(CHANNELS):
        out[ch] = float(((pred[..., i] - gt[..., i]) ** 2).mean().sqrt().item())
    return out


def zero_prediction_baseline_mae(s_t: torch.Tensor, s_tp1: torch.Tensor) -> dict[str, float]:
    """Δ ≡ 0, i.e. predict s_{t+1} = s_t. MAE of this baseline against ground truth."""
    return per_channel_mae(s_t, s_tp1)


def per_channel_ratio(num: dict[str, float], den: dict[str, float], eps: float = 1e-12) -> dict[str, float]:
    return {ch: float(num[ch] / max(den[ch], eps)) for ch in CHANNELS}


def direction_accuracy(pred_delta: torch.Tensor, gt_delta: torch.Tensor) -> dict[str, float]:
    """Per-channel sign agreement for non-zero ground-truth deltas."""
    out = {}
    for i, ch in enumerate(CHANNELS):
        gt = gt_delta[..., i]
        pred = pred_delta[..., i]
        mask = gt.abs() > 1e-12
        if not bool(mask.any()):
            out[ch] = float("nan")
            continue
        out[ch] = float((torch.sign(pred[mask]) == torch.sign(gt[mask])).float().mean().item())
    return out


def pearson_correlation(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    """Per-channel Pearson correlation over all leading dimensions."""
    out = {}
    for i, ch in enumerate(CHANNELS):
        p = pred[..., i].reshape(-1)
        g = gt[..., i].reshape(-1)
        p = p - p.mean()
        g = g - g.mean()
        denom = p.std().clamp_min(1e-12) * g.std().clamp_min(1e-12)
        out[ch] = float(((p * g).mean() / denom).clamp(-1.0, 1.0).item())
    return out
