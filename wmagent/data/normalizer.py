"""Per-channel z-score normalization computed once on train split."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from wmagent.data.schema import CHANNELS


@dataclass(frozen=True)
class NormStats:
    mean: dict[str, float]
    std: dict[str, float]
    delta_std: dict[str, float]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NormStats):
            return NotImplemented
        return (self.mean == other.mean and self.std == other.std and self.delta_std == other.delta_std)


def compute_norm_stats(state: np.ndarray, delta: np.ndarray) -> NormStats:
    if state.ndim != 4 or state.shape[-1] != len(CHANNELS):
        raise ValueError(f"state must be (B,T,N,C); got {state.shape}")
    if delta.ndim != 4 or delta.shape[-1] != len(CHANNELS):
        raise ValueError(f"delta must be (B,T-1,N,C); got {delta.shape}")
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    delta_std: dict[str, float] = {}
    for i, ch in enumerate(CHANNELS):
        s = state[..., i].astype("float64")
        d = delta[..., i].astype("float64")
        mean[ch] = float(s.mean())
        std[ch] = float(s.std() + 1e-8)
        delta_std[ch] = float(d.std() + 1e-8)
    return NormStats(mean=mean, std=std, delta_std=delta_std)


def save_stats(stats: NormStats, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(stats), indent=2))


def load_stats(path: Path) -> NormStats:
    raw = json.loads(path.read_text())
    return NormStats(mean=raw["mean"], std=raw["std"], delta_std=raw["delta_std"])


def apply_norm(state: np.ndarray, stats: NormStats) -> np.ndarray:
    out = np.empty_like(state, dtype="float32")
    for i, ch in enumerate(CHANNELS):
        out[..., i] = (state[..., i] - stats.mean[ch]) / stats.std[ch]
    return out


def invert_norm(state_norm: np.ndarray, stats: NormStats) -> np.ndarray:
    out = np.empty_like(state_norm, dtype="float32")
    for i, ch in enumerate(CHANNELS):
        out[..., i] = state_norm[..., i] * stats.std[ch] + stats.mean[ch]
    return out
