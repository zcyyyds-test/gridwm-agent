"""Compute per-channel mean/std + Δs std on the train split, write
data/norm_stats.json. Run once after collection completes.

Reads data/raw/*.h5 + data/splits.json, filters trajectories whose uid
is in the train split, and aggregates statistics streaming-friendly
(per-trajectory) so we don't load all 1k × 15001 × 10 × 4 = ~24 GB into
RAM at once.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from wmagent.data.normalizer import NormStats, save_stats
from wmagent.data.schema import CHANNELS, read_sample
from wmagent.data.splits import load_split


class _RunningStats:
    """Single-pass running mean/var via Welford's algorithm. Per-channel scalar."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0  # sum of squared differences

    def update(self, values: np.ndarray) -> None:
        flat = values.astype("float64", copy=False).ravel()
        for batch in np.array_split(flat, max(1, flat.size // 100_000)):
            n_b = batch.size
            if n_b == 0:
                continue
            mean_b = float(batch.mean())
            m2_b = float(((batch - mean_b) ** 2).sum())
            n_total = self.n + n_b
            delta = mean_b - self.mean
            self.mean = (self.n * self.mean + n_b * mean_b) / n_total
            self.m2 += m2_b + delta * delta * self.n * n_b / n_total
            self.n = n_total

    def std(self) -> float:
        if self.n < 2:
            return 1.0
        return float(np.sqrt(self.m2 / (self.n - 1)) + 1e-8)


def main() -> None:
    split = load_split(Path("data/splits.json"))
    train_uids = set(split.train_uids)
    raw_dir = Path("data/raw")

    state_stats = {ch: _RunningStats() for ch in CHANNELS}
    delta_stats = {ch: _RunningStats() for ch in CHANNELS}

    n_train_traj = 0
    h5_files = sorted(raw_dir.glob("*.h5"))
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            for uid_full in list(f.keys()):
                uid = uid_full.removeprefix("sample_")
                if uid not in train_uids:
                    continue
                sg = read_sample(f, uid)
                for i, ch in enumerate(CHANNELS):
                    state_stats[ch].update(sg.state[..., i])
                    delta_stats[ch].update(np.diff(sg.state[..., i], axis=0))
                n_train_traj += 1

    stats = NormStats(
        mean={ch: float(state_stats[ch].mean) for ch in CHANNELS},
        std={ch: float(state_stats[ch].std()) for ch in CHANNELS},
        delta_std={ch: float(delta_stats[ch].std()) for ch in CHANNELS},
    )
    out = Path("data/norm_stats.json")
    save_stats(stats, out)

    print(json.dumps(
        {
            "channels": list(CHANNELS),
            "n_train_traj": n_train_traj,
            "n_h5_files_scanned": len(h5_files),
            "out_path": str(out),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
