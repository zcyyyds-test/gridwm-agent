"""Random 80/10/10 train/val/test splits by trajectory uid (Phase 1).

Phase 2 will reintroduce parameter-disjoint or location-disjoint splits
once multi-location fault sweep is in place.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class TrajectoryRandomSplit:
    train_uids: tuple[str, ...]
    val_uids: tuple[str, ...]
    test_uids: tuple[str, ...]
    seed: int

    def split_for_uid(self, uid: str) -> SplitName:
        if uid in self.val_uids:
            return "val"
        if uid in self.test_uids:
            return "test"
        if uid in self.train_uids:
            return "train"
        raise ValueError(f"uid {uid!r} not in any split")


def build_random_split(
    *, uids: list[str], val_frac: float, test_frac: float, seed: int
) -> TrajectoryRandomSplit:
    if not (0.0 < val_frac < 1.0) or not (0.0 < test_frac < 1.0):
        raise ValueError("val_frac/test_frac must be in (0,1)")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val + test must leave a non-empty train fraction")
    rng = random.Random(seed)
    shuffled = list(uids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val = max(1, int(round(n * val_frac)))
    n_test = max(1, int(round(n * test_frac)))
    val = sorted(shuffled[:n_val])
    test = sorted(shuffled[n_val : n_val + n_test])
    train = sorted(shuffled[n_val + n_test :])
    return TrajectoryRandomSplit(
        train_uids=tuple(train),
        val_uids=tuple(val),
        test_uids=tuple(test),
        seed=seed,
    )


def save_split(s: TrajectoryRandomSplit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(s), indent=2))


def load_split(path: Path) -> TrajectoryRandomSplit:
    raw = json.loads(path.read_text())
    return TrajectoryRandomSplit(
        train_uids=tuple(raw["train_uids"]),
        val_uids=tuple(raw["val_uids"]),
        test_uids=tuple(raw["test_uids"]),
        seed=int(raw["seed"]),
    )
