import json
from pathlib import Path

import pytest

from wmagent.data.splits import TrajectoryRandomSplit, build_random_split, save_split, load_split


@pytest.fixture
def uids():
    return [f"u{i:04d}" for i in range(100)]


def test_split_proportions(uids):
    s = build_random_split(uids=uids, val_frac=0.1, test_frac=0.1, seed=42)
    # With 100 uids and round(100*0.1)=10 each, expect exactly 80/10/10
    assert len(s.val_uids) == 10
    assert len(s.test_uids) == 10
    assert len(s.train_uids) == 80


def test_splits_are_disjoint(uids):
    s = build_random_split(uids=uids, val_frac=0.1, test_frac=0.1, seed=42)
    train_set = set(s.train_uids)
    val_set = set(s.val_uids)
    test_set = set(s.test_uids)
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_split_union_is_complete(uids):
    s = build_random_split(uids=uids, val_frac=0.1, test_frac=0.1, seed=42)
    assert set(s.train_uids) | set(s.val_uids) | set(s.test_uids) == set(uids)


def test_split_seed_deterministic(uids):
    a = build_random_split(uids=uids, val_frac=0.1, test_frac=0.1, seed=42)
    b = build_random_split(uids=uids, val_frac=0.1, test_frac=0.1, seed=42)
    assert a.train_uids == b.train_uids
    assert a.val_uids == b.val_uids
    assert a.test_uids == b.test_uids


def test_split_json_round_trip(uids, tmp_path: Path):
    s = build_random_split(uids=uids, val_frac=0.1, test_frac=0.1, seed=42)
    p = tmp_path / "splits.json"
    save_split(s, p)
    loaded = load_split(p)
    assert loaded == s


def test_split_for_uid(uids):
    s = build_random_split(uids=uids, val_frac=0.1, test_frac=0.1, seed=42)
    for uid in s.val_uids:
        assert s.split_for_uid(uid) == "val"
    for uid in s.test_uids:
        assert s.split_for_uid(uid) == "test"
    for uid in s.train_uids:
        assert s.split_for_uid(uid) == "train"
    with pytest.raises(ValueError):
        s.split_for_uid("nonexistent_uid")
