"""Build a tiny anchor-state cache so the bundled demo_run quickstart
works on a fresh clone without ``data/raw/``.

Run once on a machine that has the full collected dataset; the resulting
``outputs/demo_run/anchors.pt`` is committed and shipped with the repo.
``PowerGridWorldModelSystem.anchor_state`` will read it transparently
when ``data/raw/`` is empty.

The cache holds the fixed (split, seed, horizon) combinations the
README quickstart and the FastAPI default request use, plus a small
range of anchor indices, so users can sweep ``--anchor-index 0..15``
without re-collecting.
"""
from __future__ import annotations

from pathlib import Path

import torch

from wmagent.data.dataset import WMAgentDataset


_DEFAULTS = {
    "splits": ("val", "test"),
    "seed": 2026,
    "horizon": 10,
    "n_anchors": 16,
    "raw_dir": Path("data/raw"),
    "splits_path": Path("data/splits.json"),
    "norm_stats_path": Path("data/norm_stats.json"),
    "out_path": Path("outputs/demo_run/anchors.pt"),
}


def main() -> int:
    anchors: list[dict] = []
    for split in _DEFAULTS["splits"]:
        ds = WMAgentDataset(
            raw_dir=_DEFAULTS["raw_dir"],
            splits_path=_DEFAULTS["splits_path"],
            norm_stats_path=_DEFAULTS["norm_stats_path"],
            split=split,  # type: ignore[arg-type]
            pairs_per_traj_per_epoch=max(_DEFAULTS["n_anchors"], 64),
            seed=_DEFAULTS["seed"],
            fault_window_frac=1.0,
            rollout_horizon=_DEFAULTS["horizon"],
        )
        for idx in range(_DEFAULTS["n_anchors"]):
            item = ds[idx]
            anchors.append(
                {
                    "split": split,
                    "seed": _DEFAULTS["seed"],
                    "horizon": _DEFAULTS["horizon"],
                    "anchor_index": idx,
                    "state_t": item["state_t"].clone(),
                    "action_global": item["action_global"].clone(),
                    "fault_window_active": float(item["fault_window_active"]),
                }
            )
    payload = {"version": 1, "anchors": anchors}
    _DEFAULTS["out_path"].parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, _DEFAULTS["out_path"])
    print(f"wrote {_DEFAULTS['out_path']} with {len(anchors)} anchors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
