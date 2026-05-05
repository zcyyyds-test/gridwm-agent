"""scripts/train.py — Phase 1 training entry point."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from wmagent.train.loop import TrainConfig, train


def main() -> None:
    train_cfg = yaml.safe_load(Path("configs/train.yaml").read_text())
    model_cfg_path = Path(os.environ.get("GRIDWM_MODEL_CONFIG", "configs/model_v2.yaml"))
    if not model_cfg_path.exists():
        model_cfg_path = Path("configs/model.yaml")
    model_cfg = yaml.safe_load(model_cfg_path.read_text())

    cfg = TrainConfig(
        batch_size=int(train_cfg["batch_size"]),
        pairs_per_traj_per_epoch=int(train_cfg["pairs_per_traj_per_epoch"]),
        epochs=int(train_cfg["epochs"]),
        early_stop_patience=int(train_cfg["early_stop_patience"]),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        seed=int(train_cfg["seed"]),
        dataloader_workers=int(train_cfg["dataloader_workers"]),
        output_root=Path(train_cfg["output_root"]),
        rollout_horizon=int(train_cfg.get("rollout_horizon", 1)),
        fault_window_frac=float(train_cfg.get("fault_window_frac", 0.30)),
        fault_loss_weight=float(train_cfg.get("fault_loss_weight", 1.0)),
        direction_loss_weight=float(train_cfg.get("direction_loss_weight", 0.0)),
        corr_loss_weight=float(train_cfg.get("corr_loss_weight", 0.0)),
        grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
        log_every_steps=int(train_cfg.get("log_every_steps", 25)),
    )
    run_dir = train(
        cfg,
        raw_dir=Path("data/raw"),
        splits_path=Path("data/splits.json"),
        norm_stats_path=Path("data/norm_stats.json"),
        graph_path=Path("data/graph_ieee39.h5"),
        model_kwargs=model_cfg,
    )
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    main()
