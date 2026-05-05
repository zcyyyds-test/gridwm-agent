"""Training loop for wm-agent dynamics and latent world-model variants."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from wmagent.data.dataset import wm-agentDataset
from wmagent.models.base import StaticGraph
from wmagent.models.factory import build_world_model
from wmagent.train.logging import setup_run_logger


@dataclass
class TrainConfig:
    batch_size: int
    pairs_per_traj_per_epoch: int
    epochs: int
    early_stop_patience: int
    lr: float
    weight_decay: float
    seed: int
    dataloader_workers: int
    output_root: Path
    rollout_horizon: int = 1
    fault_window_frac: float = 0.30
    fault_loss_weight: float = 1.0
    direction_loss_weight: float = 0.0
    corr_loss_weight: float = 0.0
    grad_clip_norm: float = 1.0
    log_every_steps: int = 25


def load_static_graph(path: Path, device: torch.device) -> StaticGraph:
    with h5py.File(path, "r") as f:
        return StaticGraph(
            edge_index=torch.tensor(f["edge_index"][...], dtype=torch.long, device=device),
            edge_attr=torch.tensor(f["edge_attr"][...], dtype=torch.float32, device=device),
            node_attr=torch.tensor(f["node_attr"][...], dtype=torch.float32, device=device),
        )


def curriculum_for_epoch(epoch: int, max_horizon: int) -> tuple[int, float]:
    """Return (active_horizon, teacher_forcing_ratio)."""
    if epoch < 10:
        return min(max_horizon, 3), 1.0
    if epoch < 25:
        return min(max_horizon, 5), 0.7
    return max_horizon, 0.3


def rollout_model_sequence(
    model: torch.nn.Module,
    state_t: torch.Tensor,
    action_sequence: torch.Tensor,
    graph: StaticGraph,
    *,
    teacher_sequence: torch.Tensor | None,
    teacher_forcing_ratio: float,
) -> torch.Tensor:
    """Autoregressively predict normalized states, including the initial state."""
    if hasattr(model, "rollout"):
        return model.rollout(
            state_t,
            action_sequence=action_sequence,
            graph=graph,
            teacher_sequence=teacher_sequence,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
    states = [state_t]
    state = state_t
    for step in range(action_sequence.shape[1]):
        next_state = model(state, action_global=action_sequence[:, step], graph=graph)
        states.append(next_state)
        if teacher_sequence is not None and teacher_forcing_ratio > 0.0:
            use_teacher = (
                torch.rand(state.shape[0], 1, 1, device=state.device) < teacher_forcing_ratio
            )
            state = torch.where(use_teacher, teacher_sequence[:, step + 1], next_state)
        else:
            state = next_state
    return torch.stack(states, dim=1)


def weighted_rollout_mae_loss(
    pred_sequence: torch.Tensor,
    gt_sequence: torch.Tensor,
    fault_window_sequence: torch.Tensor,
    *,
    fault_loss_weight: float,
) -> torch.Tensor:
    err = (pred_sequence[:, 1:] - gt_sequence[:, 1:]).abs().mean(dim=(-1, -2))
    weights = 1.0 + (float(fault_loss_weight) - 1.0) * fault_window_sequence.float()
    return (err * weights).sum() / weights.sum().clamp_min(1.0)


def direction_loss(pred_sequence: torch.Tensor, gt_sequence: torch.Tensor) -> torch.Tensor:
    pred_delta = pred_sequence[:, 1:] - pred_sequence[:, :-1]
    gt_delta = gt_sequence[:, 1:] - gt_sequence[:, :-1]
    return torch.relu(-pred_delta * gt_delta).mean()


def corr_loss(pred_sequence: torch.Tensor, gt_sequence: torch.Tensor) -> torch.Tensor:
    pred = pred_sequence[:, 1:].flatten(0, -2)
    gt = gt_sequence[:, 1:].flatten(0, -2)
    pred = pred - pred.mean(dim=0, keepdim=True)
    gt = gt - gt.mean(dim=0, keepdim=True)
    denom = pred.std(dim=0).clamp_min(1e-6) * gt.std(dim=0).clamp_min(1e-6)
    corr = ((pred * gt).mean(dim=0) / denom).clamp(-1.0, 1.0)
    return 1.0 - corr.mean()


def world_model_loss(
    pred_sequence: torch.Tensor,
    gt_sequence: torch.Tensor,
    fault_window_sequence: torch.Tensor,
    cfg: TrainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    rollout_loss = weighted_rollout_mae_loss(
        pred_sequence,
        gt_sequence,
        fault_window_sequence,
        fault_loss_weight=cfg.fault_loss_weight,
    )
    d_loss = direction_loss(pred_sequence, gt_sequence)
    c_loss = corr_loss(pred_sequence, gt_sequence)
    total = (
        rollout_loss
        + cfg.direction_loss_weight * d_loss
        + cfg.corr_loss_weight * c_loss
    )
    return total, {
        "rollout": float(rollout_loss.detach().item()),
        "direction": float(d_loss.detach().item()),
        "corr": float(c_loss.detach().item()),
    }


def train(
    cfg: TrainConfig,
    *,
    raw_dir: Path,
    splits_path: Path,
    norm_stats_path: Path,
    graph_path: Path,
    model_kwargs: dict[str, Any],
) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # uuid (not torch.rand-after-manual_seed) so smoke + full runs land in
    # distinct directories even with the same fixed seed.
    run_id = f"run_{uuid.uuid4().hex[:10]}"
    run_dir = cfg.output_root / run_id
    log = setup_run_logger(run_dir)
    tb = SummaryWriter(run_dir / "tb")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device={device} run_dir={run_dir}")

    train_ds = wm-agentDataset(
        raw_dir=raw_dir, splits_path=splits_path, norm_stats_path=norm_stats_path,
        split="train", pairs_per_traj_per_epoch=cfg.pairs_per_traj_per_epoch, seed=cfg.seed,
        fault_window_frac=cfg.fault_window_frac,
        rollout_horizon=cfg.rollout_horizon,
    )
    val_ds = wm-agentDataset(
        raw_dir=raw_dir, splits_path=splits_path, norm_stats_path=norm_stats_path,
        split="val", pairs_per_traj_per_epoch=cfg.pairs_per_traj_per_epoch, seed=cfg.seed + 1,
        fault_window_frac=0.0,
        rollout_horizon=cfg.rollout_horizon,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.dataloader_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.dataloader_workers, pin_memory=True,
    )

    graph = load_static_graph(graph_path, device)
    log.info(
        "loss = latent/world-model rollout "
        f"(max_horizon={cfg.rollout_horizon}, fault_window_frac={cfg.fault_window_frac}, "
        f"fault_loss_weight={cfg.fault_loss_weight}, "
        f"direction_w={cfg.direction_loss_weight}, corr_w={cfg.corr_loss_weight})"
    )

    model = build_world_model(model_kwargs, graph).to(device)
    log.info(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_val = float("inf")
    bad_epochs = 0
    best_path = run_dir / "best.pt"

    for epoch in range(cfg.epochs):
        active_horizon, teacher_forcing_ratio = curriculum_for_epoch(epoch, cfg.rollout_horizon)
        model.train()
        train_losses = []
        for step_idx, batch in enumerate(train_loader, start=1):
            state_t = batch["state_t"].to(device, non_blocking=True)
            state_sequence = batch["state_sequence"][:, : active_horizon + 1].to(
                device, non_blocking=True
            )
            action_sequence = batch["action_sequence"][:, :active_horizon].to(
                device, non_blocking=True
            )
            fault_window_sequence = batch["fault_window_sequence"][:, :active_horizon].to(
                device, non_blocking=True
            )

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred_sequence = rollout_model_sequence(
                    model,
                    state_t,
                    action_sequence,
                    graph,
                    teacher_sequence=state_sequence,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                )
                loss, loss_parts = world_model_loss(
                    pred_sequence, state_sequence, fault_window_sequence, cfg
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
            opt.step()
            train_losses.append(float(loss.item()))
            if cfg.log_every_steps > 0 and step_idx % cfg.log_every_steps == 0:
                log.info(
                    f"epoch={epoch} step={step_idx}/{len(train_loader)} "
                    f"horizon={active_horizon} loss={float(loss.item()):.6f} "
                    f"rollout={loss_parts['rollout']:.6f} "
                    f"direction={loss_parts['direction']:.6f} corr={loss_parts['corr']:.6f}"
                )
        sched.step()

        model.eval()
        v_losses = []
        with torch.no_grad():
            for batch in val_loader:
                state_t = batch["state_t"].to(device)
                state_sequence = batch["state_sequence"][:, : active_horizon + 1].to(device)
                action_sequence = batch["action_sequence"][:, :active_horizon].to(device)
                fault_window_sequence = batch["fault_window_sequence"][:, :active_horizon].to(
                    device
                )
                pred_sequence = rollout_model_sequence(
                    model,
                    state_t,
                    action_sequence,
                    graph,
                    teacher_sequence=None,
                    teacher_forcing_ratio=0.0,
                )
                loss, _ = world_model_loss(
                    pred_sequence, state_sequence, fault_window_sequence, cfg
                )
                v_losses.append(float(loss.item()))
        v_loss = float(np.mean(v_losses))
        train_loss = float(np.mean(train_losses))
        tb.add_scalar("train/loss", train_loss, epoch)
        tb.add_scalar("val/loss", v_loss, epoch)
        tb.add_scalar("train/rollout_horizon", active_horizon, epoch)
        tb.add_scalar("train/teacher_forcing_ratio", teacher_forcing_ratio, epoch)
        log.info(
            f"epoch={epoch} horizon={active_horizon} teacher_forcing={teacher_forcing_ratio:.2f} "
            f"train_loss={train_loss:.6f} val_loss={v_loss:.6f}"
        )

        if v_loss < best_val:
            best_val = v_loss
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": model_kwargs,
                    "model_type": model_kwargs.get("model_type", "mpnn"),
                    "train_cfg": cfg.__dict__ | {"output_root": str(cfg.output_root)},
                    "epoch": epoch,
                    "best_val": best_val,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop_patience:
                log.info(f"early-stop at epoch {epoch}")
                break

    tb.close()
    log.info(f"best_val={best_val:.6f} best_path={best_path}")
    return run_dir
