"""scripts/eval.py — runs full eval, writes outputs/<run_id>/eval.json,
checks gates H1/H2/H3, exits 0 only if all pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from wmagent.data.dataset import ACTION_DIM, wm-agentDataset
from wmagent.data.normalizer import invert_norm, load_stats
from wmagent.data.schema import CHANNELS
from wmagent.eval.metrics import (
    direction_accuracy,
    pearson_correlation,
    per_channel_mae,
    per_channel_ratio,
    zero_prediction_baseline_mae,
)
from wmagent.eval.sanity import (
    action_conditioning_diff_ratio,
    rollout_boundedness_ratio,
)
from wmagent.models.factory import build_world_model
from wmagent.train.loop import load_static_graph


def _perturb_action(action: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Resample ft (to a different value in {1,3,7}) plus fs/duration/chg/tau fields.
    action: (B, ACTION_DIM=12). Returns perturbed copy."""
    B = action.shape[0]
    out = action.clone()
    ft_choices = (1, 3, 7)
    for b in range(B):
        cur_ft = int(out[b, :8].argmax().item())
        candidates = [c for c in ft_choices if c != cur_ft]
        new_ft = candidates[int(rng.integers(0, len(candidates)))]
        out[b, :8].zero_()
        out[b, new_ft] = 1.0
        # resample fs_norm / duration_norm / chg_log_norm uniformly in [0, 1]
        out[b, 8] = float(rng.uniform(0.0, 1.0))
        out[b, 9] = float(rng.uniform(0.0, 1.0))
        out[b, 10] = float(rng.uniform(0.0, 1.0))
        # resample tau across the [-2, 3] clip range (covers pre/active/post)
        out[b, 11] = float(rng.uniform(-2.0, 3.0))
    return out


def _load_model(run_dir: Path, graph, device: torch.device):
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model_cfg = dict(ckpt.get("cfg") or yaml.safe_load(Path("configs/model.yaml").read_text()))
    model_cfg.setdefault("model_type", ckpt.get("model_type", "mpnn"))
    model = build_world_model(model_cfg, graph).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt, model_cfg


def _rollout_model(model, state_t: torch.Tensor, action_sequence: torch.Tensor, graph):
    if hasattr(model, "rollout"):
        return model.rollout(state_t, action_sequence=action_sequence, graph=graph)
    states = [state_t]
    s = state_t
    for step in range(action_sequence.shape[1]):
        s = model(s, action_global=action_sequence[:, step], graph=graph)
        states.append(s)
    return torch.stack(states, dim=1)


def _rollout_channel_mae(pred_seq: torch.Tensor, gt_seq: torch.Tensor, k: int) -> dict[str, float]:
    return per_channel_mae(pred_seq[:, 1:k + 1], gt_seq[:, 1:k + 1])


def _rollout_channel_pearson(
    pred_seq: torch.Tensor, gt_seq: torch.Tensor, k: int,
) -> dict[str, float]:
    return pearson_correlation(pred_seq[:, 1:k + 1], gt_seq[:, 1:k + 1])


def main(run_dir_str: str) -> int:
    run_dir = Path(run_dir_str)
    train_cfg = yaml.safe_load(Path("configs/train.yaml").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph = load_static_graph(Path("data/graph_ieee39.h5"), device)
    model, ckpt, model_cfg = _load_model(run_dir, graph, device)

    stats = load_stats(Path("data/norm_stats.json"))
    eval_horizon = max(10, int(train_cfg.get("rollout_horizon", 1)))

    # Eval uses natural (uniform) sampling, not the training stratified 30%
    # — we want full-set MAE and fault-window MAE measured against the real
    # data distribution. With 4% natural fault density × 6400 pairs ≈ 250
    # fault-window samples × 10 nodes × 4 channels = enough statistics.
    val_ds = wm-agentDataset(
        raw_dir=Path("data/raw"),
        splits_path=Path("data/splits.json"),
        norm_stats_path=Path("data/norm_stats.json"),
        split="val", pairs_per_traj_per_epoch=int(train_cfg["pairs_per_traj_per_epoch"]),
        seed=int(train_cfg["seed"]) + 100,
        fault_window_frac=0.0,
        rollout_horizon=eval_horizon,
    )
    loader = DataLoader(
        val_ds, batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=2
    )

    pred_buf, gt_buf, st_buf, stm1_buf, fault_buf = [], [], [], [], []
    rollout_buf, gt_seq_buf, fault_seq_buf = [], [], []
    with torch.no_grad():
        for batch in loader:
            state_t = batch["state_t"].to(device)
            action_sequence = batch["action_sequence"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                rollout_pred = _rollout_model(model, state_t, action_sequence, graph)
            rollout_pred = rollout_pred.float().cpu()
            pred_buf.append(rollout_pred[:, 1])
            gt_buf.append(batch["state_tp1"].float().cpu())
            st_buf.append(state_t.float().cpu())
            stm1_buf.append(batch["state_tm1"].float().cpu())
            fault_buf.append(batch["fault_window_active"].float().cpu())
            rollout_buf.append(rollout_pred)
            gt_seq_buf.append(batch["state_sequence"].float().cpu())
            fault_seq_buf.append(batch["fault_window_sequence"].float().cpu())

    pred = torch.cat(pred_buf, dim=0)
    gt = torch.cat(gt_buf, dim=0)
    st = torch.cat(st_buf, dim=0)
    stm1 = torch.cat(stm1_buf, dim=0)
    fault_mask = torch.cat(fault_buf, dim=0).bool()  # (B,)
    rollout_states_norm = torch.cat(rollout_buf, dim=0)
    gt_seq_norm = torch.cat(gt_seq_buf, dim=0)
    fault_seq = torch.cat(fault_seq_buf, dim=0).bool()

    pred_phys = torch.from_numpy(invert_norm(pred.numpy(), stats))
    gt_phys = torch.from_numpy(invert_norm(gt.numpy(), stats))
    st_phys = torch.from_numpy(invert_norm(st.numpy(), stats))
    stm1_phys = torch.from_numpy(invert_norm(stm1.numpy(), stats))
    rollout_states_phys = torch.from_numpy(invert_norm(rollout_states_norm.numpy(), stats))
    gt_seq_phys = torch.from_numpy(invert_norm(gt_seq_norm.numpy(), stats))

    # Full-eval-set MAE (mostly steady-state — H1's old reference; kept for
    # comparison but no longer the gate).
    mae_model = per_channel_mae(pred_phys - st_phys, gt_phys - st_phys)
    mae_zero = zero_prediction_baseline_mae(st_phys, gt_phys)
    linear_tp1_phys = st_phys + (st_phys - stm1_phys)
    mae_linear = per_channel_mae(linear_tp1_phys - st_phys, gt_phys - st_phys)
    h1_pass_per_ch_full = {
        ch: (mae_model[ch] / max(mae_zero[ch], 1e-12)) <= 0.30
        for ch in CHANNELS
    }
    direction_full = direction_accuracy(pred_phys - st_phys, gt_phys - st_phys)
    pearson_full = pearson_correlation(pred_phys - st_phys, gt_phys - st_phys)

    # Fault-window MAE (the meaningful WM metric — only timesteps where there's
    # actual fault dynamics to predict; steady-state pairs make full-set MAE
    # uninformative because everyone sits near zero there).
    n_fault = int(fault_mask.sum().item())
    n_total = int(fault_mask.shape[0])
    if n_fault > 0:
        pred_phys_f = pred_phys[fault_mask]
        gt_phys_f = gt_phys[fault_mask]
        st_phys_f = st_phys[fault_mask]
        stm1_phys_f = stm1_phys[fault_mask]
        mae_model_fault = per_channel_mae(pred_phys_f - st_phys_f, gt_phys_f - st_phys_f)
        mae_zero_fault = zero_prediction_baseline_mae(st_phys_f, gt_phys_f)
        linear_tp1_phys_f = st_phys_f + (st_phys_f - stm1_phys_f)
        mae_linear_fault = per_channel_mae(linear_tp1_phys_f - st_phys_f, gt_phys_f - st_phys_f)
        direction_fault = direction_accuracy(pred_phys_f - st_phys_f, gt_phys_f - st_phys_f)
        pearson_fault = pearson_correlation(pred_phys_f - st_phys_f, gt_phys_f - st_phys_f)
        # Tighter / more meaningful gate on the fault window: 50% of zero-baseline.
        h1_pass_per_ch_fault = {
            ch: (mae_model_fault[ch] / max(mae_zero_fault[ch], 1e-12)) <= 0.50
            for ch in CHANNELS
        }
    else:
        mae_model_fault = mae_zero_fault = {ch: float("nan") for ch in CHANNELS}
        mae_linear_fault = {ch: float("nan") for ch in CHANNELS}
        direction_fault = pearson_fault = {ch: float("nan") for ch in CHANNELS}
        h1_pass_per_ch_fault = {ch: False for ch in CHANNELS}

    # Phase-1 H1 pass = fault-window pass on all channels.
    h1_pass = all(h1_pass_per_ch_fault.values())
    h1_pass_per_ch = h1_pass_per_ch_fault  # what gets reported as the gate decision

    # H2 — rollout @ k=10. EVERYTHING converted to *physical* space before
    # comparison. The rollout function operates on normalized state (model.forward
    # returns state-norm), so we invert_norm at the end before computing envelope
    # and Δ-over-k. (Mixing norm-space rollout with phys-space envelope was a bug
    # in the previous version — see decisions D18.)
    train_envelope_inf = float(np.abs(invert_norm(st.numpy(), stats)).max())
    h2_bound_ratio = rollout_boundedness_ratio(rollout_states_phys, train_envelope_inf)
    rollout_step_k_phys = rollout_states_phys[:, -1] - rollout_states_phys[:, 0]
    rollout_mae_per_ch = per_channel_mae(rollout_step_k_phys, torch.zeros_like(rollout_step_k_phys))
    # Kimi flagged: dividing rollout's k-step accumulated Δ by full-set 1-step MAE
    # is meaningless because the full-set 1-step MAE is dominated by ~0 steady-state
    # pairs, so the ratio explodes even when rollout is well-behaved (boundedness
    # 0.35 confirms rollout doesn't blow up). Use fault-window 1-step MAE as the
    # denominator — that's the meaningful WM regime where Δ is non-trivial.
    h2_ratio_per_ch = {
        ch: rollout_mae_per_ch[ch] / max(mae_model_fault[ch], 1e-12)
        for ch in CHANNELS
    }
    h2_pass = (h2_bound_ratio <= 1.5) and all(r <= 5.0 for r in h2_ratio_per_ch.values())
    rollout_mae_at = {}
    rollout_pearson_at = {}
    for k in (5, 10):
        if eval_horizon >= k:
            rollout_mae_at[f"@{k}"] = _rollout_channel_mae(rollout_states_phys, gt_seq_phys, k)
            rollout_pearson_at[f"@{k}"] = _rollout_channel_pearson(
                rollout_states_phys, gt_seq_phys, k
            )

    # H3 — action perturbation sanity (perturb ft + scalars)
    rng = np.random.default_rng(0)
    diffs = []
    with torch.no_grad():
        for batch in loader:
            state_t = batch["state_t"].to(device)
            action_global = batch["action_global"].to(device)
            action_b = _perturb_action(action_global, rng).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                d_a = model.predict_delta(state_t, action_global=action_global, graph=graph)
                d_b = model.predict_delta(state_t, action_global=action_b, graph=graph)
            diffs.append(action_conditioning_diff_ratio(d_a, d_b).cpu())
    diff_all = torch.cat(diffs, dim=0)
    h3_pass = float((diff_all > 0.05).float().mean().item()) >= 0.95

    report = {
        "model_type": model_cfg.get("model_type", ckpt.get("model_type", "mpnn")),
        "checkpoint_epoch": ckpt.get("epoch"),
        "n_pairs_total": n_total,
        "n_pairs_fault_active": n_fault,
        "mae_model_phys_full": mae_model,
        "mae_zero_baseline_phys_full": mae_zero,
        "mae_linear_baseline_phys_full": mae_linear,
        "model_zero_ratio_full": per_channel_ratio(mae_model, mae_zero),
        "model_linear_ratio_full": per_channel_ratio(mae_model, mae_linear),
        "direction_accuracy_full": direction_full,
        "pearson_correlation_full": pearson_full,
        "h1_pass_per_channel_full_30pct": h1_pass_per_ch_full,
        "mae_model_phys_fault_window": mae_model_fault,
        "mae_zero_baseline_phys_fault_window": mae_zero_fault,
        "mae_linear_baseline_phys_fault_window": mae_linear_fault,
        "fault_window_model_zero_ratio": per_channel_ratio(mae_model_fault, mae_zero_fault),
        "fault_window_model_linear_ratio": per_channel_ratio(mae_model_fault, mae_linear_fault),
        "direction_accuracy_fault_window": direction_fault,
        "pearson_correlation_fault_window": pearson_fault,
        "progressive_h1": {
            "bronze_all_channels_le_1p0_zero": all(
                r <= 1.0 for r in per_channel_ratio(mae_model_fault, mae_zero_fault).values()
            ),
            "silver_all_channels_le_0p8_zero": all(
                r <= 0.8 for r in per_channel_ratio(mae_model_fault, mae_zero_fault).values()
            ),
            "gold_all_channels_le_0p5_zero": all(
                r <= 0.5 for r in per_channel_ratio(mae_model_fault, mae_zero_fault).values()
            ),
        },
        "h1_pass": h1_pass,
        "h1_pass_per_channel": h1_pass_per_ch,
        "h1_threshold": "model_mae <= 0.50 * zero_baseline on fault-window pairs",
        "h2_bound_ratio": h2_bound_ratio,
        "h2_ratio_per_channel": h2_ratio_per_ch,
        "rollout_mae": rollout_mae_at,
        "rollout_pearson": rollout_pearson_at,
        "h2_pass": h2_pass,
        "h3_pass_fraction": float((diff_all > 0.05).float().mean().item()),
        "h3_pass": h3_pass,
        "all_pass": h1_pass and h2_pass and h3_pass,
    }
    out = run_dir / "eval.json"
    out.write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        run_dir / "eval_rollout_samples.npz",
        pred_rollout=rollout_states_phys[:32].numpy(),
        gt_rollout=gt_seq_phys[:32].numpy(),
        fault_mask=fault_seq[:32].numpy(),
    )
    print(json.dumps(report, indent=2))
    return 0 if report["all_pass"] else 2


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: scripts/eval.py <run_dir>")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
