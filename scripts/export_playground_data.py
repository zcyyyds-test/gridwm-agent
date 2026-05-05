"""Export a compact gridwm-agent world-model playground snapshot.

The static HTML demo cannot run a PyTorch checkpoint in the browser, so this
script exports a small set of real V2 fault-window rollouts into a plain JS
payload. The playground then behaves like an interactive scenario library:
choose a fault event, inspect the counterfactual rollout, and read the risk
signal produced from the imagined future.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from wmagent.data.dataset import WMAgentDataset
from wmagent.data.normalizer import invert_norm, load_stats
from wmagent.data.schema import CHANNELS
from wmagent.models.factory import build_world_model
from wmagent.train.loop import load_static_graph


_FS_MIN, _FS_MAX = 1.0, 5.0
_DUR_MIN, _DUR_MAX = 0.05, 0.20
_CHG_LOG_MIN, _CHG_LOG_MAX = math.log10(0.01), math.log10(10.0)


def _load_model(run_dir: Path, graph, device: torch.device):
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
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


def _decode_action(action_vec: np.ndarray) -> dict[str, float | int | str]:
    ft = int(np.argmax(action_vec[:8]))
    fs = float(action_vec[8] * (_FS_MAX - _FS_MIN) + _FS_MIN)
    duration = float(action_vec[9] * (_DUR_MAX - _DUR_MIN) + _DUR_MIN)
    chg_ohm = float(10 ** (action_vec[10] * (_CHG_LOG_MAX - _CHG_LOG_MIN) + _CHG_LOG_MIN))
    return {
        "fault_type": ft,
        "event_code": f"FT-{ft}",
        "start_s": round(fs, 4),
        "duration_ms": round(duration * 1000.0, 1),
        "resistance_ohm": round(chg_ohm, 5),
        "tau0": round(float(action_vec[11]), 4),
    }


def _load_eval(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def _mean(values: dict[str, float]) -> float:
    return float(np.mean(list(values.values())))


def _metrics(v2_eval: dict[str, Any], baseline_eval: dict[str, Any]) -> dict[str, Any]:
    v2_fault_ratio = v2_eval.get("fault_window_model_zero_ratio", {})
    base_fault_ratio = baseline_eval.get("fault_window_model_zero_ratio", {})
    v2_direction = v2_eval.get("direction_accuracy_fault_window", {})
    base_direction = baseline_eval.get("direction_accuracy_fault_window", {})
    v2_corr = v2_eval.get("pearson_correlation_fault_window", {})
    base_corr = baseline_eval.get("pearson_correlation_fault_window", {})
    v2_rollout = v2_eval.get("rollout_pearson", {}).get("@10", {})
    base_rollout = baseline_eval.get("rollout_pearson", {}).get("@10", {})
    return {
        "v2": {
            "model_type": v2_eval.get("model_type", "latent_graph_wm"),
            "checkpoint_epoch": v2_eval.get("checkpoint_epoch"),
            "mean_fault_zero_ratio": round(_mean(v2_fault_ratio), 4) if v2_fault_ratio else None,
            "mean_direction_accuracy": round(_mean(v2_direction), 4) if v2_direction else None,
            "mean_fault_shape_correlation": round(_mean(v2_corr), 4) if v2_corr else None,
            "mean_rollout_pearson_at_10": round(_mean(v2_rollout), 4) if v2_rollout else None,
            "bounded_rollout": bool(v2_eval.get("h2_pass", False)),
            "action_sensitivity": bool(v2_eval.get("h3_pass", False)),
            "per_channel_fault_zero_ratio": v2_fault_ratio,
        },
        "baseline": {
            "model_type": baseline_eval.get("model_type", "mpnn"),
            "checkpoint_epoch": baseline_eval.get("checkpoint_epoch"),
            "mean_fault_zero_ratio": round(_mean(base_fault_ratio), 4) if base_fault_ratio else None,
            "mean_direction_accuracy": round(_mean(base_direction), 4) if base_direction else None,
            "mean_fault_shape_correlation": round(_mean(base_corr), 4) if base_corr else None,
            "mean_rollout_pearson_at_10": round(_mean(base_rollout), 4) if base_rollout else None,
            "bounded_rollout": bool(baseline_eval.get("h2_pass", False)),
            "action_sensitivity": bool(baseline_eval.get("h3_pass", False)),
            "per_channel_fault_zero_ratio": base_fault_ratio,
        },
    }


def _round_array(x: np.ndarray) -> list:
    return np.round(x.astype("float64"), 6).tolist()


def _build_samples(
    *,
    run_dir: Path,
    n_samples: int,
    horizon: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    graph = load_static_graph(Path("data/graph_ieee39.h5"), device)
    model, _ckpt, _model_cfg = _load_model(run_dir, graph, device)
    stats = load_stats(Path("data/norm_stats.json"))
    train_cfg = yaml.safe_load(Path("configs/train.yaml").read_text())
    dataset = WMAgentDataset(
        raw_dir=Path("data/raw"),
        splits_path=Path("data/splits.json"),
        norm_stats_path=Path("data/norm_stats.json"),
        split="val",
        pairs_per_traj_per_epoch=max(n_samples * 4, int(train_cfg["pairs_per_traj_per_epoch"])),
        seed=seed,
        fault_window_frac=1.0,
        rollout_horizon=horizon,
    )
    loader = DataLoader(dataset, batch_size=min(16, n_samples), shuffle=False, num_workers=0)

    samples: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            state_t = batch["state_t"].to(device)
            action_sequence = batch["action_sequence"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred_norm = _rollout_model(model, state_t, action_sequence, graph)
            pred = invert_norm(pred_norm.float().cpu().numpy(), stats)
            truth = invert_norm(batch["state_sequence"].float().cpu().numpy(), stats)
            actions = batch["action_sequence"].cpu().numpy()
            fault_mask = batch["fault_window_sequence"].cpu().numpy().astype(bool)
            for i in range(pred.shape[0]):
                p = pred[i]
                t = truth[i]
                baseline = np.broadcast_to(t[:1], t.shape)
                pred_delta = p[1:] - p[:1]
                truth_delta = t[1:] - t[:1]
                err_model = float(np.mean(np.abs(p[1:] - t[1:])))
                err_zero = float(np.mean(np.abs(baseline[1:] - t[1:])))
                flat_idx = int(np.argmax(np.abs(pred_delta)))
                _, gen_idx, ch_idx = np.unravel_index(flat_idx, pred_delta.shape)
                samples.append(
                    {
                        "id": f"event-{len(samples) + 1:02d}",
                        "title": f"Fault Event {len(samples) + 1:02d}",
                        "action": _decode_action(actions[i, 0]),
                        "horizon_steps": int(horizon),
                        "channels": list(CHANNELS),
                        "pred_rollout": _round_array(p),
                        "true_rollout": _round_array(t),
                        "fault_window": fault_mask[i].astype(int).tolist(),
                        "raw_risk": {
                            "max_predicted_delta": round(float(np.max(np.abs(pred_delta))), 6),
                            "mean_predicted_delta": round(float(np.mean(np.abs(pred_delta))), 6),
                            "model_vs_no_change_error": round(err_model / max(err_zero, 1e-12), 4),
                            "dominant_generator": int(gen_idx),
                            "dominant_channel": CHANNELS[ch_idx],
                        },
                    }
                )
                if len(samples) >= n_samples:
                    return samples
    return samples


def _attach_risk_scores(samples: list[dict[str, Any]]) -> None:
    values = np.array([s["raw_risk"]["max_predicted_delta"] for s in samples], dtype="float64")
    lo, hi = float(values.min()), float(values.max())
    span = max(hi - lo, 1e-12)
    for sample in samples:
        raw = sample["raw_risk"]["max_predicted_delta"]
        normalized = (raw - lo) / span
        score = int(round(42 + normalized * 55))
        sample["risk"] = {
            "score": score,
            "band": "CRITICAL" if score >= 82 else "ELEVATED" if score >= 64 else "WATCH",
            **sample.pop("raw_risk"),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="V2 run directory, e.g. outputs/demo_run")
    parser.add_argument("--baseline-eval", type=Path, default=None)
    parser.add_argument("--v2-eval", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("docs/world_model_playground_data.js"))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v2_eval = _load_eval(args.v2_eval or args.run_dir / "eval.json")
    baseline_eval = _load_eval(args.baseline_eval)
    samples = _build_samples(
        run_dir=args.run_dir,
        n_samples=args.samples,
        horizon=args.horizon,
        seed=args.seed,
        device=device,
    )
    _attach_risk_scores(samples)
    payload = {
        "meta": {
            "name": "gridwm-agent Playground Snapshot",
            "run_id": args.run_dir.name,
            "snapshot": "fault-window validation scenarios",
            "horizon_steps": args.horizon,
            "device": str(device),
        },
        "channels": list(CHANNELS),
        "metrics": _metrics(v2_eval, baseline_eval),
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "window.GRIDWM_PLAYGROUND_DATA = "
        + json.dumps(payload, indent=2, sort_keys=True)
        + ";\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out} with {len(samples)} scenarios")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
