"""Counterfactual scenario generation and ranking for gridwm-agent."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from wmagent.eval.risk import normalize_risk_scores, rollout_risk_features, rollout_risk_value


_FS_MIN, _FS_MAX = 1.0, 5.0
_DUR_MIN, _DUR_MAX = 0.05, 0.20
_CHG_LOG_MIN = math.log10(0.01)
_CHG_LOG_MAX = math.log10(10.0)


@dataclass(frozen=True)
class ScenarioGrid:
    """Small candidate grid for the demo scenario search."""

    fault_types: tuple[int, ...] = (1, 3, 7)
    durations_s: tuple[float, ...] = (0.06, 0.12, 0.18)
    resistance_ohm: tuple[float, ...] = (0.03, 0.1, 1.0)
    tau0_values: tuple[float, ...] = (-0.15, 0.1, 0.55)
    fs_values_s: tuple[float, ...] = (2.0, 3.5)
    output_dt_s: float = 0.001


def _encode_candidate_action(
    *,
    fault_type: int,
    fs_s: float,
    duration_s: float,
    resistance_ohm: float,
    tau: float,
    device: torch.device,
) -> torch.Tensor:
    out = torch.zeros(12, dtype=torch.float32, device=device)
    out[fault_type] = 1.0
    out[8] = (float(fs_s) - _FS_MIN) / (_FS_MAX - _FS_MIN)
    out[9] = (float(duration_s) - _DUR_MIN) / (_DUR_MAX - _DUR_MIN)
    log_r = math.log10(max(float(resistance_ohm), 1e-12))
    out[10] = (log_r - _CHG_LOG_MIN) / (_CHG_LOG_MAX - _CHG_LOG_MIN)
    out[11] = float(max(-2.0, min(3.0, tau)))
    return out


def generate_candidate_action_sequences(
    *,
    horizon: int,
    grid: ScenarioGrid | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Generate action sequences shaped ``(K,H,12)`` plus human-readable metadata."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    grid = grid or ScenarioGrid()
    device = device or torch.device("cpu")
    sequences = []
    metadata: list[dict[str, Any]] = []
    for ft in grid.fault_types:
        for fs_s in grid.fs_values_s:
            for duration_s in grid.durations_s:
                for resistance in grid.resistance_ohm:
                    for tau0 in grid.tau0_values:
                        tau_step = grid.output_dt_s / max(duration_s, 1e-6)
                        seq = torch.stack(
                            [
                                _encode_candidate_action(
                                    fault_type=ft,
                                    fs_s=fs_s,
                                    duration_s=duration_s,
                                    resistance_ohm=resistance,
                                    tau=tau0 + step * tau_step,
                                    device=device,
                                )
                                for step in range(horizon)
                            ],
                            dim=0,
                        )
                        sequences.append(seq)
                        metadata.append(
                            {
                                "fault_type": ft,
                                "event_code": f"FT-{ft}",
                                "start_s": round(fs_s, 4),
                                "duration_ms": round(duration_s * 1000.0, 1),
                                "resistance_ohm": round(resistance, 6),
                                "tau0": round(tau0, 4),
                            }
                        )
    return torch.stack(sequences, dim=0), metadata


def rank_scenario_rollouts(
    rollouts: torch.Tensor,
    metadata: list[dict[str, Any]],
    *,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Rank candidate rollouts by risk and return the top-k scenario records."""
    if rollouts.shape[0] != len(metadata):
        raise ValueError(
            f"rollout batch size {rollouts.shape[0]} != metadata size {len(metadata)}"
        )
    raw = rollout_risk_value(rollouts)
    scores = normalize_risk_scores(raw)
    features = rollout_risk_features(rollouts, scores=scores)
    order = torch.argsort(raw, descending=True).tolist()
    rows = []
    for rank, idx in enumerate(order[:top_k], start=1):
        rows.append(
            {
                "rank": rank,
                "scenario": metadata[idx],
                "risk": features[idx],
            }
        )
    return rows
