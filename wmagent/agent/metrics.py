"""Evaluation metrics for gridwm-agent's world-model-distilled agent."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from wmagent.agent.model import WorldModelDistilledRanker


@dataclass(frozen=True)
class AgentEvalResult:
    critic_pearson: float
    critic_spearman: float
    discover_top10pct_hit_rate: float
    safe_bottom10pct_hit_rate: float
    discover_oracle_regret: float
    safe_oracle_regret: float
    risk_lift_vs_random: float
    risk_reduction_vs_random: float
    agent_latency_ms: float
    exhaustive_rollout_latency_ms: float
    agent_discover_mean_risk: float
    agent_safe_mean_risk: float
    random_mean_risk: float
    physics_heuristic_mean_risk: float
    oracle_high_mean_risk: float
    oracle_low_mean_risk: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _safe_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.std(unbiased=False).clamp_min(1e-12) * b.std(unbiased=False).clamp_min(1e-12)
    return float(((a * b).mean() / denom).clamp(-1.0, 1.0).item())


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32, device=values.device)
    return ranks


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    return _safe_corr(_rank(a.flatten()), _rank(b.flatten()))


def _gather_row(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return values.gather(1, indices.view(-1, 1)).squeeze(1)


def _topk_hit(values: torch.Tensor, chosen: torch.Tensor, *, largest: bool) -> float:
    k = max(1, int(np.ceil(values.shape[1] * 0.10)))
    top = torch.topk(values, k=k, dim=1, largest=largest).indices
    hit = (top == chosen.view(-1, 1)).any(dim=1).float()
    return float(hit.mean().item())


def _physics_heuristic_indices(metadata: list[dict[str, Any]], *, safe: bool) -> torch.Tensor:
    """Physics-prior baseline: rank candidates by an unbiased linear combination
    of fault-type code, duration, and 1/resistance. Used as a sanity strawman —
    a strong model should beat it, but it is a real heuristic, not a tautology
    that always picks one fault type.
    """
    scores = []
    for row in metadata:
        fault_score = float(row["fault_type"])
        duration = float(row["duration_ms"]) / 180.0
        resistance = float(row["resistance_ohm"])
        resistance_score = 1.0 / max(resistance, 1e-6)
        scores.append(fault_score + duration + 0.05 * resistance_score)
    tensor = torch.tensor(scores, dtype=torch.float32)
    idx = tensor.argmin() if safe else tensor.argmax()
    return idx


def _relative_change(numerator: torch.Tensor, baseline: torch.Tensor) -> float:
    return float((numerator / baseline.abs().clamp_min(1e-12)).item())


def evaluate_agent_policy(
    agent: WorldModelDistilledRanker,
    *,
    states: torch.Tensor,
    context_actions: torch.Tensor | None = None,
    action_sequences: torch.Tensor,
    risk_raw: torch.Tensor,
    risk_norm: torch.Tensor,
    metadata: list[dict[str, Any]],
    exhaustive_rollout_latency_ms: float,
    n_latency_runs: int = 20,
) -> AgentEvalResult:
    device = next(agent.parameters()).device
    states = states.to(device)
    context_actions = None if context_actions is None else context_actions.to(device)
    actions = action_sequences.to(device)
    risk_raw = risk_raw.to(device)
    risk_norm = risk_norm.to(device)
    agent.eval()
    with torch.no_grad():
        output = _forward_agent(agent, states, actions, context_actions)
    discover_policy = output.discover_logits + output.values
    safe_policy = output.safe_logits - output.values
    discover_idx = discover_policy.argmax(dim=1)
    safe_idx = safe_policy.argmax(dim=1)
    discover_risk = _gather_row(risk_raw, discover_idx)
    safe_risk = _gather_row(risk_raw, safe_idx)
    oracle_high = risk_raw.amax(dim=1)
    oracle_low = risk_raw.amin(dim=1)
    random_mean = risk_raw.mean(dim=1)
    physics_idx = _physics_heuristic_indices(metadata, safe=False).to(device).repeat(risk_raw.shape[0])
    physics_risk = _gather_row(risk_raw, physics_idx)

    with torch.no_grad():
        for _ in range(3):
            _forward_agent(agent, states, actions, context_actions)
        t0 = time.perf_counter()
        for _ in range(max(1, n_latency_runs)):
            _forward_agent(agent, states, actions, context_actions)
        agent_latency_ms = (
            (time.perf_counter() - t0) * 1000.0 / max(1, n_latency_runs) / states.shape[0]
        )

    random_mean_scalar = float(random_mean.mean().item())
    return AgentEvalResult(
        critic_pearson=_safe_corr(output.values.detach().cpu(), risk_norm.detach().cpu()),
        critic_spearman=_spearman(output.values.detach().cpu(), risk_norm.detach().cpu()),
        discover_top10pct_hit_rate=_topk_hit(risk_raw, discover_idx, largest=True),
        safe_bottom10pct_hit_rate=_topk_hit(risk_raw, safe_idx, largest=False),
        discover_oracle_regret=float((oracle_high - discover_risk).mean().item()),
        safe_oracle_regret=float((safe_risk - oracle_low).mean().item()),
        risk_lift_vs_random=_relative_change(
            discover_risk.mean() - random_mean.mean(),
            random_mean.mean(),
        ),
        risk_reduction_vs_random=_relative_change(
            random_mean.mean() - safe_risk.mean(),
            random_mean.mean(),
        ),
        agent_latency_ms=float(agent_latency_ms),
        exhaustive_rollout_latency_ms=float(exhaustive_rollout_latency_ms),
        agent_discover_mean_risk=float(discover_risk.mean().item()),
        agent_safe_mean_risk=float(safe_risk.mean().item()),
        random_mean_risk=random_mean_scalar,
        physics_heuristic_mean_risk=float(physics_risk.mean().item()),
        oracle_high_mean_risk=float(oracle_high.mean().item()),
        oracle_low_mean_risk=float(oracle_low.mean().item()),
    )


def _forward_agent(
    agent: WorldModelDistilledRanker,
    states: torch.Tensor,
    actions: torch.Tensor,
    context_actions: torch.Tensor | None,
):
    if context_actions is None:
        return agent(states, actions)
    return agent(states, actions, context_actions=context_actions)
