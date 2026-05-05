"""Training loop for the world-model-distilled gridwm-agent agent."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from wmagent.agent.data import AgentImaginationDataset, CandidateEventSpace
from wmagent.agent.metrics import AgentEvalResult, evaluate_agent_policy
from wmagent.agent.model import WorldModelDistilledRanker


@dataclass(frozen=True)
class AgentTrainConfig:
    epochs: int = 50
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    hidden: int = 128
    dropout: float = 0.05
    actor_loss_weight: float = 0.5
    critic_loss_weight: float = 5.0
    ranking_loss_weight: float = 0.5
    soft_policy_loss_weight: float = 1.0
    safety_loss_weight: float = 1.0
    safety_oversample_low_band: float = 0.0
    critic_ranking_margin_loss_weight: float = 0.5
    grad_clip_norm: float = 5.0
    seed: int = 2026


def set_agent_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_agent_epoch(
    agent: WorldModelDistilledRanker,
    dataset: AgentImaginationDataset,
    event_space: CandidateEventSpace,
    *,
    cfg: AgentTrainConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    actions = event_space.action_sequences.to(device)
    agent.train()
    losses = []
    critic_losses = []
    actor_losses = []
    ce = torch.nn.CrossEntropyLoss()
    smooth = torch.nn.SmoothL1Loss()
    for batch in loader:
        state = batch["state"].to(device)
        context_action = batch["context_action"].to(device)
        risk_raw = batch["risk_raw"].to(device)
        risk_norm = batch["risk_norm"].to(device)
        out = agent(state, actions, context_actions=context_action)
        target_high = risk_raw.argmax(dim=1)
        target_low = risk_raw.argmin(dim=1)
        critic_loss = (
            smooth(out.values, risk_norm)
            + float(cfg.ranking_loss_weight) * _value_ranking_loss(out.values, risk_norm)
            + float(cfg.critic_ranking_margin_loss_weight)
            * _critic_ranking_margin_loss(out.values, risk_norm)
        )
        discover_score = out.discover_logits + out.values.detach()
        safe_score = out.safe_logits - out.values.detach()
        discover_actor_loss = (
            ce(discover_score, target_high)
            + _top_fraction_policy_loss(discover_score, risk_raw, largest=True)
            + float(cfg.soft_policy_loss_weight)
            * _soft_policy_loss(discover_score, risk_norm, largest=True)
        )
        safety_bottom_term = _top_fraction_policy_loss(safe_score, risk_raw, largest=False)
        safety_actor_loss = (
            ce(safe_score, target_low)
            + (1.0 + float(cfg.safety_oversample_low_band)) * safety_bottom_term
            + float(cfg.soft_policy_loss_weight)
            * _soft_policy_loss(safe_score, risk_norm, largest=False)
        )
        actor_loss = discover_actor_loss + float(cfg.safety_loss_weight) * safety_actor_loss
        loss = (
            float(cfg.critic_loss_weight) * critic_loss
            + float(cfg.actor_loss_weight) * actor_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().item()))
        critic_losses.append(float(critic_loss.detach().item()))
        actor_losses.append(float(actor_loss.detach().item()))
    return {
        "loss": float(np.mean(losses)),
        "critic_loss": float(np.mean(critic_losses)),
        "actor_loss": float(np.mean(actor_losses)),
    }


def _top_fraction_policy_loss(
    logits: torch.Tensor,
    risk_raw: torch.Tensor,
    *,
    largest: bool,
    frac: float = 0.10,
) -> torch.Tensor:
    k = max(1, int(np.ceil(risk_raw.shape[1] * frac)))
    top_idx = torch.topk(risk_raw, k=k, dim=1, largest=largest).indices
    positive_logits = logits.gather(1, top_idx)
    return -(torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)).mean()


def _soft_policy_loss(
    logits: torch.Tensor,
    risk_norm: torch.Tensor,
    *,
    largest: bool,
    temperature: float = 0.08,
) -> torch.Tensor:
    target_source = risk_norm if largest else 1.0 - risk_norm
    target = torch.softmax(target_source / max(temperature, 1e-6), dim=1)
    log_prob = torch.log_softmax(logits, dim=1)
    return -(target.detach() * log_prob).sum(dim=1).mean()


def _critic_ranking_margin_loss(
    values: torch.Tensor,
    risk_norm: torch.Tensor,
) -> torch.Tensor:
    """Listwise sorted-pairwise margin loss on critic ranking.

    SmoothL1 alone admits a constant-output local minimum (values ≈
    mean(risk_norm), per-element loss small, but ranking arbitrary). A naive
    Pearson loss vanishes there because var(values) → 0 sends the denominator
    to its clamp floor and the gradient with it. Sorted-pairwise margin via
    softplus has gradient -sigmoid(0) = -0.5 at the constant-output point, so
    it always pulls the critic out of that flat minimum.
    """
    sorted_idx = risk_norm.argsort(dim=1, descending=True)
    sorted_values = torch.gather(values, 1, sorted_idx)
    diff = sorted_values[:, :-1] - sorted_values[:, 1:]
    return torch.nn.functional.softplus(-diff).mean()


def _value_ranking_loss(
    values: torch.Tensor,
    risk_norm: torch.Tensor,
    *,
    frac: float = 0.10,
) -> torch.Tensor:
    k = max(1, int(np.ceil(risk_norm.shape[1] * frac)))
    high_idx = torch.topk(risk_norm, k=k, dim=1, largest=True).indices
    low_idx = torch.topk(risk_norm, k=k, dim=1, largest=False).indices
    high_values = values.gather(1, high_idx)
    low_values = values.gather(1, low_idx)
    high_margin = high_values.unsqueeze(2) - values.unsqueeze(1)
    low_margin = values.unsqueeze(1) - low_values.unsqueeze(2)
    return torch.nn.functional.softplus(-high_margin).mean() + torch.nn.functional.softplus(
        -low_margin
    ).mean()


def evaluate_dataset(
    agent: WorldModelDistilledRanker,
    dataset: AgentImaginationDataset,
    event_space: CandidateEventSpace,
    *,
    exhaustive_rollout_latency_ms: float,
) -> AgentEvalResult:
    return evaluate_agent_policy(
        agent,
        states=dataset.states,
        context_actions=dataset.context_actions,
        action_sequences=event_space.action_sequences,
        risk_raw=dataset.risk_raw,
        risk_norm=dataset.risk_norm,
        metadata=event_space.metadata,
        exhaustive_rollout_latency_ms=exhaustive_rollout_latency_ms,
    )
