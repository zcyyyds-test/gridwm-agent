"""Dataset builders for world-model-distilled imagined planning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from wmagent.eval.risk import rollout_risk_value
from wmagent.eval.scenario_search import ScenarioGrid, generate_candidate_action_sequences
from wmagent.world.base import WorldEvent
from wmagent.world.power_grid import PowerGridWorldModelSystem


@dataclass(frozen=True)
class CandidateEventSpace:
    """Fixed candidate action space used by the V4 agent benchmark."""

    action_sequences: torch.Tensor
    metadata: list[dict[str, Any]]

    @classmethod
    def from_grid(
        cls,
        *,
        horizon: int,
        grid: ScenarioGrid | None = None,
        device: torch.device | None = None,
    ) -> "CandidateEventSpace":
        actions, metadata = generate_candidate_action_sequences(
            horizon=horizon,
            grid=grid or ScenarioGrid(),
            device=device or torch.device("cpu"),
        )
        return cls(action_sequences=actions, metadata=metadata)

    @property
    def n_candidates(self) -> int:
        return int(self.action_sequences.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.action_sequences.shape[1])


class AgentImaginationDataset(Dataset):
    """Precomputed world-model imagination labels for actor/critic training."""

    def __init__(
        self,
        *,
        states: torch.Tensor,
        context_actions: torch.Tensor | None = None,
        risk_raw: torch.Tensor,
        split: str,
    ) -> None:
        if states.ndim != 3:
            raise ValueError(f"states must have shape (B,N,C); got {tuple(states.shape)}")
        if risk_raw.ndim != 2:
            raise ValueError(f"risk_raw must have shape (B,K); got {tuple(risk_raw.shape)}")
        if states.shape[0] != risk_raw.shape[0]:
            raise ValueError("states and risk labels must have the same anchor count")
        self.states = states.float()
        if context_actions is None:
            context_actions = torch.zeros(states.shape[0], 12, dtype=torch.float32)
        if context_actions.ndim != 2 or context_actions.shape[0] != states.shape[0]:
            raise ValueError(
                "context_actions must have shape (B,A); "
                f"got {tuple(context_actions.shape)}"
            )
        self.context_actions = context_actions.float()
        self.risk_raw = risk_raw.float()
        self.risk_norm = _row_minmax(self.risk_raw)
        self.split = split

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "state": self.states[idx],
            "context_action": self.context_actions[idx],
            "risk_raw": self.risk_raw[idx],
            "risk_norm": self.risk_norm[idx],
        }


def _row_minmax(values: torch.Tensor) -> torch.Tensor:
    lo = values.amin(dim=1, keepdim=True)
    hi = values.amax(dim=1, keepdim=True)
    return (values - lo) / (hi - lo).clamp_min(1e-12)


def build_imagination_dataset(
    system: PowerGridWorldModelSystem,
    *,
    split: str,
    n_anchors: int,
    seed: int,
    horizon: int,
    event_space: CandidateEventSpace,
    rollout_batch_size: int,
) -> AgentImaginationDataset:
    """Use the frozen world model to label candidate futures for each anchor."""
    if event_space.horizon != horizon:
        raise ValueError(f"event_space horizon={event_space.horizon} but requested {horizon}")
    events = [
        WorldEvent(
            tensor=event_space.action_sequences[i].to(system.device),
            domain=system.domain,
            metadata=event_space.metadata[i],
        )
        for i in range(event_space.n_candidates)
    ]
    states = []
    context_actions = []
    risks = []
    for anchor_index in range(n_anchors):
        state = system.anchor_state(
            split=split,
            seed=seed,
            horizon=horizon,
            anchor_index=anchor_index,
        )
        futures = system.imagine_many(state, events, batch_size=rollout_batch_size)
        rollouts = torch.stack([future.rollout for future in futures], dim=0)
        states.append(state.tensor.cpu())
        context_actions.append(
            torch.tensor(state.metadata["action_global"], dtype=torch.float32)
        )
        risks.append(rollout_risk_value(rollouts).cpu())
    return AgentImaginationDataset(
        states=torch.stack(states, dim=0),
        context_actions=torch.stack(context_actions, dim=0),
        risk_raw=torch.stack(risks, dim=0),
        split=split,
    )
