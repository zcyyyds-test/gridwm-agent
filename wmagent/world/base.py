"""Generic world-model system abstractions.

The model package answers "can we predict the next state?". This system layer
answers the employment-facing question: "can an application ask the world model
to imagine futures, score them, and search an event space?"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class WorldState:
    """A domain state that can be fed into a world model."""

    tensor: Any
    domain: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldEvent:
    """A controllable action/event sequence for counterfactual rollout."""

    tensor: Any
    domain: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImaginedFuture:
    """A world-model rollout produced from a state and event."""

    rollout: Any
    state: WorldState
    event: WorldEvent
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskSignal:
    """Decision-facing summary extracted from an imagined future."""

    score: int
    band: str
    value: float
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """A ranked future discovered by a world-model search loop."""

    rank: int
    future: ImaginedFuture
    risk: RiskSignal


class WorldModelSystem(Protocol):
    """Application-facing contract for domain world models."""

    domain: str

    def imagine(self, state: WorldState, event: WorldEvent) -> ImaginedFuture:
        """Roll the world forward under a proposed event."""

    def score(self, future: ImaginedFuture) -> RiskSignal:
        """Turn an imagined future into a decision-facing risk signal."""

    def search(
        self,
        state: WorldState,
        events: list[WorldEvent],
        *,
        top_k: int,
    ) -> list[SearchResult]:
        """Rank a candidate event space by imagined future risk."""
