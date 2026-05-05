import torch

from wmagent.agent.data import CandidateEventSpace
from wmagent.agent.metrics import evaluate_agent_policy
from wmagent.agent.model import AgentOutput, WorldModelDistilledRanker


def test_candidate_event_space_shape():
    space = CandidateEventSpace.from_grid(horizon=4)
    assert space.action_sequences.shape == (162, 4, 12)
    assert len(space.metadata) == 162
    assert space.metadata[0]["event_code"].startswith("FT-")


def test_distilled_ranker_forward_and_select_shape():
    agent = WorldModelDistilledRanker(in_channels=4, action_dim=12, horizon=5, hidden=32)
    state = torch.randn(3, 10, 4)
    actions = torch.randn(7, 5, 12)
    out = agent(state, actions)
    assert out.discover_logits.shape == (3, 7)
    assert out.safe_logits.shape == (3, 7)
    assert out.values.shape == (3, 7)
    assert agent.select(state, actions, mode="discover").shape == (3,)
    assert agent.select(state, actions, mode="safe").shape == (3,)

    contextual_agent = WorldModelDistilledRanker(
        in_channels=4,
        action_dim=12,
        horizon=5,
        n_candidates=7,
        n_nodes=10,
        hidden=32,
    )
    context = torch.randn(3, 12)
    contextual_out = contextual_agent(state, actions, context_actions=context)
    assert contextual_out.discover_logits.shape == (3, 7)
    assert contextual_out.safe_logits.shape == (3, 7)
    assert contextual_out.values.shape == (3, 7)


class FixedAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(()))

    def forward(self, state, action_sequences):
        bsz = state.shape[0]
        logits = torch.tensor([[0.0, 1.0, 5.0, 2.0, -1.0]], device=state.device)
        safe = torch.tensor([[-1.0, 1.0, 0.0, 2.0, 5.0]], device=state.device)
        values = torch.tensor([[0.2, 0.3, 1.0, 0.4, 0.0]], device=state.device)
        return AgentOutput(
            discover_logits=logits.expand(bsz, -1),
            safe_logits=safe.expand(bsz, -1),
            values=values.expand(bsz, -1),
        )


def test_agent_metrics_oracle_hits_for_fixed_policy():
    risk = torch.tensor([[0.2, 0.3, 1.0, 0.4, 0.0], [0.1, 0.2, 0.9, 0.5, 0.0]])
    metadata = [
        {"fault_type": 1, "duration_ms": 60.0, "resistance_ohm": 1.0},
        {"fault_type": 3, "duration_ms": 60.0, "resistance_ohm": 1.0},
        {"fault_type": 7, "duration_ms": 180.0, "resistance_ohm": 0.03},
        {"fault_type": 3, "duration_ms": 120.0, "resistance_ohm": 0.1},
        {"fault_type": 1, "duration_ms": 60.0, "resistance_ohm": 1.0},
    ]
    result = evaluate_agent_policy(
        FixedAgent(),
        states=torch.zeros(2, 10, 4),
        action_sequences=torch.zeros(5, 3, 12),
        risk_raw=risk,
        risk_norm=risk,
        metadata=metadata,
        exhaustive_rollout_latency_ms=10.0,
        n_latency_runs=1,
    )
    assert result.discover_top10pct_hit_rate == 1.0
    assert result.safe_bottom10pct_hit_rate == 1.0
    assert result.discover_oracle_regret == 0.0
    assert result.safe_oracle_regret == 0.0
