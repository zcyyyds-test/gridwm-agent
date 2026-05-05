import torch

from wmagent.eval.scenario_search import (
    ScenarioGrid,
    generate_candidate_action_sequences,
    rank_scenario_rollouts,
)


def test_generate_candidate_action_sequences_shape_and_tau_progression():
    grid = ScenarioGrid(
        fault_types=(1,),
        durations_s=(0.1,),
        resistance_ohm=(0.1,),
        tau0_values=(0.0,),
        fs_values_s=(2.0,),
        output_dt_s=0.001,
    )
    seq, meta = generate_candidate_action_sequences(horizon=4, grid=grid)
    assert seq.shape == (1, 4, 12)
    assert meta[0]["event_code"] == "FT-1"
    assert torch.all(seq[0, :, 1] == 1.0)
    assert torch.all(seq[0, 1:, 11] > seq[0, :-1, 11])


def test_rank_scenario_rollouts_returns_highest_risk_first():
    rollouts = torch.zeros(2, 5, 10, 4)
    rollouts[0, 1:, 0, 0] = 0.1
    rollouts[1, 1:, 0, 0] = 1.0
    rows = rank_scenario_rollouts(
        rollouts,
        [
            {"event_code": "small"},
            {"event_code": "large"},
        ],
        top_k=2,
    )
    assert rows[0]["scenario"]["event_code"] == "large"
    assert rows[0]["risk"]["score"] >= rows[1]["risk"]["score"]
