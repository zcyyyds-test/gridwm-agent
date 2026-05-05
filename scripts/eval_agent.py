"""Evaluate a trained wm-agent Dreamer-style agent.

With ``--candidate-fault-types``, the eval restricts the candidate event
space to a subset of fault types (default keeps all 162 candidates). This
is the OOD slice mode: pass ``--candidate-fault-types 7`` to eval the
agent only on FT-7 candidates while it was trained on the full 162-event
space, which exposes how much of the agent's headline hit rate transfers
to a single fault family.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from wmagent.agent.data import CandidateEventSpace, build_imagination_dataset
from wmagent.agent.metrics import evaluate_agent_policy
from wmagent.agent.model import DreamerStyleAgent
from wmagent.eval.scenario_search import ScenarioGrid
from wmagent.world.power_grid import PowerGridWorldModelSystem


def _parse_fault_type_csv(text: str | None) -> tuple[int, ...] | None:
    if text is None or text.strip() == "":
        return None
    return tuple(int(token.strip()) for token in text.split(",") if token.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_dir", type=Path)
    parser.add_argument("--world-run-dir", type=Path, default=Path("outputs/run_3acec7d14d"))
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--n-anchors", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--rollout-batch-size", type=int, default=64)
    parser.add_argument(
        "--candidate-fault-types",
        type=str,
        default=None,
        help="Comma-separated fault types (e.g. '7' or '1,3') to keep in the candidate event space. "
             "Default keeps the full grid (1,3,7) -> 162 candidates.",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="eval_agent.json",
        help="Filename for the eval JSON inside agent_dir (override for OOD slices).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.agent_dir / "agent.pt", map_location=device)
    cfg = dict(ckpt["agent_cfg"])
    if "discover_prior" not in ckpt["model"] and cfg.get("n_candidates") is not None:
        cfg["n_candidates"] = None
    agent = DreamerStyleAgent(**cfg).to(device)
    agent.load_state_dict(ckpt["model"])
    agent.eval()

    system = PowerGridWorldModelSystem.from_run_dir(args.world_run_dir, device=device)
    fault_types = _parse_fault_type_csv(args.candidate_fault_types)
    grid = replace(ScenarioGrid(), fault_types=fault_types) if fault_types else None
    event_space = CandidateEventSpace.from_grid(horizon=args.horizon, grid=grid, device=device)
    t0 = time.perf_counter()
    dataset = build_imagination_dataset(
        system,
        split=args.split,
        n_anchors=args.n_anchors,
        seed=args.seed,
        horizon=args.horizon,
        event_space=event_space,
        rollout_batch_size=args.rollout_batch_size,
    )
    exhaustive_ms = (time.perf_counter() - t0) * 1000.0 / max(1, args.n_anchors)
    result = evaluate_agent_policy(
        agent,
        states=dataset.states,
        context_actions=dataset.context_actions,
        action_sequences=event_space.action_sequences,
        risk_raw=dataset.risk_raw,
        risk_norm=dataset.risk_norm,
        metadata=event_space.metadata,
        exhaustive_rollout_latency_ms=exhaustive_ms,
    )
    out = {
        "agent_dir": str(args.agent_dir),
        "world_run_dir": str(args.world_run_dir),
        "split": args.split,
        "n_anchors": args.n_anchors,
        "horizon": args.horizon,
        "candidate_fault_types": list(fault_types) if fault_types else None,
        "n_candidates": int(event_space.n_candidates),
        **result.to_dict(),
    }
    out_path = args.agent_dir / args.out_name
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
