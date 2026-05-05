"""Rank counterfactual fault scenarios with the wm-agent V3 system API."""
from __future__ import annotations

import argparse
from pathlib import Path

from wmagent.world.power_grid import PowerGridWorldModelSystem, search_result_to_record, write_json


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# wm-agent Scenario Search",
        "",
        f"- Run: `{payload['meta']['run_id']}`",
        f"- Anchor split: `{payload['meta']['anchor_split']}`",
        f"- Candidates: {payload['meta']['n_candidates']}",
        f"- Horizon: {payload['meta']['horizon_steps']} steps",
        "",
        "| Rank | Risk | Band | Fault | Duration ms | Resistance ohm | Dominant | Max delta |",
        "|---:|---:|---|---|---:|---:|---|---:|",
    ]
    for row in payload["top_scenarios"]:
        s = row["scenario"]
        r = row["risk"]
        dominant = f"G{r['dominant_node']} / {r['dominant_channel']} / step {r['dominant_step']}"
        lines.append(
            "| {rank} | {score} | {band} | {event_code} | {duration_ms:.1f} | "
            "{resistance_ohm:g} | {dominant} | {max_abs_delta:.6g} |".format(
                rank=row["rank"],
                score=r["score"],
                band=r["band"],
                event_code=s["event_code"],
                duration_ms=s["duration_ms"],
                resistance_ohm=s["resistance_ohm"],
                dominant=dominant,
                max_abs_delta=r["max_abs_delta"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        type=Path,
        help="trained run directory, e.g. outputs/run_3acec7d14d",
    )
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--anchor-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/scenario_search/top_scenarios.json"),
    )
    args = parser.parse_args()

    system = PowerGridWorldModelSystem.from_run_dir(args.run_dir)
    state = system.anchor_state(
        split=args.split,
        seed=args.seed,
        horizon=args.horizon,
        anchor_index=args.anchor_index,
    )
    events = system.candidate_events(horizon=args.horizon)
    results = system.search(state, events, top_k=args.top_k, batch_size=args.batch_size)
    event_to_idx = {id(event): idx for idx, event in enumerate(events)}
    payload = {
        "meta": {
            "run_id": system.run_id,
            "model_type": system.model_type,
            "checkpoint_epoch": system.checkpoint_epoch,
            "anchor_split": args.split,
            "anchor_index": args.anchor_index,
            "seed": args.seed,
            "horizon_steps": args.horizon,
            "n_candidates": len(events),
            "device": str(system.device),
        },
        "top_scenarios": [
            search_result_to_record(
                result,
                candidate_index=event_to_idx[id(result.future.event)],
            )
            for result in results
        ],
    }
    write_json(args.out, payload)
    _write_markdown(args.out.with_suffix(".md"), payload)
    print(f"ranked {len(events)} scenarios; wrote {args.out} and {args.out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
