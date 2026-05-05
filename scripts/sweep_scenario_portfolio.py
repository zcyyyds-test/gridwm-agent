"""Portfolio-level scenario search with the wm-agent V3 system API."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from wmagent.eval.risk import normalize_risk_scores, rollout_risk_features, rollout_risk_value
from wmagent.world.power_grid import (
    PowerGridWorldModelSystem,
    aggregate_risk_records,
    write_json,
)


def _write_markdown(path: Path, payload: dict) -> None:
    meta = payload["meta"]
    lines = [
        "# wm-agent Portfolio Scenario Sweep",
        "",
        f"- Run: `{meta['run_id']}`",
        f"- Anchors: {meta['n_anchors']}",
        f"- Candidate actions per anchor: {meta['n_candidates_per_anchor']}",
        f"- Total imagined futures: {meta['n_total_futures']}",
        f"- Horizon: {meta['horizon_steps']} steps",
        "",
        "## Top Risky Futures",
        "",
        "| Rank | Anchor | Risk | Band | Fault | Duration ms | Resistance ohm | Dominant |",
        "|---:|---:|---:|---|---|---:|---:|---|",
    ]
    for row in payload["top_futures"]:
        s = row["scenario"]
        r = row["risk"]
        dominant = f"G{r['dominant_node']} / {r['dominant_channel']} / step {r['dominant_step']}"
        lines.append(
            "| {rank} | {anchor} | {score} | {band} | {event_code} | {duration:.1f} | "
            "{resistance:g} | {dominant} |".format(
                rank=row["rank"],
                anchor=row["anchor_index"],
                score=r["score"],
                band=r["band"],
                event_code=s["event_code"],
                duration=s["duration_ms"],
                resistance=s["resistance_ohm"],
                dominant=dominant,
            )
        )

    lines.extend(["", "## Fault Family Risk", ""])
    lines.extend(["| Fault | Count | Mean score | Max score |", "|---|---:|---:|---:|"])
    for row in payload["aggregate"]["by_fault"]:
        lines.append(
            f"| {row['key']} | {row['count']} | {row['mean_score']:.2f} | {row['max_score']} |"
        )

    lines.extend(["", "## Dominant Response Channels", ""])
    lines.extend(["| Channel | Count | Mean score | Max score |", "|---|---:|---:|---:|"])
    for row in payload["aggregate"]["by_dominant_channel"]:
        lines.append(
            f"| {row['key']} | {row['count']} | {row['mean_score']:.2f} | {row['max_score']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _portfolio_records(
    system: PowerGridWorldModelSystem,
    *,
    split: str,
    seed: int,
    horizon: int,
    n_anchors: int,
    batch_size: int,
) -> list[dict]:
    events = system.candidate_events(horizon=horizon)
    pending = []
    raw_values = []
    for anchor_index in range(n_anchors):
        state = system.anchor_state(
            split=split,
            seed=seed,
            horizon=horizon,
            anchor_index=anchor_index,
        )
        futures = system.imagine_many(state, events, batch_size=batch_size)
        rollouts = torch.stack([future.rollout for future in futures], dim=0)
        raw = rollout_risk_value(rollouts)
        raw_values.append(raw)
        for idx, future in enumerate(futures):
            pending.append(
                {
                    "anchor_index": anchor_index,
                    "candidate_index": idx,
                    "scenario": future.event.metadata,
                    "future": future,
                    "risk_value": float(raw[idx].item()),
                }
            )

    all_raw = torch.cat(raw_values, dim=0)
    scores = normalize_risk_scores(all_raw)
    records = []
    for idx, row in enumerate(pending):
        feature = rollout_risk_features(
            row["future"].rollout.unsqueeze(0),
            scores=scores[idx: idx + 1],
        )[0]
        records.append(
            {
                "anchor_index": row["anchor_index"],
                "candidate_index": row["candidate_index"],
                "scenario": row["scenario"],
                "risk": {
                    **feature,
                    "risk_value": row["risk_value"],
                },
            }
        )
    records.sort(key=lambda r: r["risk"]["risk_value"], reverse=True)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--n-anchors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/scenario_portfolio/portfolio_sweep.json"),
    )
    args = parser.parse_args()

    system = PowerGridWorldModelSystem.from_run_dir(args.run_dir)
    records = _portfolio_records(
        system,
        split=args.split,
        seed=args.seed,
        horizon=args.horizon,
        n_anchors=args.n_anchors,
        batch_size=args.batch_size,
    )
    top = []
    for rank, row in enumerate(records[: args.top_k], start=1):
        top.append({"rank": rank, **row})
    payload = {
        "meta": {
            "run_id": system.run_id,
            "model_type": system.model_type,
            "checkpoint_epoch": system.checkpoint_epoch,
            "split": args.split,
            "n_anchors": args.n_anchors,
            "seed": args.seed,
            "horizon_steps": args.horizon,
            "n_candidates_per_anchor": len(system.candidate_events(horizon=args.horizon)),
            "n_total_futures": len(records),
            "device": str(system.device),
        },
        "top_futures": top,
        "aggregate": aggregate_risk_records(records),
    }
    write_json(args.out, payload)
    _write_markdown(args.out.with_suffix(".md"), payload)
    print(
        f"swept {payload['meta']['n_total_futures']} futures "
        f"across {args.n_anchors} anchors; wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
