"""Create a README-ready table from wm-agent agent eval files.

With ``--add-ft7-baseline`` the output gains a second table that compares the
agent's mean horizon-10 risk against four reference points already serialized
in the eval JSON: oracle low / random candidates / FT-7-only heuristic /
oracle high. The FT-7 heuristic is the natural strawman -- the portfolio sweep
shows FT-7 dominates top-risk futures, so a sane reviewer will ask whether a
"always pick FT-7" rule already matches the agent. This table answers that.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _resolve_specs(specs: list[str]) -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    for spec in specs:
        if "=" in spec:
            name, path_str = spec.split("=", 1)
        else:
            path = Path(spec)
            name = path.parent.name
            path_str = spec
        report = json.loads(Path(path_str).read_text())
        rows.append((name, report))
    return rows


def _agent_table(rows: list[tuple[str, dict]]) -> list[str]:
    lines = [
        "# wm-agent Agent Benchmark",
        "",
        "| Run | Pearson | Spearman | Discover hit | Safe hit | Risk lift | "
        "Risk reduction | Agent ms | Exhaustive ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, report in rows:
        lines.append(
            "| {name} | {pearson} | {spearman} | {dhit} | {shit} | "
            "{lift} | {reduction} | {agent_ms} | {rollout_ms} |".format(
                name=name,
                pearson=_fmt(report["critic_pearson"]),
                spearman=_fmt(report["critic_spearman"]),
                dhit=_fmt(report["discover_top10pct_hit_rate"]),
                shit=_fmt(report["safe_bottom10pct_hit_rate"]),
                lift=_fmt(report["risk_lift_vs_random"]),
                reduction=_fmt(report["risk_reduction_vs_random"]),
                agent_ms=_fmt(report["agent_latency_ms"]),
                rollout_ms=_fmt(report["exhaustive_rollout_latency_ms"]),
            )
        )
    return lines


def _ft7_table(rows: list[tuple[str, dict]]) -> list[str]:
    """Compare agent discover/safe mean risk to oracle / random / FT-7-only."""
    lines = [
        "",
        "## Risk continuum vs reference points",
        "",
        "Mean horizon-10 risk score on the same 162-candidate event space.",
        "Discover policies should push toward `oracle_high`; safety policies toward `oracle_low`.",
        "The FT-7-only column is the natural rule-based strawman.",
        "",
        "| Run | Oracle low | Agent safe | Random | FT-7 heuristic | "
        "Agent discover | Oracle high | Discover gain over FT-7 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, report in rows:
        ft7 = report.get("ft7_heuristic_mean_risk")
        oracle_high = report.get("oracle_high_mean_risk")
        agent_discover = report.get("agent_discover_mean_risk")
        if ft7 is None or oracle_high is None or agent_discover is None:
            gain = "n/a"
        else:
            denom = oracle_high - ft7
            gain = f"{(agent_discover - ft7) / denom * 100:+.1f}%" if denom > 1e-9 else "n/a"
        lines.append(
            "| {name} | {ol} | {asf} | {rnd} | {ft7} | {ad} | {oh} | {gain} |".format(
                name=name,
                ol=_fmt(report.get("oracle_low_mean_risk")),
                asf=_fmt(report.get("agent_safe_mean_risk")),
                rnd=_fmt(report.get("random_mean_risk")),
                ft7=_fmt(ft7),
                ad=_fmt(agent_discover),
                oh=_fmt(oracle_high),
                gain=gain,
            )
        )
    lines.append("")
    lines.append(
        "*Discover gain over FT-7* = (agent_discover − ft7) / (oracle_high − ft7). "
        "Reads as: \"how much of the FT-7 → oracle gap did the agent close?\""
    )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval", nargs="+", help="NAME=path/to/eval_agent.json or path")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--add-ft7-baseline",
        action="store_true",
        help="Append a second table comparing mean risk vs FT-7-only / random / oracle.",
    )
    args = parser.parse_args()

    rows = _resolve_specs(args.eval)
    lines = _agent_table(rows)
    if args.add_ft7_baseline:
        lines.extend(_ft7_table(rows))

    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
