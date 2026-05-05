"""Create a compact Markdown report from an eval.json file."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from wmagent.data.schema import CHANNELS


def _fmt(v: float) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _row(name: str, values: dict[str, float]) -> str:
    cells = " | ".join(_fmt(values.get(ch, float("nan"))) for ch in CHANNELS)
    return f"| {name} | {cells} |"


def main(eval_path_str: str) -> int:
    eval_path = Path(eval_path_str)
    report = json.loads(eval_path.read_text())
    out_path = eval_path.with_suffix(".md")

    lines = [
        f"# wm-agent Eval Report: `{eval_path.parent.name}`",
        "",
        f"- Model: `{report.get('model_type', 'unknown')}`",
        (
            f"- Fault-window pairs: {report.get('n_pairs_fault_active')} / "
            f"{report.get('n_pairs_total')}"
        ),
        f"- H2 boundedness ratio: {_fmt(report.get('h2_bound_ratio', float('nan')))}",
        (
            "- H3 action sensitivity pass fraction: "
            f"{_fmt(report.get('h3_pass_fraction', float('nan')))}"
        ),
        "",
        "## Fault-Window Ratios",
        "",
        "| Metric | wr | LA | VT | IT |",
        "|---|---:|---:|---:|---:|",
        _row("model / zero", report["fault_window_model_zero_ratio"]),
        _row("model / linear", report["fault_window_model_linear_ratio"]),
        "",
        "## Direction And Correlation",
        "",
        "| Metric | wr | LA | VT | IT |",
        "|---|---:|---:|---:|---:|",
        _row("direction accuracy", report["direction_accuracy_fault_window"]),
        _row("pearson", report["pearson_correlation_fault_window"]),
        "",
        "## Rollout",
        "",
        "| Horizon | wr | LA | VT | IT |",
        "|---|---:|---:|---:|---:|",
    ]
    for horizon, values in report.get("rollout_mae", {}).items():
        lines.append(_row(f"MAE {horizon}", values))
    for horizon, values in report.get("rollout_pearson", {}).items():
        lines.append(_row(f"Pearson {horizon}", values))
    lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: scripts/make_report.py <outputs/run_x/eval.json>")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
