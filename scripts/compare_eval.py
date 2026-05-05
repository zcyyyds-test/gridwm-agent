"""Create a compact Markdown comparison from multiple wm-agent eval.json files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from wmagent.data.schema import CHANNELS


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return f"{v:.4g}"
    if v is None:
        return "-"
    return str(v)


def _mean_channel(values: dict[str, float] | None) -> float | None:
    if not values:
        return None
    present = [float(values[ch]) for ch in CHANNELS if ch in values]
    if not present:
        return None
    return sum(present) / len(present)


def _channel_cells(values: dict[str, float] | None) -> str:
    values = values or {}
    return " | ".join(_fmt(values.get(ch)) for ch in CHANNELS)


def _load_eval(spec: str) -> tuple[str, dict[str, Any], Path]:
    if "=" in spec:
        name, path_str = spec.split("=", 1)
    else:
        path = Path(spec)
        name = path.parent.name
        path_str = spec
    path = Path(path_str)
    return name, json.loads(path.read_text()), path


def _row(name: str, report: dict[str, Any], path: Path) -> str:
    fw_ratio = report.get("fault_window_model_zero_ratio")
    rollout_p10 = report.get("rollout_pearson", {}).get("@10")
    direction = report.get("direction_accuracy_fault_window")
    pearson = report.get("pearson_correlation_fault_window")
    return (
        f"| {name} | `{report.get('model_type', 'mpnn')}` | "
        f"{_fmt(report.get('checkpoint_epoch'))} | "
        f"{_fmt(report.get('n_pairs_fault_active'))} | "
        f"{_fmt(_mean_channel(fw_ratio))} | {_channel_cells(fw_ratio)} | "
        f"{_fmt(_mean_channel(direction))} | {_fmt(_mean_channel(pearson))} | "
        f"{_fmt(_mean_channel(rollout_p10))} | "
        f"{_fmt(report.get('h2_pass'))} | {_fmt(report.get('h3_pass'))} | "
        f"`{path.parent.name}` |"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval", nargs="+", help="NAME=path/to/eval.json or path/to/eval.json")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    loaded = [_load_eval(spec) for spec in args.eval]
    lines = [
        "# wm-agent Eval Comparison",
        "",
        "| Run | Model | Ckpt epoch | Fault pairs | Mean fault/zero | wr | LA | VT | IT | Mean direction | Mean fault Pearson | Mean rollout Pearson@10 | H2 | H3 | Source |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for name, report, path in loaded:
        lines.append(_row(name, report, path))
    lines.extend(
        [
            "",
            "Notes:",
            "- Fault/zero below 1.0 means the model beats the zero-delta baseline on fault-window pairs.",
            "- H2 is rollout boundedness; H3 is action perturbation sensitivity.",
            "- Linear extrapolation is intentionally not the main packaging baseline because short-step EMT trajectories are very smooth.",
            "",
        ]
    )
    text = "\n".join(lines)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
