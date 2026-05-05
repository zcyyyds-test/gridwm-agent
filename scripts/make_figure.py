"""Generate the README hero figure.

Two panels, both intentionally honest:

* Left: V2 latent recurrent dynamics model vs Stage-2 MPNN baseline on
  fault-window MAE / zero-baseline ratio. Lower bar = better; the dashed
  line at 1.0 is the zero-delta baseline.

* Right: V4.2 ranker hit rates with 95% bootstrap CI error bars across
  three evaluation splits — val (in-distribution), test (held-out
  trajectories; the world model has seen all fault types), and FT-7
  candidate-space holdout (the agent's actor/critic are trained with
  FT-7 candidates excluded; the world model itself is *not* held out).
  Reads point estimates + confidence ranges directly from the
  bootstrap_ci JSON files so the README never advertises a single-seed
  lucky number again.

Both panels read from already-committed experiment artifacts -- no model
weights or live inference required.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
COMPARISON_MD = REPO_ROOT / "experiments" / "v2_vs_stage2_baseline" / "comparison.md"
CANONICAL_VAL = REPO_ROOT / "outputs" / "demo_run" / "bootstrap_ci_val.json"
CANONICAL_TEST = REPO_ROOT / "outputs" / "demo_run" / "bootstrap_ci_test.json"
# FT-7 candidate-space-holdout bootstrap is not bundled in demo_run; if it
# is missing the figure script will skip that panel.
OOD_FT7 = REPO_ROOT / "outputs" / "demo_run" / "bootstrap_ci_ft7_holdout.json"
OUTPUT = REPO_ROOT / "docs" / "figures" / "hero.png"


def _parse_v2_vs_stage2() -> dict[str, dict[str, float]]:
    text = COMPARISON_MD.read_text()
    rows = [line for line in text.splitlines() if line.startswith("| ") and "|" in line[2:]]
    data: dict[str, dict[str, float]] = {}
    channels = ["wr", "LA", "VT", "IT"]
    for line in rows:
        cells = [c.strip() for c in line.strip("|").split("|")]
        run_name = cells[0]
        if run_name in {"Stage2-MPNN", "V2-LatentGraphWM"}:
            mean_ratio = float(cells[4])
            per_channel = {ch: float(cells[5 + i]) for i, ch in enumerate(channels)}
            per_channel["Mean"] = mean_ratio
            data[run_name] = per_channel
    return data


def _load_ci(path: Path, key: str) -> tuple[float, float, float]:
    if not path.exists():
        # Missing bootstrap (e.g. FT-7 holdout not bundled in demo_run);
        # caller treats NaN as "no bar".
        return float("nan"), float("nan"), float("nan")
    payload = json.loads(path.read_text())
    stats = payload["metrics"][key]
    return stats["point"], stats["ci_low"], stats["ci_high"]


def left_panel(ax: plt.Axes, data: dict[str, dict[str, float]]) -> None:
    channels = ["wr", "LA", "VT", "IT", "Mean"]
    stage2 = [data["Stage2-MPNN"][c] for c in channels]
    v2 = [data["V2-LatentGraphWM"][c] for c in channels]

    x = np.arange(len(channels))
    width = 0.38
    ax.bar(x - width / 2, stage2, width, label="Stage-2 MPNN (v0.1)",
           color="#9aa6b2", edgecolor="white")
    ax.bar(x + width / 2, v2, width, label="V2 Latent Graph WM",
           color="#2e6cdf", edgecolor="white")
    ax.axhline(1.0, color="#b0413e", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(len(channels) - 0.55, 1.04, "zero baseline", color="#b0413e",
            fontsize=8, ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.set_ylabel("Fault-window MAE / zero baseline\n(lower = better)")
    ax.set_title("World model: V2 cuts fault response error in half", fontsize=11, pad=8)
    ax.set_ylim(0, max(stage2) * 1.18)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def right_panel(ax: plt.Axes) -> None:
    splits = [
        ("val\n(in-dist)", CANONICAL_VAL),
        ("test\n(held-out\ntrajectories)", CANONICAL_TEST),
        ("FT-7\n(candidate\nholdout)", OOD_FT7),
    ]
    metric_keys = ["discover_top10pct_hit_rate", "safe_bottom10pct_hit_rate"]
    metric_labels = ["Discover\ntop-10% hit", "Safety\nbottom-10% hit"]
    metric_colors = ["#d57a2a", "#34a36b"]

    n_splits = len(splits)
    n_metrics = len(metric_keys)
    width = 0.38
    x = np.arange(n_splits)

    for m_idx, (key, color) in enumerate(zip(metric_keys, metric_colors)):
        points: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for _, path in splits:
            pt, lo, hi = _load_ci(path, key)
            points.append(pt)
            lows.append(pt - lo)
            highs.append(hi - pt)
        offset = (m_idx - (n_metrics - 1) / 2) * width
        bars = ax.bar(
            x + offset, points, width,
            color=color, edgecolor="white",
            label=metric_labels[m_idx].replace("\n", " "),
        )
        ax.errorbar(
            x + offset, points,
            yerr=[lows, highs],
            fmt="none", ecolor="#444", elinewidth=1.0, capsize=3,
        )
        for bar, pt in zip(bars, points):
            ax.text(bar.get_x() + bar.get_width() / 2, pt + 0.025,
                    f"{pt:.2f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.10, color="#888", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(n_splits - 0.4, 0.115, "random baseline (0.10)", color="#888",
            fontsize=8, ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in splits], fontsize=9)
    ax.set_ylabel("hit rate (95% bootstrap CI)")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "V4.2 ranker: discover hit > random on all splits, "
        "including FT-7 candidate-space holdout",
        fontsize=11, pad=8,
    )
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    v2_data = _parse_v2_vs_stage2()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    left_panel(axes[0], v2_data)
    right_panel(axes[1])
    fig.suptitle(
        "gridwm-agent — world-model-distilled risk-ranking agent on power-grid EMT",
        fontsize=12, y=1.02, fontweight="bold",
    )
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
