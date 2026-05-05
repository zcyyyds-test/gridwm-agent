"""Generate README-ready figures from eval outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wmagent.data.schema import CHANNELS


def main(run_dir_str: str) -> int:
    run_dir = Path(run_dir_str)
    report = json.loads((run_dir / "eval.json").read_text())
    samples = np.load(run_dir / "eval_rollout_samples.npz")
    fig_dir = Path("docs/figures") / run_dir.name
    fig_dir.mkdir(parents=True, exist_ok=True)

    pred = samples["pred_rollout"][0]  # (H+1,N,C)
    gt = samples["gt_rollout"][0]
    t = np.arange(pred.shape[0])
    gen = 0
    for ch_idx, ch in enumerate(CHANNELS):
        plt.figure(figsize=(7, 3.5))
        plt.plot(t, gt[:, gen, ch_idx], label="CloudPSS", linewidth=2)
        plt.plot(t, pred[:, gen, ch_idx], label="gridwm-agent-V2", linewidth=2)
        plt.plot(t, np.full_like(t, gt[0, gen, ch_idx]), label="zero baseline", linestyle="--")
        plt.title(f"{ch} rollout, generator {gen}")
        plt.xlabel("rollout step")
        plt.ylabel(ch)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"rollout_{ch}.png", dpi=160)
        plt.close()

    ratios = report["fault_window_model_zero_ratio"]
    plt.figure(figsize=(6, 3.5))
    xs = np.arange(len(CHANNELS))
    plt.bar(xs, [ratios[ch] for ch in CHANNELS])
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.xticks(xs, CHANNELS)
    plt.ylabel("model / zero MAE")
    plt.title("Fault-window baseline ratio")
    plt.tight_layout()
    plt.savefig(fig_dir / "fault_window_ratio.png", dpi=160)
    plt.close()

    pearson = report.get("rollout_pearson", {})
    if pearson:
        plt.figure(figsize=(6, 3.5))
        for ch in CHANNELS:
            horizons = sorted(pearson, key=lambda x: int(x.removeprefix("@")))
            plt.plot(
                [int(h.removeprefix("@")) for h in horizons],
                [pearson[h][ch] for h in horizons],
                marker="o",
                label=ch,
            )
        plt.xlabel("rollout horizon")
        plt.ylabel("Pearson r")
        plt.title("Rollout waveform correlation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "rollout_pearson.png", dpi=160)
        plt.close()

    print(f"wrote figures under {fig_dir}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: scripts/plot_eval.py <outputs/run_x>")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
