import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_json", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    records = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    training = [
        row
        for row in records
        if "loss" in row and "eval_loss" not in row and "step" in row
    ]
    steps = [row["step"] for row in training]
    losses = [row["loss"] for row in training]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 17,
            "axes.labelsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFBFC")

    color = "#176B9C"
    ax.plot(steps, losses, color=color, linewidth=2.4, label="Logged training loss")
    ax.scatter(steps, losses, color=color, s=13, zorder=3)
    ax.axhline(
        0.39504344951710313,
        color="#C45A35",
        linewidth=1.6,
        linestyle="--",
        label="Full-run mean loss (0.3950)",
    )

    ax.set_title("Continued MLM Pretraining: Training Loss", pad=16, weight="bold")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Logged MLM training loss")
    ax.set_xlim(0, 8350)
    ax.set_ylim(0, 4.85)
    ax.grid(True, color="#D9DEE3", linewidth=0.8, alpha=0.8)
    ax.legend(loc="upper right", frameon=True, framealpha=0.96)

    late = [(step, loss) for step, loss in zip(steps, losses) if step >= 1000]
    inset = inset_axes(ax, width="49%", height="43%", loc="center right", borderpad=2.1)
    inset.set_facecolor("white")
    inset.plot(
        [item[0] for item in late],
        [item[1] for item in late],
        color=color,
        linewidth=1.8,
    )
    inset.scatter(
        [item[0] for item in late],
        [item[1] for item in late],
        color=color,
        s=8,
    )
    inset.set_title("Late-stage convergence (step >= 1,000)", fontsize=10, weight="bold")
    inset.set_xlim(950, 8300)
    inset.set_ylim(0.18, 0.50)
    inset.set_xlabel("Step", fontsize=9)
    inset.set_ylabel("Loss", fontsize=9)
    inset.tick_params(labelsize=8)
    inset.grid(True, color="#E2E6EA", linewidth=0.6)

    fig.text(
        0.5,
        0.015,
        "Loss is logged every 100 optimisation steps. The full-run mean includes the high-loss initial adaptation period.",
        ha="center",
        color="#4A4A4A",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
