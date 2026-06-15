import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_PATTERN = re.compile(
    r"(?P<dataset>mnist|fmnist|cifar)_cnn_50_C\[1.0\]_iid\[1\]"
    r"_E\[(?P<local_ep>\d+)\]_B\[(?P<local_bs>\d+)\]"
    r"_Gama\[(?P<gamma>[^\]]+)\]_numSteal\[(?P<num_steal>\d+)\]"
    r"(?:_AttackPos\[spread\])?_(?P<metric>acc|loss|mape)\.npy$"
)

EXPERIMENTS = [
    ("mnist", "MNIST / CNN"),
    ("fmnist", "Fashion-MNIST / CNN"),
    ("cifar", "CIFAR-10 / CNN"),
]

NUM_STEAL_VALUES = [1, 2, 3, 4, 5, 10]

NUM_COLORS = {
    1: "#4C78A8",
    2: "#F58518",
    3: "#54A24B",
    4: "#E45756",
    5: "#B279A2",
    10: "#72B7B2",
}

NUM_MARKERS = {
    1: "o",
    2: "s",
    3: "^",
    4: "D",
    5: "P",
    10: "X",
}


def configure_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def load_results(results_dir: Path):
    data = {}
    for path in sorted(results_dir.glob("*.npy")):
        match = RESULT_PATTERN.match(path.name)
        if not match:
            continue

        dataset = match.group("dataset")
        gamma = float(match.group("gamma"))
        num_steal = int(match.group("num_steal"))
        metric = match.group("metric")

        if gamma not in (0.0, 0.5):
            continue

        data.setdefault(dataset, {}).setdefault(gamma, {}).setdefault(num_steal, {})[metric] = np.load(path)

    return data


def require_metric(data, dataset, gamma, num_steal, metric):
    try:
        return data[dataset][gamma][num_steal][metric]
    except KeyError as exc:
        raise FileNotFoundError(
            f"Missing {metric} data for dataset={dataset}, gamma={gamma}, num_steal={num_steal}"
        ) from exc


def plot_num_steal_effect(data, output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 2.85))
    fig.subplots_adjust(left=0.060, right=0.960, bottom=0.20, top=0.78, wspace=0.88)

    legend_handles = []
    legend_labels = []

    for ax, (dataset, title) in zip(axes, EXPERIMENTS):
        x = np.array(NUM_STEAL_VALUES)
        acc = np.array([
            require_metric(data, dataset, 0.5, num_steal, "acc")[-1] * 100.0
            for num_steal in NUM_STEAL_VALUES
        ])
        mape = np.array([
            require_metric(data, dataset, 0.5, num_steal, "mape")[-1]
            for num_steal in NUM_STEAL_VALUES
        ])
        baseline_acc = require_metric(data, dataset, 0.0, 5, "acc")[-1] * 100.0

        acc_line, = ax.plot(
            x,
            acc,
            color="#2F5597",
            marker="o",
            markersize=4.0,
            label="Final accuracy",
            zorder=3,
        )
        baseline_line = ax.axhline(
            baseline_acc,
            color="#555555",
            linestyle="--",
            linewidth=1.0,
            alpha=0.85,
            label="No-attack accuracy",
        )

        ax2 = ax.twinx()
        mape_line, = ax2.plot(
            x,
            mape,
            color="#C44E52",
            marker="s",
            markersize=4.0,
            label="Final MAPE",
            zorder=3,
        )

        if not legend_handles:
            legend_handles = [acc_line, mape_line, baseline_line]
            legend_labels = ["Final accuracy", "Final MAPE", "No-attack accuracy"]

        ax.set_title(title)
        ax.set_xlabel("Number of attacked clients")
        ax.set_ylabel("Accuracy (%)", color="#2F5597")
        ax2.set_ylabel("MAPE", color="#C44E52", labelpad=6)
        ax.tick_params(axis="y", labelcolor="#2F5597")
        ax2.tick_params(axis="y", labelcolor="#C44E52")
        ax.set_xticks(NUM_STEAL_VALUES)
        ax.grid(True, axis="y", alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        acc_min = min(acc.min(), baseline_acc)
        acc_max = max(acc.max(), baseline_acc)
        acc_pad = max((acc_max - acc_min) * 0.22, 0.35)
        ax.set_ylim(acc_min - acc_pad, acc_max + acc_pad)

        mape_min = mape.min()
        mape_max = mape.max()
        mape_pad = max((mape_max - mape_min) * 0.20, 0.015)
        ax2.set_ylim(max(0.0, mape_min - mape_pad), mape_max + mape_pad)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.975),
        borderaxespad=0.0,
    )
    fig.savefig(output_dir / "paper_num_steal_effect.png", dpi=400)
    plt.close(fig)


def plot_num_steal_tradeoff(data, output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.65))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.20, top=0.78, wspace=0.36)

    legend_handles = []
    legend_labels = []

    for ax, (dataset, title) in zip(axes, EXPERIMENTS):
        x_values = []
        y_values = []

        for num_steal in NUM_STEAL_VALUES:
            mape = require_metric(data, dataset, 0.5, num_steal, "mape")[-1]
            acc = require_metric(data, dataset, 0.5, num_steal, "acc")[-1] * 100.0
            x_values.append(mape)
            y_values.append(acc)

            handle = ax.scatter(
                mape,
                acc,
                s=46,
                marker=NUM_MARKERS[num_steal],
                color=NUM_COLORS[num_steal],
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
            ax.annotate(
                str(num_steal),
                xy=(mape, acc),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=NUM_COLORS[num_steal],
            )

            if num_steal not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(num_steal)

        ax.set_title(title)
        ax.set_xlabel("Final MAPE")
        ax.set_ylabel("Final accuracy (%)")
        ax.grid(True, alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        x_pad = max((x_max - x_min) * 0.18, 0.01)
        y_pad = max((y_max - y_min) * 0.22, 0.35)
        ax.set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    labels = [rf"$n_s={num_steal}$" for num_steal in legend_labels]
    fig.legend(
        legend_handles,
        labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.975),
        borderaxespad=0.0,
    )
    fig.savefig(output_dir / "paper_num_steal_tradeoff.png", dpi=400)
    plt.close(fig)


def main():
    results_dir = Path("save/results")
    output_dir = Path("save/plots/paper")
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    data = load_results(results_dir)
    plot_num_steal_effect(data, output_dir)
    plot_num_steal_tradeoff(data, output_dir)

    print(f"Saved paper figures to {output_dir}")
    print(f"- {output_dir / 'paper_num_steal_effect.png'}")
    print(f"- {output_dir / 'paper_num_steal_tradeoff.png'}")


if __name__ == "__main__":
    main()
