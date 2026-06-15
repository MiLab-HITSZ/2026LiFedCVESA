import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_PATTERN = re.compile(
    r"(?P<dataset>mnist|fmnist|cifar)_(?P<model>cnn|resnet18)_50_C\[1.0\]_iid\[1\]"
    r"_E\[(?P<local_ep>\d+)\]_B\[(?P<local_bs>\d+)\]"
    r"_Gama\[(?P<gamma>[^\]]+)\]_numSteal\[5\]"
    r"(?:_AttackPos\[spread\])?_(?P<metric>acc|loss|mape)\.npy$"
)

EXPERIMENTS = [
    ("mnist", "cnn", "MNIST / CNN"),
    ("fmnist", "cnn", "Fashion-MNIST / CNN"),
    ("cifar", "cnn", "CIFAR-10 / CNN"),
    ("cifar", "resnet18", "CIFAR-10 / ResNet18"),
]

GAMMAS = [0.05, 0.2, 0.5, 1.0]
GAMMA_LABELS = {
    0.05: r"$\gamma=0.05$",
    0.2: r"$\gamma=0.2$",
    0.5: r"$\gamma=0.5$",
    1.0: r"$\gamma=1.0$",
}

COLORS = {
    0.05: "#4C78A8",
    0.2: "#F58518",
    0.5: "#54A24B",
    1.0: "#B279A2",
}

MARKERS = {
    0.05: "o",
    0.2: "s",
    0.5: "^",
    1.0: "D",
}

SCATTER_ZORDERS = {
    0.05: 3,
    0.2: 4,
    0.5: 6,
    1.0: 5,
}

LABEL_OFFSETS = {
    ("mnist", "cnn", 0.05): (5, 6),
    ("mnist", "cnn", 0.2): (0, -5),
    ("mnist", "cnn", 0.5): (0, -5),
    ("mnist", "cnn", 1.0): (5, 1),
    ("fmnist", "cnn", 0.05): (5, 4),
    ("fmnist", "cnn", 0.2): (5, 6),
    ("fmnist", "cnn", 0.5): (-5, 6),
    ("fmnist", "cnn", 1.0): (-5, -5),
    ("cifar", "cnn", 0.05): (5, 5),
    ("cifar", "cnn", 0.2): (5, 5),
    ("cifar", "cnn", 0.5): (5, 6),
    ("cifar", "cnn", 1.0): (-5, 5),
    ("cifar", "resnet18", 0.05): (9, 8),
    ("cifar", "resnet18", 0.2): (-11, 0),
    ("cifar", "resnet18", 0.5): (5, -6),
    ("cifar", "resnet18", 1.0): (-3, 9),
}

MAPE_START_ROUND = 1
TOTAL_ROUNDS = 50


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
        model = match.group("model")
        gamma = float(match.group("gamma"))
        metric = match.group("metric")
        if gamma not in [0.0] + GAMMAS:
            continue

        data.setdefault((dataset, model), {}).setdefault(gamma, {})[metric] = np.load(path)

    return data


def require_metric(data, dataset, model, gamma, metric):
    try:
        return data[(dataset, model)][gamma][metric]
    except KeyError as exc:
        raise FileNotFoundError(
            f"Missing {metric} data for dataset={dataset}, model={model}, gamma={gamma}"
        ) from exc


def label_position(dataset, model, gamma):
    dx, dy = LABEL_OFFSETS.get((dataset, model, gamma), (4, 4))
    ha = "left" if dx >= 0 else "right"
    if dy > 2:
        va = "bottom"
    elif dy < -2:
        va = "top"
    else:
        va = "center"
    return dx, dy, ha, va


def plot_tradeoff(data, output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.35))
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.085, top=0.835, wspace=0.32, hspace=0.48)
    axes = axes.flatten()

    legend_handles = []
    legend_labels = []

    for ax, (dataset, model, title) in zip(axes, EXPERIMENTS):
        baseline_acc = require_metric(data, dataset, model, 0.0, "acc")[-1] * 100.0
        ax.axhline(
            baseline_acc,
            color="#444444",
            linestyle="--",
            linewidth=1.1,
            alpha=0.85,
            label="No attack",
        )

        x_values = []
        y_values = []
        for gamma in GAMMAS:
            mape = require_metric(data, dataset, model, gamma, "mape")[-1]
            acc = require_metric(data, dataset, model, gamma, "acc")[-1] * 100.0
            x_values.append(mape)
            y_values.append(acc)

            handle = ax.scatter(
                mape,
                acc,
                s=42,
                marker=MARKERS[gamma],
                color=COLORS[gamma],
                edgecolor="white",
                linewidth=0.7,
                zorder=SCATTER_ZORDERS[gamma],
            )
            dx, dy, ha, va = label_position(dataset, model, gamma)
            ax.annotate(
                f"{gamma:g}",
                xy=(mape, acc),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=7,
                color=COLORS[gamma],
                zorder=7,
            )

            if gamma not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(gamma)

        ax.set_title(title)
        ax.set_xlabel("Final MAPE")
        ax.set_ylabel("Final accuracy (%)")
        ax.grid(True, alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        x_min, x_max = min(x_values), max(x_values)
        y_min = min(min(y_values), baseline_acc)
        y_max = max(max(y_values), baseline_acc)
        x_pad = max((x_max - x_min) * 0.18, 0.01)
        y_pad = max((y_max - y_min) * 0.22, 0.35)
        ax.set_xlim(max(0, x_min - x_pad), x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    handles = legend_handles + [
        plt.Line2D([0], [0], color="#444444", linestyle="--", linewidth=1.1)
    ]
    labels = [GAMMA_LABELS[gamma] for gamma in legend_labels] + ["No attack"]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        borderaxespad=0.0,
    )

    fig.savefig(output_dir / "paper_gamma_tradeoff.png", dpi=400)
    plt.close(fig)


def plot_mape_curves(data, output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.35))
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.085, top=0.845, wspace=0.32, hspace=0.48)
    axes = axes.flatten()

    for ax, (dataset, model, title) in zip(axes, EXPERIMENTS):
        for gamma in GAMMAS:
            values = require_metric(data, dataset, model, gamma, "mape")
            rounds = np.arange(MAPE_START_ROUND, MAPE_START_ROUND + len(values))
            ax.plot(
                rounds,
                values,
                color=COLORS[gamma],
                label=GAMMA_LABELS[gamma],
                marker=MARKERS[gamma],
                markersize=3.2,
                markevery=max(len(values) // 6, 1),
            )

        ax.set_title(title)
        ax.set_xlabel("Communication round")
        ax.set_ylabel("MAPE (lower is better)")
        ax.set_xlim(MAPE_START_ROUND, TOTAL_ROUNDS)
        ax.grid(True, alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        borderaxespad=0.0,
    )

    fig.savefig(output_dir / "paper_gamma_mape_curves.png", dpi=400)
    plt.close(fig)


def main():
    results_dir = Path("save/results")
    output_dir = Path("save/plots/paper")
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    data = load_results(results_dir)
    plot_tradeoff(data, output_dir)
    plot_mape_curves(data, output_dir)

    print(f"Saved paper figures to {output_dir}")
    print(f"- {output_dir / 'paper_gamma_tradeoff.png'}")
    print(f"- {output_dir / 'paper_gamma_mape_curves.png'}")


if __name__ == "__main__":
    main()
