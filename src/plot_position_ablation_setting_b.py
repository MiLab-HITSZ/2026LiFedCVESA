import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_PATTERN = re.compile(
    r"(?P<dataset>mnist|fmnist|cifar)_cnn_200_C\[1.0\]_iid\[1\]"
    r"_E\[(?P<local_ep>\d+)\]_B\[(?P<local_bs>\d+)\]"
    r"_Gama\[0.5\]_numSteal\[5\]_numImgPerClient\[10\]"
    r"_AttackPos\[(?P<position>front|spread)\]_(?P<metric>acc|loss|mape)\.npy$"
)

EXPERIMENTS = [
    ("mnist", "MNIST"),
    ("fmnist", "Fashion-MNIST"),
    ("cifar", "CIFAR-10"),
]

POSITIONS = ["front", "spread"]
POSITION_LABELS = {
    "front": "Contiguous placement",
    "spread": "Dispersed placement",
}

COLORS = {
    "front": "#4C78A8",
    "spread": "#C44E52",
}

HATCHES = {
    "front": "////",
    "spread": "\\\\\\\\",
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
        "patch.linewidth": 0.7,
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
        position = match.group("position")
        metric = match.group("metric")
        data.setdefault(dataset, {}).setdefault(position, {})[metric] = np.load(path)

    return data


def require_final_value(data, dataset, position, metric):
    try:
        values = data[dataset][position][metric]
    except KeyError as exc:
        raise FileNotFoundError(
            f"Missing {metric} data for dataset={dataset}, position={position}"
        ) from exc

    if len(values) == 0:
        raise ValueError(
            f"Empty {metric} data for dataset={dataset}, position={position}"
        )

    return float(values[-1])


def add_value_labels(ax, bars, fmt, dy):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#2A2A2A",
        )


def draw_metric_panel(ax, values, ylabel, title, y_is_percent=False):
    x = np.arange(len(EXPERIMENTS))
    width = 0.34
    offsets = {
        "front": -width / 2.0,
        "spread": width / 2.0,
    }

    handles = []
    for position in POSITIONS:
        bar_values = np.array([
            values[dataset][position]
            for dataset, _ in EXPERIMENTS
        ])
        if y_is_percent:
            bar_values = bar_values * 100.0

        bars = ax.bar(
            x + offsets[position],
            bar_values,
            width=width,
            color=COLORS[position],
            alpha=0.88,
            edgecolor="#1F1F1F",
            linewidth=0.6,
            hatch=HATCHES[position],
            label=POSITION_LABELS[position],
            zorder=3,
        )
        handles.append(bars[0])

        if y_is_percent:
            add_value_labels(ax, bars, "{:.2f}", 2)
        else:
            add_value_labels(ax, bars, "{:.3f}", 2)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in EXPERIMENTS])
    ax.grid(True, axis="y", alpha=0.28, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)

    plotted_values = np.array([
        values[dataset][position] * (100.0 if y_is_percent else 1.0)
        for dataset, _ in EXPERIMENTS
        for position in POSITIONS
    ])
    value_min = float(plotted_values.min())
    value_max = float(plotted_values.max())
    pad = max((value_max - value_min) * 0.35, 0.02 if not y_is_percent else 0.6)
    ax.set_ylim(max(0.0, value_min - pad), value_max + pad)

    return handles


def plot_setting_b(data, output_dir: Path):
    final_acc = {
        dataset: {
            position: require_final_value(data, dataset, position, "acc")
            for position in POSITIONS
        }
        for dataset, _ in EXPERIMENTS
    }
    final_mape = {
        dataset: {
            position: require_final_value(data, dataset, position, "mape")
            for position in POSITIONS
        }
        for dataset, _ in EXPERIMENTS
    }

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.65))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.19, top=0.78, wspace=0.34)

    handles = draw_metric_panel(
        axes[0],
        final_acc,
        "Final accuracy (%)",
        "(a) Model utility",
        y_is_percent=True,
    )
    draw_metric_panel(
        axes[1],
        final_mape,
        "Final MAPE",
        "(b) Reconstruction error",
        y_is_percent=False,
    )

    fig.legend(
        handles,
        [POSITION_LABELS[position] for position in POSITIONS],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        borderaxespad=0.0,
        handlelength=1.8,
        columnspacing=1.6,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "paper_position_ablation_setting_b.png"
    fig.savefig(output_path, dpi=400)
    plt.close(fig)
    return output_path, final_acc, final_mape


def main():
    results_dir = Path("save/results")
    output_dir = Path("save/plots/paper")

    configure_style()
    data = load_results(results_dir)
    output_path, final_acc, final_mape = plot_setting_b(data, output_dir)

    print(f"Saved setting-B position ablation figure to {output_path}")
    for dataset, label in EXPERIMENTS:
        front_acc = final_acc[dataset]["front"] * 100.0
        spread_acc = final_acc[dataset]["spread"] * 100.0
        front_mape = final_mape[dataset]["front"]
        spread_mape = final_mape[dataset]["spread"]
        print(
            f"{label}: "
            f"Contiguous placement acc={front_acc:.2f}%, "
            f"Dispersed placement acc={spread_acc:.2f}%, "
            f"Contiguous placement MAPE={front_mape:.4f}, "
            f"Dispersed placement MAPE={spread_mape:.4f}"
        )


if __name__ == "__main__":
    main()
