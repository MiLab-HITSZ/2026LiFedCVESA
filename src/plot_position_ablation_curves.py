import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASETS = [
    {
        "key": "mnist",
        "label": "MNIST / CNN",
        "local_ep": 10,
    },
    {
        "key": "fmnist",
        "label": "Fashion-MNIST / CNN",
        "local_ep": 10,
    },
    {
        "key": "cifar",
        "label": "CIFAR-10 / CNN",
        "local_ep": 5,
    },
]

SETTINGS = [
    {
        "key": "numimg1",
        "label": "num_steal=10, num_img_per_client=1",
        "num_steal": 10,
        "num_img_per_client": 1,
    },
    {
        "key": "numimg5",
        "label": "num_steal=10, num_img_per_client=5",
        "num_steal": 10,
        "num_img_per_client": 5,
    },
    {
        "key": "numimg10",
        "label": "num_steal=10, num_img_per_client=10",
        "num_steal": 10,
        "num_img_per_client": 10,
    },
    {
        "key": "numimg50",
        "label": "num_steal=10, num_img_per_client=50",
        "num_steal": 10,
        "num_img_per_client": 50,
    },
]

POSITIONS = ["front", "spread"]
POSITION_LABELS = {
    "front": "front",
    "spread": "spread",
}

COLORS = {
    "front": "#4C78A8",
    "spread": "#C44E52",
}

MARKERS = {
    "front": "o",
    "spread": "s",
}

METRICS = [
    ("acc", "Accuracy (%)"),
    ("loss", "Loss"),
    ("mape", "MAPE"),
]


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


def result_path(results_dir: Path, iid: int, dataset, setting, position: str, metric: str):
    suffix = ""
    if setting["num_img_per_client"] != 1:
        suffix = f"_numImgPerClient[{setting['num_img_per_client']}]"

    filename = (
        f"{dataset['key']}_cnn_50_C[1.0]_iid[{iid}]_E[{dataset['local_ep']}]_B[16]"
        f"_Gama[0.5]_numSteal[{setting['num_steal']}]{suffix}"
        f"_AttackPos[{position}]_{metric}.npy"
    )
    return results_dir / filename


def load_metric(results_dir: Path, iid: int, dataset, setting, position: str, metric: str):
    path = result_path(results_dir, iid, dataset, setting, position, metric)
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")

    values = np.load(path)
    if values.size == 0:
        raise ValueError(f"Empty result file: {path}")

    return values


def set_metric_ylim(ax, plotted_values, metric: str):
    values = np.concatenate(plotted_values)
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    pad = max((value_max - value_min) * 0.12, 0.001)

    if metric in ("acc", "mape"):
        ax.set_ylim(max(0.0, value_min - pad), value_max + pad)
    else:
        ax.set_ylim(value_min - pad, value_max + pad)


def plot_dataset_setting(results_dir: Path, output_dir: Path, iid: int, dataset, setting):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), sharex=False)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.19, top=0.75, wspace=0.32)

    legend_handles = []
    legend_labels = []

    for ax, (metric, ylabel) in zip(axes, METRICS):
        plotted_values = []

        for position in POSITIONS:
            values = load_metric(results_dir, iid, dataset, setting, position, metric)
            if metric == "acc":
                values = values * 100.0

            rounds = np.arange(1, len(values) + 1)
            line, = ax.plot(
                rounds,
                values,
                color=COLORS[position],
                marker=MARKERS[position],
                markersize=3.0,
                markevery=max(len(values) // 8, 1),
                label=POSITION_LABELS[position],
            )
            plotted_values.append(values)

            if metric == "acc":
                final_label = f"{values[-1]:.2f}%"
            else:
                final_label = f"{values[-1]:.4f}"
            ax.annotate(
                final_label,
                xy=(rounds[-1], values[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7,
                color=COLORS[position],
            )

            if metric == "acc":
                legend_handles.append(line)
                legend_labels.append(POSITION_LABELS[position])

        ax.set_title(ylabel.replace(" (%)", ""))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, 50)
        ax.grid(True, alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        set_metric_ylim(ax, plotted_values, metric)

    fig.suptitle(f"{dataset['label']} - IID={iid}, {setting['label']}", y=0.96, fontsize=11)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.875),
        borderaxespad=0.0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset['key']}_iid{iid}_setting_{setting['key'].lower()}_position_ablation_curves.png"
    fig.savefig(output_path, dpi=400)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot front/spread position-ablation curves for the latest ablation results."
    )
    parser.add_argument(
        "--results-dir",
        default="save/results",
        help="Directory containing .npy result files.",
    )
    parser.add_argument(
        "--output-dir",
        default="save/plots/ablation",
        help="Directory to store generated figures.",
    )
    parser.add_argument(
        "--iid",
        type=int,
        default=0,
        choices=[0, 1],
        help="Result IID flag to plot. Default: 0 for non-IID.",
    )
    args = parser.parse_args()

    configure_style()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    saved_paths = []
    for setting in SETTINGS:
        for dataset in DATASETS:
            output_path = plot_dataset_setting(results_dir, output_dir, args.iid, dataset, setting)
            saved_paths.append(output_path)
            print(f"Saved: {output_path}")

    print(f"Generated {len(saved_paths)} position-ablation curve figure(s).")


if __name__ == "__main__":
    main()
