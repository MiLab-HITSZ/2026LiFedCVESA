from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


RESULTS_DIR = Path("save/results")
PLOTS_DIR = Path("save/plots")
OUTPUT_DIR = Path("figures")
CONFIG_TAG = "rank10_dirichlet05"

DATASETS = [
    {
        "key": "mnist",
        "model": "cnn",
        "label": "MNIST / CNN",
        "tag_dataset": "mnist",
        "file_prefix": "mnist_cnn",
    },
    {
        "key": "fmnist",
        "model": "cnn",
        "label": "Fashion-MNIST / CNN",
        "tag_dataset": "fmnist",
        "file_prefix": "fmnist_cnn",
    },
    {
        "key": "cifar",
        "model": "resnet18_cifar",
        "label": "CIFAR-10 / ResNet18",
        "tag_dataset": "cifar",
        "file_prefix": "cifar_resnet18_cifar",
    },
]

GAMMAS = [0.05, 0.2, 0.5, 1.0]
GAMMA_TAGS = {0.05: "0.05", 0.2: "0.2", 0.5: "0.5", 1.0: "1.0"}
NUM_STEAL_VALUES = [1, 2, 3, 4, 5, 10]

GAMMA_COLORS = {
    0.05: "#4C78A8",
    0.2: "#F58518",
    0.5: "#54A24B",
    1.0: "#B279A2",
}
GAMMA_MARKERS = {0.05: "o", 0.2: "s", 0.5: "^", 1.0: "D"}

# Offset for gamma labels in paper_gamma_tradeoff.png.
# Keys are (dataset_key, gamma); values are (dx, dy) in display points.
# Positive dx moves right, negative dx moves left; positive dy moves up.
GAMMA_LABEL_OFFSETS = {
    ("mnist", 0.05): (5, 8),
    ("mnist", 0.2): (8, 5),
    ("mnist", 0.5): (0, 8),
    ("mnist", 1.0): (-8, 3),

    ("fmnist", 0.05): (7, 7),
    ("fmnist", 0.2): (7, 5),
    ("fmnist", 0.5): (7, 5),
    ("fmnist", 1.0): (5, 7),

    ("cifar", 0.05): (7, 7),
    ("cifar", 0.2): (7, 7),
    ("cifar", 0.5): (0, -8),
    ("cifar", 1.0): (-3, 7),
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


def find_metric_file(tag, metric):
    matches = [
        path for path in RESULTS_DIR.iterdir()
        if path.is_file() and path.name.endswith(f"_Tag[{tag}]_{metric}.npy")
    ]
    if not matches:
        raise FileNotFoundError(f"Missing {metric} result for tag {tag}")
    if len(matches) > 1:
        matches = sorted(matches, key=lambda path: path.stat().st_mtime)
    return matches[-1]


def load_metric(tag, metric):
    values = np.asarray(np.load(find_metric_file(tag, metric), allow_pickle=True), dtype=float).reshape(-1)
    if metric == "acc" and values.size and np.nanmax(values) <= 1.5:
        values = values * 100.0
    return values


def gamma_tag(dataset, gamma):
    suffix = "g0" if gamma == 0 else f"g{GAMMA_TAGS[gamma]}"
    return f"{CONFIG_TAG}_gamma_{dataset['tag_dataset']}_{dataset['model']}_{suffix}"


def numsteal_tag(dataset, num_steal):
    return f"{CONFIG_TAG}_numsteal_{dataset['tag_dataset']}_{dataset['model']}_n{num_steal}"


def gamma_label_alignment(offset):
    dx, dy = offset
    ha = "left" if dx >= 0 else "right"
    if dy > 2:
        va = "bottom"
    elif dy < -2:
        va = "top"
    else:
        va = "center"
    return ha, va


def plot_gamma_tradeoff():
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.95))
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.22, top=0.74, wspace=0.34)

    legend_handles = []
    legend_labels = []

    for ax, dataset in zip(axes, DATASETS):
        baseline_acc = load_metric(gamma_tag(dataset, 0), "acc")[-1]
        ax.axhline(
            baseline_acc,
            color="#444444",
            linestyle="--",
            linewidth=1.1,
            alpha=0.85,
        )

        x_values = []
        y_values = []
        for gamma in GAMMAS:
            tag = gamma_tag(dataset, gamma)
            mape = load_metric(tag, "mape")[-1]
            acc = load_metric(tag, "acc")[-1]
            x_values.append(mape)
            y_values.append(acc)

            handle = ax.scatter(
                mape,
                acc,
                s=44,
                marker=GAMMA_MARKERS[gamma],
                color=GAMMA_COLORS[gamma],
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
            label_offset = GAMMA_LABEL_OFFSETS.get((dataset["key"], gamma), (5, 5))
            label_ha, label_va = gamma_label_alignment(label_offset)
            ax.annotate(
                f"{gamma:g}",
                xy=(mape, acc),
                xytext=label_offset,
                textcoords="offset points",
                ha=label_ha,
                va=label_va,
                fontsize=7,
                color=GAMMA_COLORS[gamma],
            )
            if gamma not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(gamma)

        ax.set_title(dataset["label"])
        ax.set_xlabel("Final MAPE")
        ax.set_ylabel("Final accuracy (%)")
        ax.grid(True, alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        x_min, x_max = min(x_values), max(x_values)
        y_min = min(min(y_values), baseline_acc)
        y_max = max(max(y_values), baseline_acc)
        x_pad = max((x_max - x_min) * 0.18, 0.01)
        y_pad = max((y_max - y_min) * 0.25, 0.35)
        ax.set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    handles = legend_handles + [Line2D([0], [0], color="#444444", linestyle="--", linewidth=1.1)]
    labels = [rf"$\gamma={gamma:g}$" for gamma in legend_labels] + ["No attack"]
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Attack Strength Analysis: Dirichlet non-IID $\\alpha=0.5$, 100 rounds", y=1.02, fontsize=10)
    fig.savefig(OUTPUT_DIR / "paper_gamma_tradeoff.png", dpi=400)
    plt.close(fig)


def plot_num_steal_effect():
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 2.95))
    fig.subplots_adjust(left=0.060, right=0.960, bottom=0.20, top=0.76, wspace=0.88)

    legend_handles = []
    legend_labels = []

    for ax, dataset in zip(axes, DATASETS):
        x = np.array(NUM_STEAL_VALUES)
        acc = np.array([load_metric(numsteal_tag(dataset, n), "acc")[-1] for n in NUM_STEAL_VALUES])
        mape = np.array([load_metric(numsteal_tag(dataset, n), "mape")[-1] for n in NUM_STEAL_VALUES])
        baseline_acc = load_metric(gamma_tag(dataset, 0), "acc")[-1]

        acc_line, = ax.plot(
            x,
            acc,
            color="#2F5597",
            marker="o",
            markersize=4.2,
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
            markersize=4.2,
            label="Final MAPE",
            zorder=3,
        )

        if not legend_handles:
            legend_handles = [acc_line, mape_line, baseline_line]
            legend_labels = ["Final accuracy", "Final MAPE", "No-attack accuracy"]

        ax.set_title(dataset["label"])
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

    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Target-Client Scale Analysis: Dirichlet non-IID $\\alpha=0.5$, 100 rounds", y=1.02, fontsize=10)
    fig.savefig(OUTPUT_DIR / "paper_num_steal_effect.png", dpi=400)
    plt.close(fig)


SOURCE_PLOTS = [
    {
        "label": "MNIST / CNN",
        "path": PLOTS_DIR / "mnist_cnn_100_C[1.0]_iid[0]_E[10]_B[16]_Gama[0.5]_numSteal[5]_AttackPos[spread]_Tag[rank10_dirichlet05_gamma_mnist_cnn_g0.5]_final_comparison.png",
        # Source plot sample indices are 0-based. Use [1, 3, 4] for source images 2, 4, 5.
        "samples": [0, 1, 2],
        "layout": {},
    },
    {
        "label": "Fashion-MNIST / CNN",
        "path": PLOTS_DIR / "fmnist_cnn_100_C[1.0]_iid[0]_E[10]_B[16]_Gama[0.5]_numSteal[5]_AttackPos[spread]_Tag[rank10_dirichlet05_gamma_fmnist_cnn_g0.5]_final_comparison.png",
        "samples": [0, 1, 2],
        "layout": {},
    },
    {
        "label": "CIFAR-10 / ResNet18",
        "path": PLOTS_DIR / "cifar_resnet18_cifar_100_C[1.0]_iid[0]_E[1]_B[64]_Gama[0.5]_numSteal[5]_AttackPos[spread]_Tag[rank10_dirichlet05_gamma_cifar_resnet18_cifar_g0.5]_final_comparison.png",
        "samples": [0, 1, 2],
        "layout": {},
    },
]

# Layout knobs for paper_final_comparison_examples.png.
# Coordinates are normalized within each dataset panel. Tweak these values and
# re-run this script if labels or image groups need manual spacing.
FINAL_COMPARISON_LAYOUT = {
    "figsize": (9.85, 3.15),
    "panel_left": 0.015,
    "panel_right": 0.985,
    "panel_bottom": 0.03,
    "panel_top": 0.97,
    "panel_gap": 0.014,
    "title_y": 0.895,
    "title_dx": 0.0,
    "row_label_x": 0.052,
    "original_label_y": 0.630,
    "recovered_label_y": 0.210,
    "image_x_start": 0.260,
    "image_x_step": 0.255,
    "image_y_original": 0.525,
    "image_y_recovered": 0.105,
    "image_size": 0.205,
    "separator_y": 0.455,
    "separator_x0": 0.050,
    "separator_x1": 0.955,
    "title_fontsize": 11,
    "label_fontsize": 8,
}

CELL_SIZE = 221
SOURCE_COL_START = 46
SOURCE_COL_STEP = 297
SOURCE_ORIGINAL_Y = 98
SOURCE_RECOVERED_Y = 346


def crop_cell(source, sample_idx, row_y):
    x0 = SOURCE_COL_START + sample_idx * SOURCE_COL_STEP
    return source.crop((x0, row_y, x0 + CELL_SIZE, row_y + CELL_SIZE))


def merged_layout(overrides):
    layout = dict(FINAL_COMPARISON_LAYOUT)
    layout.update(overrides or {})
    return layout


def load_dataset_pairs(path, sample_indices):
    if not path.exists():
        raise FileNotFoundError(f"Missing source comparison plot: {path}")
    source = Image.open(path).convert("RGB")
    pairs = []
    for sample_idx in sample_indices:
        original = crop_cell(source, sample_idx, SOURCE_ORIGINAL_Y)
        recovered = crop_cell(source, sample_idx, SOURCE_RECOVERED_Y)
        pairs.append((np.asarray(original), np.asarray(recovered)))
    return pairs


def add_panel_frame(fig, panel):
    x, y, w, h = panel
    frame = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.002,rounding_size=0.008",
        linewidth=0.85,
        edgecolor="#333333",
        facecolor="none",
        transform=fig.transFigure,
        clip_on=False,
    )
    fig.add_artist(frame)


def panel_to_figure(panel, x, y, width, height):
    px, py, pw, ph = panel
    return px + x * pw, py + y * ph, width * pw, height * ph


def add_image(fig, panel, image, sample_idx, row_y, layout):
    x = layout["image_x_start"] + sample_idx * layout["image_x_step"]
    ax = fig.add_axes(panel_to_figure(panel, x, row_y, layout["image_size"], layout["image_size"]))
    ax.imshow(image, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.45)
        spine.set_color("#333333")


def plot_final_comparison_examples():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "standard",
    })

    data = [
        (
            source["label"],
            load_dataset_pairs(source["path"], source["samples"]),
            merged_layout(source.get("layout")),
        )
        for source in SOURCE_PLOTS
    ]

    base_layout = FINAL_COMPARISON_LAYOUT
    fig = plt.figure(figsize=base_layout["figsize"], facecolor="white")
    panel_left = base_layout["panel_left"]
    panel_right = base_layout["panel_right"]
    panel_bottom = base_layout["panel_bottom"]
    panel_top = base_layout["panel_top"]
    panel_gap = base_layout["panel_gap"]
    panel_width = (panel_right - panel_left - panel_gap * (len(data) - 1)) / len(data)
    panel_height = panel_top - panel_bottom

    for dataset_idx, (dataset_label, pairs, layout) in enumerate(data):
        panel = (
            panel_left + dataset_idx * (panel_width + panel_gap),
            panel_bottom,
            panel_width,
            panel_height,
        )
        add_panel_frame(fig, panel)
        px, py, pw, ph = panel
        fig.text(
            px + (0.5 + layout["title_dx"]) * pw,
            py + layout["title_y"] * ph,
            dataset_label,
            ha="center",
            va="center",
            fontsize=layout["title_fontsize"],
            fontweight="bold",
        )
        fig.text(
            px + layout["row_label_x"] * pw,
            py + layout["original_label_y"] * ph,
            "Original",
            ha="left",
            va="center",
            fontsize=layout["label_fontsize"],
        )
        fig.text(
            px + layout["row_label_x"] * pw,
            py + layout["recovered_label_y"] * ph,
            "FedCVESA",
            ha="left",
            va="center",
            fontsize=layout["label_fontsize"],
        )
        sep = Line2D(
            [px + layout["separator_x0"] * pw, px + layout["separator_x1"] * pw],
            [py + layout["separator_y"] * ph, py + layout["separator_y"] * ph],
            linewidth=0.8,
            linestyle=(0, (4, 2.5)),
            color="#bdbdbd",
            transform=fig.transFigure,
            solid_capstyle="butt",
        )
        fig.add_artist(sep)

        for sample_idx, (original, recovered) in enumerate(pairs):
            add_image(fig, panel, original, sample_idx, layout["image_y_original"], layout)
            add_image(fig, panel, recovered, sample_idx, layout["image_y_recovered"], layout)

    fig.savefig(OUTPUT_DIR / "paper_final_comparison_examples.png", dpi=400, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_gamma_tradeoff()
    plot_num_steal_effect()
    plot_final_comparison_examples()
    print(f"Saved figures to {OUTPUT_DIR}")
    print(f"- {OUTPUT_DIR / 'paper_gamma_tradeoff.png'}")
    print(f"- {OUTPUT_DIR / 'paper_num_steal_effect.png'}")
    print(f"- {OUTPUT_DIR / 'paper_final_comparison_examples.png'}")


if __name__ == "__main__":
    main()
