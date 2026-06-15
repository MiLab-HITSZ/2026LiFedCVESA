from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SOURCE_PLOTS = [
    (
        "MNIST / CNN",
        Path(
            "save/plots/"
            "mnist_cnn_50_C[1.0]_iid[1]_E[10]_B[16]"
            "_Gama[0.5]_numSteal[5]_AttackPos[spread]_final_comparison.png"
        ),
    ),
    (
        "Fashion-MNIST / CNN",
        Path(
            "save/plots/"
            "fmnist_cnn_50_C[1.0]_iid[1]_E[10]_B[16]"
            "_Gama[0.5]_numSteal[5]_AttackPos[spread]_final_comparison.png"
        ),
    ),
    (
        "CIFAR-10 / CNN",
        Path(
            "save/plots/"
            "cifar_cnn_50_C[1.0]_iid[1]_E[5]_B[16]"
            "_Gama[0.5]_numSteal[5]_AttackPos[spread]_final_comparison.png"
        ),
    ),
]

OUTPUT_PATH = Path("save/plots/paper/paper_final_comparison_examples.png")

NUM_PAIRS = 3
CELL_SIZE = 221
SOURCE_COL_START = 46
SOURCE_COL_STEP = 297
SOURCE_ORIGINAL_Y = 98
SOURCE_RECOVERED_Y = 346

FIGSIZE = (9.85, 3.15)
PANEL_TOP = 0.97
PANEL_BOTTOM = 0.03
PANEL_LEFT = 0.015
PANEL_RIGHT = 0.985
PANEL_GAP = 0.014

TITLE_Y = 0.895
LABEL_X = 0.045
IMAGE_X_START = 0.255
IMAGE_X_STEP = 0.25
IMAGE_SIZE = 0.21
ORIGINAL_Y = 0.525
RECOVERED_Y = 0.105
SEPARATOR_Y = 0.455

TITLE_FONTSIZE = 11
LABEL_FONTSIZE = 8


def configure_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": LABEL_FONTSIZE,
        "axes.titlesize": LABEL_FONTSIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "standard",
    })


def crop_cell(source: Image.Image, sample_idx: int, row_y: int):
    x0 = SOURCE_COL_START + sample_idx * SOURCE_COL_STEP
    y0 = row_y
    return source.crop((x0, y0, x0 + CELL_SIZE, y0 + CELL_SIZE))


def load_dataset_pairs(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing source comparison plot: {path}")

    source = Image.open(path).convert("RGB")
    pairs = []
    for sample_idx in range(NUM_PAIRS):
        original = crop_cell(source, sample_idx, SOURCE_ORIGINAL_Y)
        recovered = crop_cell(source, sample_idx, SOURCE_RECOVERED_Y)
        pairs.append((np.asarray(original), np.asarray(recovered)))
    return pairs


def panel_to_figure(panel, x, y, width=0.0, height=0.0):
    panel_x, panel_y, panel_w, panel_h = panel
    return (
        panel_x + x * panel_w,
        panel_y + y * panel_h,
        width * panel_w,
        height * panel_h,
    )


def add_panel_frame(fig, panel):
    panel_x, panel_y, panel_w, panel_h = panel
    frame = FancyBboxPatch(
        (panel_x, panel_y),
        panel_w,
        panel_h,
        boxstyle="round,pad=0.002,rounding_size=0.012",
        linewidth=0.85,
        edgecolor="#333333",
        facecolor="none",
        transform=fig.transFigure,
        clip_on=False,
    )
    fig.add_artist(frame)


def add_panel_text(fig, panel, dataset_label):
    panel_x, panel_y, panel_w, panel_h = panel
    fig.text(
        panel_x + panel_w / 2.0,
        panel_y + TITLE_Y * panel_h,
        dataset_label,
        ha="center",
        va="center",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )
    for label, y in (("Original", ORIGINAL_Y + IMAGE_SIZE / 2), ("FedCVESA", RECOVERED_Y + IMAGE_SIZE / 2)):
        fig.text(
            panel_x + LABEL_X * panel_w,
            panel_y + y * panel_h,
            label,
            ha="left",
            va="center",
            fontsize=LABEL_FONTSIZE,
        )


def add_separator(fig, panel):
    panel_x, panel_y, panel_w, panel_h = panel
    line = Line2D(
        [panel_x + 0.045 * panel_w, panel_x + 0.955 * panel_w],
        [panel_y + SEPARATOR_Y * panel_h, panel_y + SEPARATOR_Y * panel_h],
        linewidth=0.8,
        linestyle=(0, (4, 2.5)),
        color="#bdbdbd",
        transform=fig.transFigure,
        solid_capstyle="butt",
    )
    fig.add_artist(line)


def add_image(fig, panel, image, sample_idx, row_y):
    x = IMAGE_X_START + sample_idx * IMAGE_X_STEP
    ax = fig.add_axes(panel_to_figure(panel, x, row_y, IMAGE_SIZE, IMAGE_SIZE))
    ax.imshow(image, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.45)
        spine.set_color("#333333")


def main():
    configure_style()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = [
        (dataset_label, load_dataset_pairs(path))
        for dataset_label, path in SOURCE_PLOTS
    ]

    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    panel_width = (PANEL_RIGHT - PANEL_LEFT - PANEL_GAP * (len(data) - 1)) / len(data)
    panel_height = PANEL_TOP - PANEL_BOTTOM

    for dataset_idx, (dataset_label, pairs) in enumerate(data):
        panel = (
            PANEL_LEFT + dataset_idx * (panel_width + PANEL_GAP),
            PANEL_BOTTOM,
            panel_width,
            panel_height,
        )
        add_panel_frame(fig, panel)
        add_panel_text(fig, panel, dataset_label)
        add_separator(fig, panel)

        for sample_idx, (original, recovered) in enumerate(pairs):
            add_image(fig, panel, original, sample_idx, ORIGINAL_Y)
            add_image(fig, panel, recovered, sample_idx, RECOVERED_Y)

    fig.savefig(OUTPUT_PATH, dpi=400, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
