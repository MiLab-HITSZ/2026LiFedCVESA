import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_PATTERN = re.compile(
    r"(?P<prefix>.+?)_Gama\[(?P<gamma>[^\]]+)\]_numSteal\[(?P<num_steal>[^\]]+)\]"
    r"(?:_Agg\[(?P<agg_mode>[^\]]+)\])?(?:_Alpha\[(?P<alpha>[^\]]+)\])?"
    r"_(?P<metric>acc|mape|loss)\.npy$"
)

METRIC_META = {
    "acc": {"title": "Accuracy", "ylabel": "Accuracy"},
    "mape": {"title": "MAPE", "ylabel": "MAPE"},
    "loss": {"title": "Loss", "ylabel": "Loss"},
}

LINE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
MAPE_START_ROUND = 51


def build_agg_label(agg_mode, alpha):
    if not agg_mode:
        return "segmented"
    if agg_mode == "segmented_soft":
        return f"segmented_soft(alpha={alpha})"
    return agg_mode


def parse_result_files(results_dir: Path):
    grouped_files = defaultdict(lambda: defaultdict(dict))

    for file_path in sorted(results_dir.glob("*.npy")):
        match = RESULT_PATTERN.match(file_path.name)
        if not match:
            continue

        prefix = match.group("prefix")
        dataset = prefix.split("_", 1)[0]
        gamma = float(match.group("gamma"))
        num_steal = match.group("num_steal")
        metric = match.group("metric")
        agg_label = build_agg_label(match.group("agg_mode"), match.group("alpha"))
        experiment_id = f"{prefix}_Gama[{gamma:g}]_numSteal[{num_steal}]"

        grouped_files[(dataset, experiment_id)][metric][agg_label] = file_path

    return grouped_files


def build_output_name(experiment_id: str, metric: str):
    safe_experiment = experiment_id.replace("/", "_").replace(" ", "_")
    return f"{safe_experiment}_{metric}_by_agg.png"


def plot_metric(dataset: str, experiment_id: str, metric: str, agg_to_file, output_path: Path):
    meta = METRIC_META[metric]
    plt.figure(figsize=(10, 6), dpi=120)

    ordered_labels = sorted(
        agg_to_file,
        key=lambda label: (
            0 if label == "segmented" else
            1 if label == "target_only_avg" else
            2 if label == "avg" else
            3,
            label
        )
    )

    for index, agg_label in enumerate(ordered_labels):
        values = np.load(agg_to_file[agg_label])
        if metric == "mape":
            rounds = np.arange(MAPE_START_ROUND, MAPE_START_ROUND + len(values))
        else:
            rounds = np.arange(1, len(values) + 1)
        marker = LINE_MARKERS[index % len(LINE_MARKERS)]

        plt.plot(
            rounds,
            values,
            label=agg_label,
            linewidth=2,
            marker=marker,
            markersize=4,
            markevery=max(len(values) // 12, 1),
        )

    plt.title(f"{dataset.upper()} {meta['title']} under Aggregation Ablation")
    plt.xlabel("Communication Rounds")
    plt.ylabel(meta["ylabel"])
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)

    if metric == "acc":
        plt.ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def use_plot_style():
    for style_name in ("seaborn-v0_8-muted", "seaborn-muted"):
        try:
            plt.style.use(style_name)
            return
        except OSError:
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Plot acc, mape, and loss curves across different aggregation modes."
    )
    parser.add_argument(
        "--results-dir",
        default="save/results",
        help="Directory containing .npy result files.",
    )
    parser.add_argument(
        "--output-dir",
        default="save/plots",
        help="Directory to store generated figures.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_plot_style()
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    grouped_files = parse_result_files(results_dir)
    if not grouped_files:
        raise FileNotFoundError(f"No result files matched the expected pattern in {results_dir}.")

    saved_paths = []
    for (dataset, experiment_id), metric_map in sorted(grouped_files.items()):
        for metric in ("acc", "mape", "loss"):
            agg_to_file = metric_map.get(metric)
            if not agg_to_file:
                continue

            output_name = build_output_name(experiment_id, metric)
            output_path = output_dir / output_name
            plot_metric(dataset, experiment_id, metric, agg_to_file, output_path)
            saved_paths.append(output_path)
            print(f"Saved: {output_path}")

    print(f"Generated {len(saved_paths)} figure(s).")


if __name__ == "__main__":
    main()
