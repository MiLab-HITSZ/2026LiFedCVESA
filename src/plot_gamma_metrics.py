import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_PATTERN = re.compile(
    r"(?P<prefix>.+?)_Gama\[(?P<gamma>[^\]]+)\]_numSteal\[(?P<num_steal>[^\]]+)\]_(?P<metric>acc|mape|loss)\.npy$"
)

METRIC_META = {
    "acc": {"title": "Accuracy", "ylabel": "Accuracy"},
    "mape": {"title": "MAPE", "ylabel": "MAPE"},
    "loss": {"title": "Loss", "ylabel": "Loss"},
}

LINE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
MAPE_START_ROUND = 51


def parse_result_files(results_dir: Path):
    grouped_files = defaultdict(lambda: defaultdict(dict))

    for file_path in sorted(results_dir.glob("*.npy")):
        match = RESULT_PATTERN.match(file_path.name)
        if not match:
            continue

        prefix = match.group("prefix")
        dataset = prefix.split("_", 1)[0]
        experiment_id = f"{prefix}_numSteal[{match.group('num_steal')}]"
        metric = match.group("metric")
        gamma = float(match.group("gamma"))

        grouped_files[(dataset, experiment_id)][metric][gamma] = file_path

    return grouped_files


def build_output_name(dataset: str, experiment_id: str, metric: str, multi_experiment: bool):
    if not multi_experiment:
        return f"{dataset}_{metric}_by_gamma.png"

    safe_experiment = experiment_id.replace("/", "_").replace(" ", "_")
    return f"{safe_experiment}_{metric}_by_gamma.png"


def plot_metric(dataset: str, metric: str, gamma_to_file, output_path: Path):
    meta = METRIC_META[metric]
    plt.figure(figsize=(10, 6), dpi=120)

    for index, gamma in enumerate(sorted(gamma_to_file)):
        values = np.load(gamma_to_file[gamma])
        if metric == "mape":
            rounds = np.arange(MAPE_START_ROUND, MAPE_START_ROUND + len(values))
        else:
            rounds = np.arange(1, len(values) + 1)
        marker = LINE_MARKERS[index % len(LINE_MARKERS)]

        plt.plot(
            rounds,
            values,
            label=fr"$\gamma={gamma:g}$",
            linewidth=2,
            marker=marker,
            markersize=4,
            markevery=max(len(values) // 12, 1),
        )

    plt.title(f"{dataset.upper()} {meta['title']} under Different Gamma")
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
        description="Plot acc, mape, and loss curves for each dataset across different gamma values."
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

    experiment_count_by_dataset = defaultdict(int)
    for dataset, _ in grouped_files:
        experiment_count_by_dataset[dataset] += 1

    saved_paths = []
    for (dataset, experiment_id), metric_map in sorted(grouped_files.items()):
        multi_experiment = experiment_count_by_dataset[dataset] > 1

        for metric in ("acc", "mape", "loss"):
            gamma_to_file = metric_map.get(metric)
            if not gamma_to_file:
                continue

            output_name = build_output_name(dataset, experiment_id, metric, multi_experiment)
            output_path = output_dir / output_name
            plot_metric(dataset, metric, gamma_to_file, output_path)
            saved_paths.append(output_path)
            print(f"Saved: {output_path}")

    print(f"Generated {len(saved_paths)} figure(s).")


if __name__ == "__main__":
    main()
