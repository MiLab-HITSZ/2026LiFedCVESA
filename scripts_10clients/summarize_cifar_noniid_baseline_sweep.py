#!/usr/bin/env python3

import re
from pathlib import Path

import numpy as np


RESULT_RE = re.compile(
    r"cifar_(?P<model>cnn|resnet18)_(?P<epochs>\d+)_C\[(?P<frac>[^\]]+)\]_iid\[0\]"
    r"_E\[(?P<local_ep>\d+)\]_B\[(?P<local_bs>\d+)\]"
    r"_Gama\[0(?:\.0)?\]_numSteal\[5\]_Agg\[avg\]_AttackPos\[spread\]"
    r"_Tag\[tune_cifar_noniid_(?:(?P<tag_model>cnn|resnet18)_)?lr(?P<lr>[^_]+)_b(?P<tag_bs>\d+)"
    r"_e(?P<tag_ep>\d+)_d(?P<decay>[^_\]]+)"
    r"(?:_c(?P<crop>\d+)_n(?P<normalize>\d+))?\]_acc\.npy$"
)


def decode_float(value):
    return value.replace("p", ".")


def main():
    results_dir = Path("save/results")
    rows = []

    for path in sorted(results_dir.iterdir()):
        if not (
            path.name.startswith("cifar_")
            and "_Tag[tune_cifar_noniid_" in path.name
            and path.name.endswith("_acc.npy")
        ):
            continue
        match = RESULT_RE.match(path.name)
        if not match:
            continue

        acc = np.load(path)
        loss_path = path.with_name(path.name.replace("_acc.npy", "_loss.npy"))
        loss = np.load(loss_path) if loss_path.exists() else None

        rows.append({
            "path": path,
            "model": match.group("model"),
            "epochs": int(match.group("epochs")),
            "local_ep": int(match.group("local_ep")),
            "local_bs": int(match.group("local_bs")),
            "lr": decode_float(match.group("lr")),
            "decay": decode_float(match.group("decay")),
            "crop": match.group("crop") or "-",
            "normalize": match.group("normalize") or "-",
            "final_acc": float(acc[-1]) * 100,
            "best_acc": float(np.max(acc)) * 100,
            "final_loss": float(loss[-1]) if loss is not None and len(loss) else None,
        })

    if not rows:
        print("No CIFAR non-IID baseline sweep results found.")
        return

    rows.sort(key=lambda row: (row["best_acc"], row["final_acc"]), reverse=True)

    print("| Rank | Model | Crop | Norm | LR | Decay | Batch | Local Ep | Epochs | Final Acc | Best Acc | Final Loss |")
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rank, row in enumerate(rows, 1):
        final_loss = "-" if row["final_loss"] is None else f"{row['final_loss']:.4f}"
        print(
            f"| {rank} | {row['model']} | {row['crop']} | {row['normalize']} | "
            f"{row['lr']} | {row['decay']} | {row['local_bs']} | "
            f"{row['local_ep']} | {row['epochs']} | {row['final_acc']:.2f}% | "
            f"{row['best_acc']:.2f}% | {final_loss} |"
        )


if __name__ == "__main__":
    main()
