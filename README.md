# FedCVESA

This repository contains the slim code release for the FedCVESA v5 experiments used in the paper draft. The retained experiment setting is the 10-client Dirichlet non-IID split with alpha=0.5 on MNIST, Fashion-MNIST, and CIFAR-10.

## What Is Included

- `src/federated_main.py`: main federated training and CVEA recovery entrypoint.
- `src/models.py`: only the v5 paper models:
  - `CNNFashion_Enhanced` for MNIST and Fashion-MNIST.
  - `ResNet18Cifar` for CIFAR-10.
- `src/options.py`: CLI options used by the v5 experiments.
- `src/update.py`, `src/utils.py`, `src/sampling.py`, `src/attack_utils.py`, `src/plot.py`: client training, partitioning, aggregation, attack, and plotting helpers.
- `scripts_10clients/run_all_rank10_dirichlet05.sh`: final v5 reproduction script.
- `experiment_results_summary_v5.md`: final experiment table used for the paper.
- `experiment_results_summary_v5_model_summary.md`: model architecture and parameter summary.
- `figures/`: final paper figures only.

Datasets, logs, `.npy` metrics, checkpoints, generated recovery images, and historical ablation scripts are intentionally not included.

## Setup

Install dependencies from the repository root:

```bash
pip install -r requirments.txt
```

The dependency filename is kept as `requirments.txt` to match the original project. Datasets are downloaded by `torchvision` into `data/` when an experiment first runs.

## Quick Checks

Syntax check:

```bash
python -m compileall src
```

Small CPU sanity run:

```bash
python src/federated_main.py \
  --dataset=mnist \
  --model=cnn \
  --epochs=1 \
  --num_users=10 \
  --frac=1.0 \
  --iid=0 \
  --noniid_mode=dirichlet \
  --dirichlet_alpha=0.5 \
  --dirichlet_min_size=100 \
  --gama=0.5 \
  --num_steal=5 \
  --num_img_per_client=1 \
  --agg_mode=segmented \
  --attack_position_mode=spread
```

## Reproduce v5

Preview the 57 jobs without running them:

```bash
DRY_RUN=1 bash scripts_10clients/run_all_rank10_dirichlet05.sh
```

Run all v5 jobs:

```bash
GPU_LIST="0 1 2 3 4 5 6 7" bash scripts_10clients/run_all_rank10_dirichlet05.sh
```

The script defaults to:

```text
--num_users=10 --frac=1.0 --iid=0
--noniid_mode=dirichlet --dirichlet_alpha=0.5 --dirichlet_min_size=100
--gama_warmup_epochs=0 --lr_scheduler=cosine --min_lr=0.0001
MNIST/Fashion-MNIST: model=cnn, lr=0.01, local_ep=10, local_bs=16
CIFAR-10: model=resnet18_cifar, lr=0.03, local_ep=1, local_bs=64, cifar_crop_size=32, cifar_normalize=1
```

Outputs are generated under `save/results`, `save/plots`, `save/objects`, and `scripts_10clients/logs_rank10_dirichlet05`; these paths are ignored by git.

## Paper Figures

After reproducing v5 metrics and recovery plots, regenerate the retained paper figures with:

```bash
python src/plot_rank10_v5_paper_figures.py
```

The checked-in `figures/` directory contains only the final paper images.
