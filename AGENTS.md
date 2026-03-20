# Repository Guidelines

## Project Structure & Module Organization
Core experiment code lives in `src/`. Use `federated_main.py` for federated runs, `baseline_main.py` for non-federated baselines, `models.py` for network definitions, `update.py` and `sampling.py` for client training and data partitioning, and `attack_utils.py` / `plot.py` for attack logic and visualization. Dataset files are stored under `data/`. Generated artifacts are written under `save/results`, `save/plots`, and `save/objects`; treat these as experiment outputs, not hand-edited source.

## Build, Test, and Development Commands
Install dependencies from the repo root with `pip install -r requirments.txt`. Run a baseline check with `python src/baseline_main.py --model=mlp --dataset=mnist --epochs=10`. Run a federated experiment with `python src/federated_main.py --model=cnn --dataset=cifar --iid=1 --epochs=10`. The shell scripts `mnist.sh`, `fashion_mnist.sh`, `cifar.sh`, and `cifar_resnet18.sh` capture longer experiment presets. For a quick syntax sanity check before opening a PR, use `python -m compileall src`.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, module-level imports, and `snake_case` for functions, variables, and CLI flags. Keep new options in `src/options.py` and thread them through the main scripts explicitly. Match the current file naming pattern for experiment helpers and outputs, for example `cifar_resnet18.sh` or `save/results/..._Gama[0.5]_acc.npy`. No formatter or linter is configured here, so keep changes small and consistent with adjacent code.

## Testing Guidelines
There is no dedicated `tests/` directory yet. Validate changes by running the smallest relevant training command and confirming that metrics or plots are regenerated in `save/`. When changing argument parsing, sampling, or aggregation logic, include the exact command you used and note dataset, model, `--epochs`, and attack-related flags such as `--gama` or `--num_steal`.

## Commit & Pull Request Guidelines
Recent commits use short, informal subjects such as `revise detail` and often append an emoji. Keep commit titles brief and focused on one change. Pull requests should describe the research or code change, list the reproduction command, call out any new outputs under `save/`, and include plots or screenshots when visualization behavior changes.

## Artifact & Data Hygiene
This repository already contains large generated files. Avoid committing fresh dataset downloads, `__pycache__` contents, or bulk result files unless they are intentionally part of the change.
