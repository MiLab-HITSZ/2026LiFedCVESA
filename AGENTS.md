# Repository Guidelines

## Project Structure & Module Organization
Core experiment code lives in `src/`. Use `federated_main.py` for FedCVESA
federated runs, `baseline_main.py` for non-federated baselines, `models.py` for
network definitions, `update.py` and `sampling.py` for client training and data
partitioning, `utils.py` for dataset loading and aggregation, and
`attack_utils.py` / `plot.py` for CVEA attack logic, recovery metrics, and
visualization. The paper draft is `FedCVESA.pdf`. `PROJECT_FLOW.md` summarizes
the current end-to-end workflow; `experiment_summary.md` summarizes shell-script
experiment settings; `experiment_results_summary.md` summarizes saved results.
Dataset files are stored under `data/`. Generated artifacts are written under
`save/results`, `save/plots`, and `save/objects`; treat these as experiment
outputs, not hand-edited source.

## Build, Test, and Development Commands
Install dependencies from the repo root with `pip install -r requirments.txt`.
Run a baseline check with `python src/baseline_main.py --model=mlp --dataset=mnist --epochs=10`.
Run a no-attack federated baseline with `python src/federated_main.py --model=cnn --dataset=mnist --iid=1 --epochs=10 --gama=0`.
Run a FedCVESA attack sanity check with `python src/federated_main.py --model=cnn --dataset=mnist --iid=1 --epochs=10 --gama=0.5 --num_steal=5 --agg_mode=segmented`.
The shell scripts `mnist.sh`, `fashion_mnist.sh`, `cifar.sh`,
`cifar_resnet.sh`, and the `*_num_steal.sh` / `*_seg_agg_ablation.sh` scripts
capture longer experiment presets. For a quick syntax sanity check before
opening a PR, use `python -m compileall src`.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, module-level imports, and
`snake_case` for functions, variables, and CLI flags. Keep new options in
`src/options.py` and thread them through the main scripts explicitly. Keep the
FedCVESA naming convention consistent: `gama` is the existing CLI spelling,
`num_steal` controls target clients, `num_img_per_client` controls per-client
payload, `agg_mode` controls aggregation, and `attack_position_mode` controls
carrier-position selection. Match the current file naming pattern for
experiment helpers and outputs, for example `cifar_resnet.sh` or
`save/results/..._Gama[0.5]_numSteal[5]_AttackPos[spread]_acc.npy`. No
formatter or linter is configured here, so keep changes small and consistent
with adjacent code.

## FedCVESA Workflow Notes
The main attack flow is implemented in `src/federated_main.py`. It loads both
normalized training datasets and raw/cropped datasets, extracts target-client
images, prepares a centered CVEA data vector with `prepare_cvea_stolen_data`,
and passes that vector only to selected target clients. In `src/update.py`,
target clients optimize `classification_loss - gama * |correlation|`; non-target
clients train normally. When `gama > 0`, the first `num_steal` clients are forced
to participate in each round. Aggregation is selected through `agg_mode`:
standard `avg`, hard `segmented`, blended `segmented_soft`, or
`target_only_avg`. Carrier positions must stay consistent across
`cor_attack`, `segmented_average_weights`, MAPE calculation, and recovery.

## Testing Guidelines
There is no dedicated `tests/` directory yet. Validate code changes with
`python -m compileall src` plus the smallest relevant training command. When
changing argument parsing, sampling, local loss, aggregation, carrier-position
selection, or recovery logic, include the exact command used and note dataset,
model, `--epochs`, `--gama`, `--num_steal`, `--num_img_per_client`,
`--agg_mode`, and `--attack_position_mode`. For visualization changes, confirm
that the relevant files are regenerated under `save/plots` or
`save/plots/epoch_recovery`.

## Commit & Pull Request Guidelines
Recent commits use short, informal subjects such as `revise detail` and often
append an emoji. Keep commit titles brief and focused on one change. Pull
requests should describe the research or code change, list the reproduction
command, call out any new outputs under `save/`, and include plots or
screenshots when visualization behavior changes.

## Artifact & Data Hygiene
This repository already contains large generated files. Avoid committing fresh
dataset downloads, `__pycache__` contents, or bulk result files unless they are
intentionally part of the change. Before editing docs or code, check
`git status --short` and avoid reverting unrelated user changes in scripts,
saved results, or generated plots.
