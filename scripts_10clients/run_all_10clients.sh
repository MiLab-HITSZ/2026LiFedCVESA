#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_GAMMA="${RUN_GAMMA:-1}"
RUN_NUM_STEAL="${RUN_NUM_STEAL:-1}"
RUN_POSITION="${RUN_POSITION:-1}"

run_script() {
    script_path="$1"
    echo "==> Running ${script_path}"
    bash "${SCRIPT_DIR}/${script_path}"
}

if [ "${RUN_GAMMA}" = "1" ]; then
    run_script "gamma_sweep/mnist_cnn_gamma.sh"
    run_script "gamma_sweep/fmnist_cnn_gamma.sh"
    run_script "gamma_sweep/cifar_cnn_gamma.sh"
    run_script "gamma_sweep/cifar_resnet18_gamma.sh"
fi

if [ "${RUN_NUM_STEAL}" = "1" ]; then
    run_script "num_steal_sweep/mnist_cnn_num_steal.sh"
    run_script "num_steal_sweep/fmnist_cnn_num_steal.sh"
    run_script "num_steal_sweep/cifar_cnn_num_steal.sh"
fi

if [ "${RUN_POSITION}" = "1" ]; then
    run_script "position_ablation/mnist_cnn_position.sh"
    run_script "position_ablation/fmnist_cnn_position.sh"
    run_script "position_ablation/cifar_cnn_position.sh"
fi
