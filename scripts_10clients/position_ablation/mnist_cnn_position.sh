#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

GPU="${GPU:-2}"

for setting in "10 1" "10 5" "10 10" "10 50"; do
    read -r num_steal num_img_per_client <<< "${setting}"
    for attack_position_mode in front spread; do
        run_fed \
            --model=cnn \
            --dataset=mnist \
            --gpu="${GPU}" \
            --lr=0.01 \
            --local_bs=16 \
            --local_ep=10 \
            --gama=0.5 \
            --num_steal="${num_steal}" \
            --num_img_per_client="${num_img_per_client}" \
            --agg_mode=segmented \
            --attack_position_mode="${attack_position_mode}"
    done
done
