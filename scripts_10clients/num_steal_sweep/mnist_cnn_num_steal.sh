#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

GPU="${GPU:-3}"

for num_steal in 1 2 3 4 5 10; do
    run_fed \
        --model=cnn \
        --dataset=mnist \
        --gpu="${GPU}" \
        --lr=0.01 \
        --local_bs=16 \
        --local_ep=10 \
        --gama=0.5 \
        --num_steal="${num_steal}" \
        --num_img_per_client=1 \
        --agg_mode=segmented \
        --attack_position_mode=spread
done
