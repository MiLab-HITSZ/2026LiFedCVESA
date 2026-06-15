#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

GPU="${GPU:-2}"

for gama in 0 0.05 0.2 0.5 1.0; do
    run_fed \
        --model=cnn \
        --dataset=mnist \
        --gpu="${GPU}" \
        --lr=0.01 \
        --local_bs=16 \
        --local_ep=10 \
        --gama="${gama}" \
        --num_steal=5 \
        --num_img_per_client=1 \
        --agg_mode=segmented \
        --attack_position_mode=spread
done
