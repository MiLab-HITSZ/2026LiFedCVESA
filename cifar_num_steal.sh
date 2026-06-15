#!/usr/bin/env sh

# set -eu

GPU="${GPU:-0}"

for num_steal in 1 2 3 4 5 10; do
    for gama in 0.5; do
        python src/federated_main.py \
            --model=cnn \
            --dataset=cifar \
            --gpu="${GPU}" \
            --iid=1 \
            --epochs=200 \
            --lr=0.15 \
            --local_bs=50 \
            --local_ep=5 \
            --gama="${gama}" \
            --gama_warmup_epochs=100 \
            --num_steal="${num_steal}" \
            --num_img_per_client=1
    done
done
