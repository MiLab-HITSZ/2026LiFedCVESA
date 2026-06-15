#!/usr/bin/env sh

# set -eu

GPU="${GPU:-2}"

for num_steal in 1 2 3 4 5 10; do
    for gama in 0.5; do
        python src/federated_main.py \
            --model=cnn \
            --dataset=fmnist \
            --gpu="${GPU}" \
            --iid=1 \
            --epochs=200 \
            --lr=0.01 \
            --local_bs=10 \
            --local_ep=10 \
            --gama="${gama}" \
            --gama_warmup_epochs=100 \
            --num_steal="${num_steal}" \
            --num_img_per_client=1
    done
done
