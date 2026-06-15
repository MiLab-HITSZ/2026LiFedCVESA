#!/usr/bin/env sh

# Ablation: attack target parameter position selection.

GPU="${GPU:-0}"
NUM_STEAL=10
NUM_IMG_PER_CLIENT=1
FRAC=0.1
GAMA=0.5
SEED=1

for attack_position_mode in front spread; do
    python src/federated_main.py \
        --model=cnn \
        --dataset=cifar \
        --gpu="${GPU}" \
        --iid=1 \
        --frac="${FRAC}" \
        --epochs=200 \
        --lr=0.15 \
        --local_bs=50 \
        --local_ep=5 \
        --gama="${GAMA}" \
        --gama_warmup_epochs=100 \
        --num_steal="${NUM_STEAL}" \
        --num_img_per_client="${NUM_IMG_PER_CLIENT}" \
        --agg_mode=segmented \
        --attack_position_mode="${attack_position_mode}" \
        --seed="${SEED}"
done
