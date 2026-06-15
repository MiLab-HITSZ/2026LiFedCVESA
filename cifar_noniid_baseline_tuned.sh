#!/usr/bin/env bash

set -eu

GPU="${GPU:-0}"
SEED="${SEED:-1}"
NUM_USERS="${NUM_USERS:-10}"
FRAC="${FRAC:-1.0}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-0.05}"
LR_DECAY="${LR_DECAY:-0.995}"
LOCAL_BS="${LOCAL_BS:-64}"
LOCAL_EP="${LOCAL_EP:-1}"

python src/federated_main.py \
    --model=cnn \
    --dataset=cifar \
    --gpu="${GPU}" \
    --seed="${SEED}" \
    --num_users="${NUM_USERS}" \
    --frac="${FRAC}" \
    --iid=0 \
    --epochs="${EPOCHS}" \
    --lr="${LR}" \
    --lr_decay="${LR_DECAY}" \
    --local_bs="${LOCAL_BS}" \
    --local_ep="${LOCAL_EP}" \
    --gama=0 \
    --gama_warmup_epochs=0 \
    --num_steal=5 \
    --num_img_per_client=1 \
    --agg_mode=avg \
    --attack_position_mode=spread
