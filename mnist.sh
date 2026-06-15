#!/usr/bin/env sh

set -eu

GPU="${GPU:-2}"

python src/federated_main.py \
    --model=cnn \
    --dataset=mnist \
    --gpu="${GPU}" \
    --iid=1 \
    --epochs=200 \
    --lr=0.01 \
    --local_bs=10 \
    --local_ep=10 \
    --gama=0 \
    --gama_warmup_epochs=100 \
    --num_steal=5 \
    --num_img_per_client=1

python src/federated_main.py \
    --model=cnn \
    --dataset=mnist \
    --gpu="${GPU}" \
    --iid=1 \
    --epochs=200 \
    --lr=0.01 \
    --local_bs=10 \
    --local_ep=10 \
    --gama=0.05 \
    --gama_warmup_epochs=100 \
    --num_steal=5 \
    --num_img_per_client=1

python src/federated_main.py \
    --model=cnn \
    --dataset=mnist \
    --gpu="${GPU}" \
    --iid=1 \
    --epochs=200 \
    --lr=0.01 \
    --local_bs=10 \
    --local_ep=10 \
    --gama=0.2 \
    --gama_warmup_epochs=100 \
    --num_steal=5 \
    --num_img_per_client=1

python src/federated_main.py \
    --model=cnn \
    --dataset=mnist \
    --gpu="${GPU}" \
    --iid=1 \
    --epochs=200 \
    --lr=0.01 \
    --local_bs=10 \
    --local_ep=10 \
    --gama=0.5 \
    --gama_warmup_epochs=100 \
    --num_steal=5 \
    --num_img_per_client=1

python src/federated_main.py \
    --model=cnn \
    --dataset=mnist \
    --gpu="${GPU}" \
    --iid=1 \
    --epochs=200 \
    --lr=0.01 \
    --local_bs=10 \
    --local_ep=10 \
    --gama=1.0 \
    --gama_warmup_epochs=100 \
    --num_steal=5 \
    --num_img_per_client=1
