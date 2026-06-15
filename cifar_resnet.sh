#!/usr/bin/env sh

# set -eu

GPU="${GPU:-7}"

python src/federated_main.py \
    --model=resnet18 \
    --dataset=cifar \
    --gpu="${GPU}" \
    --iid=1 \
    --epochs=200 \
    --lr=0.05 \
    --local_bs=50 \
    --local_ep=5 \
    --gama=0 \
    --gama_warmup_epochs=100 \
    --num_steal=5 \
    --num_img_per_client=1

python src/federated_main.py \
    --model=resnet18 \
    --dataset=cifar \
    --gpu="${GPU}" \
    --iid=1 \
    --epochs=200 \
    --lr=0.05 \
    --local_bs=50 \
    --local_ep=5 \
    --gama=0.05 \
    --gama_warmup_epochs=100 \
    --num_steal=5 \
    --num_img_per_client=1

python src/federated_main.py \
    --model=resnet18 \
    --dataset=cifar \
    --gpu="${GPU}" \
    --iid=1 \
    --epochs=200 \
    --lr=0.05 \
    --local_bs=50 \
    --local_ep=5 \
    --gama=0.2 \
    --gama_warmup_epochs=100 \
    --num_steal=5 \
    --num_img_per_client=1

python src/federated_main.py \
    --model=resnet18 \
    --dataset=cifar \
    --gpu="${GPU}" \
    --iid=1 \
    --epochs=200 \
    --lr=0.05 \
    --local_bs=50 \
    --local_ep=5 \
    --gama=0.5 \
    --gama_warmup_epochs=100 \
    --num_steal=5 \
    --num_img_per_client=1

python src/federated_main.py \
    --model=resnet18 \
    --dataset=cifar \
    --gpu="${GPU}" \
    --iid=1 \
    --epochs=200 \
    --lr=0.05 \
    --local_bs=50 \
    --local_ep=5 \
    --gama=1.0 \
    --gama_warmup_epochs=100 \
    --num_steal=5 \
    --num_img_per_client=1
