python src/federated_main.py \
    --model=mlp \
    --dataset=mnist \
    --gpu=1 \
    --iid=1 \
    --epochs=210 \
    --lr=0.01 \
    --local_bs=10 \
    --local_ep=10 \
    --gama=0.2