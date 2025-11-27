python src/federated_main.py \
    --model=cnn \
    --dataset=cifar \
    --gpu=0 \
    --iid=1 \
    --epochs=500 \
    --lr=0.15 \
    --local_bs=50 \
    --local_ep=5 \
    --gama=1