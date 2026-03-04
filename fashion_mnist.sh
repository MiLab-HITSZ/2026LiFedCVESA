python src/federated_main.py \
    --model=cnn \
    --dataset=fmnist \
    --gpu=3 \
    --iid=1 \
    --epochs=210 \
    --lr=0.01 \
    --local_bs=10 \
    --local_ep=10 \
    --gama=0.5