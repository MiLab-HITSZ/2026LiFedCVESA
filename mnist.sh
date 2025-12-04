python src/federated_main.py \
    --model=cnn \
    --dataset=mnist \
    --gpu=1 \
    --iid=1 \
    --epochs=500 \
    --lr=0.01 \
    --local_bs=50 \
    --local_ep=5 \
    --gama=0.5