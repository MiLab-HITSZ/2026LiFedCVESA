python src/federated_main.py \
    --model=cnn \
    --dataset=fmnist \
    --gpu=4 \
    --iid=1 \
    --epochs=200 \
    --lr=0.01 \
    --local_bs=10 \
    --local_ep=10 \
    --gama=1 \
    --gama_warmup_epochs=100 \
    --num_steal=5 \
    --num_img_per_client=2

python src/federated_main.py \
    --model=cnn \
    --dataset=fmnist \
    --gpu=5 \
    --iid=1 \
    --epochs=200 \
    --lr=0.01 \
    --local_bs=10 \
    --local_ep=10 \
    --gama=0.5 \
    --gama_warmup_epochs=100 \
    --num_steal=5 \
    --num_img_per_client=2
