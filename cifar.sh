python src/federated_main.py \
    --model=cnn \
    --dataset=cifar \
    --gpu=0 \
    --iid=1 \
    --epochs=200 \
    --lr=0.15 \
    --local_bs=50 \
    --local_ep=5 \
    --gama=1 \
    --gama_warmup_epochs=100 \
    --num_steal=5 \
    --num_img_per_client=2

python src/federated_main.py \
    --model=cnn \
    --dataset=cifar \
    --gpu=0 \
    --iid=1 \
    --epochs=200 \
    --lr=0.15 \
    --local_bs=50 \
    --local_ep=5 \
    --gama=1 \
    --gama_warmup_epochs=100 \
    --num_steal=10 \
    --num_img_per_client=1
