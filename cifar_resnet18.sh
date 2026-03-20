python src/federated_main.py \
    --model=resnet18 \
    --dataset=cifar \
    --gpu=1 \
    --iid=1 \
    --epochs=200 \
    --lr=0.1 \
    --local_bs=50 \
    --local_ep=5 \
    --gama=1 \
    --gama_warmup_epochs=100

python src/federated_main.py \
    --model=resnet18 \
    --dataset=cifar \
    --gpu=1 \
    --iid=1 \
    --epochs=200 \
    --lr=0.1 \
    --local_bs=50 \
    --local_ep=5 \
    --gama=0.5 \
    --gama_warmup_epochs=100
