# FedCVESA

本仓库是一个基于 PyTorch 的联邦学习实验项目，用于研究 FedCVESA：在白盒恶意服务器假设下，通过 Correlation Value Encoding Attack (CVEA) 和分段聚合实现的 Taking Away Training Data 研究原型。

当前实现参考 `FedCVESA.pdf` 中的方法流程：保留标准联邦学习训练框架，同时在目标客户端本地训练目标中加入主动记忆项，并在服务器聚合阶段保护选定的模型参数载体位置，最后从全局模型参数中恢复目标客户端训练图像。

## 项目流程

完整代码级流程见 `PROJECT_FLOW.md`。简要来说，主入口是 `src/federated_main.py`：

1. 从 `src/options.py` 读取实验参数。
2. 加载训练用标准化数据和攻击评估用 raw/cropped 数据。
3. 选择前 `--num_steal` 个客户端作为目标客户端，并从每个目标客户端准备 `--num_img_per_client` 张私有图像作为编码数据。
4. 执行同步联邦训练。
5. 在目标客户端本地 loss 中加入 CVEA 相关性项：`loss = classification_loss - gama * |corr(model_params, stolen_data)|`。
6. 使用标准 FedAvg 或 segmented aggregation 聚合本地模型。
7. 从全局模型载体参数位置恢复图像，并保存指标、曲线和恢复对比图。

## 目录结构

- `src/federated_main.py`：联邦实验主入口。
- `src/baseline_main.py`：非联邦 baseline 训练入口。
- `src/options.py`：联邦、模型、攻击、聚合和恢复相关 CLI 参数。
- `src/update.py`：客户端本地训练和推理，包含分类损失与 CVEA 攻击损失。
- `src/attack_utils.py`：目标数据准备、相关性攻击损失、MAPE 计算和图像恢复。
- `src/utils.py`：数据加载、客户端划分、FedAvg、分段聚合和实验信息输出。
- `src/models.py`：MLP、CNN、ResNet 等模型定义。
- `src/plot.py` 和 `src/plot_*.py`：曲线、恢复图和论文图绘制脚本。
- `data/`：torchvision 数据集目录。
- `save/results/`：生成的 `.npy` 指标数组。
- `save/plots/`：生成的训练曲线和恢复图。
- `save/objects/`：生成的 pickle 训练对象。
- `PROJECT_FLOW.md`：当前项目主流程总结。
- `experiment_summary.md`：当前 shell 脚本实验矩阵总结。
- `experiment_results_summary.md`：当前保存结果总结。
- `FedCVESA.pdf`：方法和实验论文草稿。

`save/` 下文件通常是实验产物，不建议手工修改或随意提交新的大批量输出。

## 安装依赖

在仓库根目录安装依赖：

```bash
pip install -r requirments.txt
```

注意：本仓库中的依赖文件名就是 `requirments.txt`。当前记录的历史环境包括 Python 3.7.3、PyTorch 1.2.0、torchvision 0.4.0、NumPy 1.15.4、tensorboardX 1.4 和 matplotlib 3.0.1。

## 运行实验

语法检查：

```bash
python -m compileall src
```

非联邦 baseline：

```bash
python src/baseline_main.py --model=mlp --dataset=mnist --epochs=10
```

无攻击联邦 baseline：

```bash
python src/federated_main.py --model=cnn --dataset=mnist --iid=1 --epochs=10 --gama=0
```

FedCVESA 攻击示例：

```bash
python src/federated_main.py \
  --model=cnn \
  --dataset=cifar \
  --gpu=0 \
  --iid=1 \
  --epochs=200 \
  --local_ep=5 \
  --local_bs=50 \
  --lr=0.15 \
  --gama=0.5 \
  --gama_warmup_epochs=100 \
  --num_steal=5 \
  --num_img_per_client=1 \
  --agg_mode=segmented \
  --attack_position_mode=spread
```

较完整的实验预设脚本包括：

- `mnist.sh`
- `fashion_mnist.sh`
- `cifar.sh`
- `cifar_resnet.sh`
- `mnist_num_steal.sh`
- `fashion_mnist_num_steal.sh`
- `cifar_num_steal.sh`
- `mnist_seg_agg_ablation.sh`
- `fashion_mnist_seg_agg_ablation.sh`
- `cifar_seg_agg_ablation.sh`

多数脚本支持用环境变量覆盖 GPU：

```bash
GPU=3 sh mnist.sh
```

## 关键参数

联邦和模型参数：

- `--dataset`：`mnist`、`fmnist` 或 `cifar`。
- `--model`：`mlp`、`cnn`，CIFAR 和 Fashion-MNIST 还支持 `resnet18`。
- `--epochs`：全局通信轮数。
- `--num_users`：客户端总数。
- `--frac`：每轮参与训练的客户端比例。
- `--local_ep`：每个客户端本地训练 epoch 数。
- `--local_bs`：本地 batch size。
- `--lr`：学习率。
- `--lr_decay`：每轮学习率衰减系数。
- `--iid`：`1` 表示 IID 划分，`0` 表示 non-IID 划分。
- `--seed`：随机种子。

攻击和聚合参数：

- `--gama`：CVEA 攻击强度；设为 `0` 表示关闭攻击。
- `--gama_warmup_epochs`：攻击强度 warm-up 轮数。默认 `100` 时，前半段 `gama=0`，后半段按余弦曲线增长到目标值，之后固定目标值。
- `--num_steal`：目标客户端数量。当前实现默认攻击前 `num_steal` 个客户端，并在 `gama > 0` 时强制它们每轮参与。
- `--num_img_per_client`：每个目标客户端编码的图像数量。
- `--agg_mode`：`segmented`、`avg`、`segmented_soft` 或 `target_only_avg`。
- `--seg_alpha`：`segmented_soft` 的混合系数。
- `--attack_position_mode`：`spread` 表示在展平模型参数中分散选择载体位置；`front` 表示使用前 N 个载体位置。

## 输出文件

每次联邦运行会按参数生成结果文件名后缀，主要输出包括：

- `save/results/*_acc.npy`：每轮平均训练准确率。
- `save/results/*_loss.npy`：每轮平均训练 loss。
- `save/results/*_mape.npy`：攻击生效时的恢复误差曲线。
- `save/objects/*.pkl`：原始训练 loss 和 accuracy 列表。
- `save/plots/*_acc.png`：准确率曲线。
- `save/plots/*_loss.png`：loss 曲线。
- `save/plots/*_mape.png`：MAPE 曲线。
- `save/plots/*_final_comparison.png`：最终原图与恢复图对比。
- `save/plots/epoch_recovery/`：攻击运行中每 10 轮保存一次的恢复对比图。

## 说明

本项目是受控研究原型，用于分析联邦学习中模型参数级记忆通道带来的隐私风险。当前攻击假设服务器是白盒恶意方，能够修改目标客户端训练目标、定制聚合逻辑、读取全局模型参数并在训练后执行恢复流程。
