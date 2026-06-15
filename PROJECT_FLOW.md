# FedCVESA 项目主要流程

本文档根据当前目录下的 `src/federated_main.py` 和 `FedCVESA.pdf` 整理，重点说明本项目从联邦训练、CVEA 编码、分段聚合到图像恢复与结果保存的主流程。

## 1. 项目目标

本项目研究联邦学习中的白盒 Taking Away Training Data 攻击。核心思想是：恶意服务器不只是被动观察梯度或参数，而是在目标客户端本地训练目标中加入 CVEA 相关性编码项，使目标客户端的私有图像逐步写入全局模型的指定参数位置。服务器端再通过分段聚合保护这些参数位置，训练结束后从全局模型参数中恢复目标图像。

论文 `FedCVESA.pdf` 将该方法概括为三部分：

1. Local Correlation Encoding：目标客户端在分类损失之外最大化模型参数与目标数据向量的相关性。
2. Segmented Aggregation over Dispersed Carrier Positions：服务器在聚合时保护被选中的载体参数位置，减少普通 FedAvg 对编码数据的覆盖。
3. Server-Side Extraction and Recovery：服务器按同一索引规则从全局模型取出载体位置，归一化并 reshape 成图像。

## 2. 入口与整体控制流

当前联邦实验入口是：

```bash
python src/federated_main.py
```

启动后主流程如下：

1. 调用 `args_parser()` 读取 `src/options.py` 中的参数。
2. 设置 TensorBoard logger、随机种子、GPU/CPU 设备。
3. 根据 `--gama` 和 `--gama_warmup_epochs` 配置攻击强度 warm-up。
4. 加载训练用数据集和攻击评估用 raw 数据集。
5. 构建全局模型。
6. 准备 CVEA 要编码的目标图像向量 `stolen_data_dm`。
7. 进入全局通信轮次循环，执行客户端采样、本地训练、服务器聚合、评估和恢复。
8. 训练结束后保存 `.npy` 指标、`.pkl` 对象、曲线图和最终恢复图。

## 3. 数据加载与目标图像准备

`src/federated_main.py` 会同时加载两套数据：

- `get_dataset(args)`：用于正常训练的增强或标准化数据集。
- `get_raw_dataset(args)`：用于提取目标客户端原始图像和做恢复对比的数据集。

MNIST 和 Fashion-MNIST 使用灰度图；CIFAR-10 在当前实现中被裁剪到 `24 x 24`，恢复与评估时会转成灰度图。因此 CIFAR 实验应理解为简化的灰度 `24 x 24` 设置。

目标数据来自前 `--num_steal` 个客户端，每个目标客户端取 `--num_img_per_client` 张图。`get_ordered_target_images_np()` 会按客户端顺序提取原图，用于后续 MAPE 和恢复图对比。`prepare_cvea_stolen_data()` 会把这些目标图像转为灰度、展平、归一化并中心化，得到攻击向量 `d_m`。

目标向量长度为：

```text
num_steal * num_img_per_client * attack_h * attack_w
```

其中 `attack_h` 和 `attack_w` 由目标图像尺寸确定。

## 4. 模型构建

模型由 `--dataset` 和 `--model` 决定：

- `--model=cnn`：
  - MNIST / Fashion-MNIST 使用 `CNNFashion_Enhanced`。
  - CIFAR-10 使用 `CNNCifar_Enhanced_V3`。
- `--model=resnet18`：
  - CIFAR-10 使用 `CNNCifar_ResNet_V1`。
  - Fashion-MNIST 使用 `CNNFashion_ResNet18`。
- `--model=mlp`：
  - 使用 `MLP`，主要保留给基线或简单实验。

模型初始化后会复制初始 `state_dict()` 作为全局权重，并把 `args.device` 写入参数对象，供本地训练和攻击工具使用。

## 5. CVEA 本地相关性编码

本地训练由 `src/update.py` 中的 `LocalUpdate.update_weights()` 执行。普通客户端只优化分类损失；目标客户端会额外加入 CVEA 攻击项：

```text
loss = loss_ce - gama * |corr(theta, d_m)|
```

其中：

- `loss_ce` 是分类任务的 NLL loss。
- `gama` 是攻击强度，由 `--gama` 控制。
- `theta` 是从模型展平参数中选出的载体位置向量。
- `d_m` 是目标图像展平、归一化、中心化后的数据向量。
- `corr()` 是 Pearson 风格相关系数，由 `attack_utils.cor_attack()` 计算。

因为训练过程最小化 loss，所以 `-gama * |corr|` 会推动模型参数与目标图像向量更相关，从而把目标图像信息写入模型参数。

## 6. Gama Warm-up 策略

当 `--gama > 0` 且 `--gama_warmup_epochs > 0` 时，`src/federated_main.py` 会动态更新当前轮次的 `args.gama`：

| 阶段 | 轮次范围 | 行为 |
| --- | --- | --- |
| Phase 1 | warm-up 前半段 | `gama=0`，只做分类训练 |
| Phase 2 | warm-up 后半段 | `gama` 按余弦曲线从 0 增长到目标值 |
| Phase 3 | warm-up 结束后 | 固定使用目标 `gama` |

默认 `--gama_warmup_epochs=100` 时，前 50 轮不攻击，第 51-100 轮逐渐加大攻击强度，第 101 轮开始固定为目标 `gama`。

MAPE 也从 warm-up 后半段开始记录，避免在纯分类训练阶段记录没有意义的恢复误差。

## 7. 客户端选择

无攻击时，代码按标准 FedAvg 方式随机选择客户端：

```text
m = max(int(frac * num_users), 1)
```

有攻击时，前 `num_steal` 个目标客户端每轮都被强制选中，其余名额从非目标客户端中随机补齐。这样做是为了保证目标数据能持续写入载体参数位置，避免目标客户端长时间不参与导致编码不稳定。

每个目标客户端在 `LocalUpdate` 初始化时会拿到 `stolen_data_dm`；非目标客户端拿到 `None`，因此不会执行攻击损失。

## 8. 载体参数位置选择

`attack_position_mode` 控制目标数据写入哪些模型参数位置：

- `front`：使用展平模型参数的前 `N` 个位置。
- `spread`：在完整展平模型参数上均匀采样 `N` 个位置。

这里的 `N` 等于目标向量长度。论文中强调 `spread` 的动机是避免把大量编码负担集中到单个层或连续参数块，从而减少对主任务表示的破坏。

同一套位置选择规则必须在以下位置保持一致：

- `attack_utils.cor_attack()`
- `utils.segmented_average_weights()`
- `attack_utils.calculate_cor_mape()`
- `attack_utils.recover_cor_stolen_data_new()`

如果这些位置不一致，训练写入、聚合保护、误差计算和恢复读取就会错位。

## 9. 服务器聚合

每轮本地训练结束后，`src/federated_main.py` 先保存上一轮全局权重，然后根据 `--agg_mode` 选择聚合方式：

- `avg`：标准 FedAvg，所有参数位置都做普通平均。
- `segmented`：非载体位置做普通平均；载体位置按目标客户端分段，用对应目标客户端的本地值覆盖全局平均值。
- `segmented_soft`：载体位置不完全覆盖，而是用 `seg_alpha * target_value + (1 - seg_alpha) * avg_value` 混合。
- `target_only_avg`：载体位置只对参与本轮的目标客户端取平均。

当前默认是 `segmented`。它对应论文中的 segmented aggregation：在不改变非载体位置 FedAvg 路径的前提下，保护目标客户端负责的载体片段，减少其他客户端更新对编码数据的覆盖。

## 10. 训练过程评估

每轮聚合后代码会：

1. 把聚合后的 `global_weights` 加载回全局模型。
2. 衰减学习率：`args.lr *= args.lr_decay`。
3. 记录平均本地训练 loss。
4. 在所有客户端上计算平均训练准确率和 loss。
5. 当攻击有效并达到 MAPE 记录阶段时，调用 `calculate_cor_mape()` 计算恢复误差。
6. 每 10 轮恢复一次图像并保存中间对比图。

MAPE 计算时会从当前全局模型中取出载体参数位置，按每张图像的像素数切分，做动态 min-max 归一化，并同时比较正向图像和反色图像，取误差较小者。

## 11. 服务器端恢复

训练结束后，如果 `stolen_data_dm` 存在且 `gama > 0`，代码会执行最终恢复：

```text
recover_cor_stolen_data_new(global_model, x_train_gray_np, ...)
```

恢复过程为：

1. 展平全局模型所有参数。
2. 按 `attack_position_mode` 重新取出载体位置。
3. 按 `num_steal * num_img_per_client` 和图像尺寸切成每张图像对应的片段。
4. 对每个片段做 min-max 归一化。
5. 映射到 `[0, 255]` 并 reshape 成灰度图。
6. 与原始目标图像并排绘制，必要时用反色版本比较误差。

最终图像保存到 `save/plots/*_final_comparison.png`，中间轮次图像保存到 `save/plots/epoch_recovery/`。

## 12. 输出文件

一次联邦运行会根据数据集、模型、轮数、联邦参数、攻击参数和聚合参数生成带后缀的文件名。主要输出包括：

| 输出 | 位置 | 含义 |
| --- | --- | --- |
| `*_acc.npy` | `save/results/` | 每轮全局训练准确率 |
| `*_loss.npy` | `save/results/` | 每轮全局训练 loss |
| `*_mape.npy` | `save/results/` | 攻击恢复误差曲线 |
| `*.pkl` | `save/objects/` | 原始 `train_loss` 和 `train_accuracy` |
| `*_acc.png` | `save/plots/` | 准确率曲线 |
| `*_loss.png` | `save/plots/` | loss 曲线 |
| `*_mape.png` | `save/plots/` | MAPE 曲线 |
| `*_final_comparison.png` | `save/plots/` | 最终原图与恢复图对比 |
| `epoch_*_recovery_*.png` | `save/plots/epoch_recovery/` | 中间轮次恢复图 |

## 13. 典型运行命令

无攻击联邦基线：

```bash
python src/federated_main.py --model=cnn --dataset=mnist --iid=1 --epochs=10 --gama=0
```

MNIST 上运行一个简短 FedCVESA 检查：

```bash
python src/federated_main.py \
  --model=cnn \
  --dataset=mnist \
  --iid=1 \
  --epochs=10 \
  --gama=0.5 \
  --num_steal=5 \
  --num_img_per_client=1 \
  --agg_mode=segmented \
  --attack_position_mode=spread
```

CIFAR-10 灰度 `24 x 24` 设置下运行论文风格配置：

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

## 14. 与论文方法的对应关系

| 论文概念 | 当前代码位置 |
| --- | --- |
| 白盒恶意服务器威胁模型 | `federated_main.py` 中的目标客户端强制参与、服务器自定义聚合和最终恢复 |
| Local Correlation Encoding | `update.py::LocalUpdate.update_weights()` 和 `attack_utils.cor_attack()` |
| 目标数据向量 `d_m` | `attack_utils.prepare_cvea_stolen_data()` |
| Dispersed Carrier Positions | `attack_utils.get_attack_param_indices(..., mode='spread')` |
| Segmented Aggregation | `utils.segmented_average_weights(..., mode='segmented')` |
| Server-Side Extraction and Recovery | `attack_utils.recover_cor_stolen_data_new()` |
| Utility 指标 | `test_inference()`、每轮平均训练准确率和 loss |
| Stealing 指标 | `attack_utils.calculate_cor_mape()` 和恢复图 |

## 15. 注意事项

- 本项目是受控研究验证代码，默认假设服务器是白盒恶意方，能够修改目标客户端本地目标函数、控制聚合并读取全局模型参数。
- 当前实现用 `gama` 作为 CLI 参数名，文档中保留这个拼写以匹配代码。
- `save/` 下文件通常是实验产物，不应手工修改。
- 修改攻击位置选择、聚合逻辑或恢复逻辑时，必须同步检查编码、聚合、MAPE 和恢复四处的索引规则。
