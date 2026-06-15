# 当前实验设置与 Shell 脚本总结

本文档根据仓库根目录下当前所有 `.sh` 文件，以及 `src/options.py`、`src/federated_main.py` 中的默认设置整理。当前实验主体均为联邦学习场景下的 CVEA 攻击实验，入口统一为：

```bash
python src/federated_main.py
```

实验主要观察不同攻击强度、攻击目标数量、攻击参数位置选择和模型结构下，全局模型训练性能与图像恢复效果之间的关系。

## 一、公共实验设置

所有 `.sh` 脚本显式设置的公共参数如下：

| 参数 | 当前设置 |
| --- | --- |
| 数据划分 | `--iid=1`，即 IID 联邦划分 |
| 全局通信轮数 | `--epochs=200` |
| 攻击 warm-up | `--gama_warmup_epochs=100` |
| 联邦入口 | `src/federated_main.py` |
| 优化器 | 使用默认 `--optimizer=sgd` |
| 学习率衰减 | 使用默认 `--lr_decay=0.99` |

代码默认但多数脚本未显式覆盖的参数如下：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--num_users` | `100` | 联邦客户端总数 |
| `--frac` | `0.1` | 每轮参与训练的客户端比例 |
| `--seed` | `1` | 随机种子 |
| `--agg_mode` | `segmented` | 默认分段聚合方式 |
| `--seg_alpha` | `0.5` | `segmented_soft` 的默认混合系数 |
| `--attack_position_mode` | `spread` | 默认在全部展平模型参数上均匀分散选择攻击参数位置 |

GPU 通过脚本中的 `GPU="${GPU:-x}"` 设置默认值，运行时可以用环境变量覆盖，例如：

```bash
GPU=3 sh mnist.sh
```

## 二、CVEA 攻击与 warm-up 机制

当前攻击项由 `--gama` 控制。`gama=0` 表示无攻击基线；`gama>0` 时，本地训练 loss 中加入 CVEA 相关性攻击项：

```text
loss = classification_loss - gama * |correlation(model_params, stolen_data)|
```

当 `--gama_warmup_epochs=100` 且目标 `gama>0` 时，代码实际分为三段：

| 阶段 | 轮次范围 | 行为 |
| --- | --- | --- |
| Phase 1 | epoch 0-49 | `gama=0`，纯分类训练 |
| Phase 2 | epoch 50-99 | `gama` 按余弦曲线从 0 增长到目标值 |
| Phase 3 | epoch 100 以后 | 固定使用目标 `gama` |

攻击目标默认来自前 `num_steal` 个客户端，每个目标客户端取 `num_img_per_client` 张图像。当前主实验、`num_steal` 扫描和攻击参数位置消融中均为每个目标客户端取 1 张图像。

输出指标主要包括：

| 输出 | 保存位置 | 说明 |
| --- | --- | --- |
| 准确率曲线 | `save/results/*_acc.npy`、`save/plots/*_acc.png` | 每轮全局训练准确率 |
| loss 曲线 | `save/results/*_loss.npy`、`save/plots/*_loss.png` | 每轮全局训练 loss |
| MAPE 曲线 | `save/results/*_mape.npy`、`save/plots/*_mape.png` | 仅攻击生效时保存 |
| 恢复图像对比 | `save/plots/*_final_comparison.png` | 训练结束后的原图与恢复图对比 |
| 中间轮次恢复图 | `save/plots/epoch_recovery/` | 每 10 轮保存一次恢复效果 |

## 三、实验主线一：攻击强度 `gama` 扫描

该组实验固定 `num_steal=5`、`num_img_per_client=1`，扫描不同攻击强度：

```text
0, 0.05, 0.2, 0.5, 1.0
```

涉及脚本：

| 脚本 | 数据集 | 模型 | 学习率 | 本地 batch | 本地 epoch | 默认 GPU |
| --- | --- | --- | --- | --- | --- | --- |
| `mnist.sh` | `mnist` | `cnn` | `0.01` | `10` | `10` | `2` |
| `fashion_mnist.sh` | `fmnist` | `cnn` | `0.01` | `10` | `10` | `5` |
| `cifar.sh` | `cifar` | `cnn` | `0.15` | `50` | `5` | `0` |
| `cifar_resnet.sh` | `cifar` | `resnet18` | `0.05` | `50` | `5` | `7` |

实验目的：

- 比较无攻击基线与不同攻击强度下的模型准确率、loss 和恢复误差。
- 观察 `gama` 增大时攻击恢复效果与模型性能之间的权衡。
- 在 CIFAR 上额外比较 CNN 与 ResNet18 结构对攻击和训练的影响。

该组共 `4 * 5 = 20` 个配置。

## 四、实验主线二：攻击目标数量 `num_steal` 扫描

该组实验固定 `gama=0.5`、`num_img_per_client=1`，扫描被攻击客户端数量：

```text
1, 2, 3, 4, 5, 10
```

涉及脚本：

| 脚本 | 数据集 | 模型 | 学习率 | 本地 batch | 本地 epoch | 默认 GPU |
| --- | --- | --- | --- | --- | --- | --- |
| `mnist_num_steal.sh` | `mnist` | `cnn` | `0.01` | `10` | `10` | `3` |
| `fashion_mnist_num_steal.sh` | `fmnist` | `cnn` | `0.01` | `10` | `10` | `2` |
| `cifar_num_steal.sh` | `cifar` | `cnn` | `0.15` | `50` | `5` | `0` |

实验目的：

- 分析攻击目标客户端数量增加时，MAPE 和恢复图像质量如何变化。
- 观察攻击规模扩大是否影响全局模型准确率和 loss。
- 对比 MNIST、Fashion-MNIST、CIFAR 三个数据集上的攻击可扩展性。

该组共 `3 * 6 = 18` 个配置。

## 五、实验主线三：攻击参数位置选择消融

该组实验固定 `gama=0.5`、`num_steal=5`、`num_img_per_client=1`、`agg_mode=segmented`，比较 CVEA 攻击绑定到展平模型参数时的位置选择方式。该设置与攻击强度扫描中的默认攻击规模保持一致，用于观察攻击参数集中在模型前部，或分散到全模型参数中时，对模型效用和恢复误差的影响。

涉及攻击参数位置选择方式：

| 位置方式 | 参数设置 | 含义 |
| --- | --- | --- |
| front | `--attack_position_mode=front` | 取展平模型参数的前 `N` 个位置作为攻击参数 |
| spread | `--attack_position_mode=spread` | 在全部展平模型参数范围内均匀采样 `N` 个位置作为攻击参数 |

涉及脚本：

| 脚本 | 数据集 | 模型 | 学习率 | 本地 batch | 本地 epoch | 默认 GPU |
| --- | --- | --- | --- | --- | --- | --- |
| `mnist_seg_agg_ablation.sh` | `mnist` | `cnn` | `0.01` | `10` | `10` | `2` |
| `fashion_mnist_seg_agg_ablation.sh` | `fmnist` | `cnn` | `0.01` | `10` | `10` | `1` |
| `cifar_seg_agg_ablation.sh` | `cifar` | `cnn` | `0.15` | `50` | `5` | `0` |

实验目的：

- 验证攻击参数选择在模型前部或分散到全模型参数中，哪种方式更有利于图像恢复。
- 分析参数位置选择对全局模型准确率、loss 和 MAPE 的影响。
- 对比 MNIST、Fashion-MNIST、CIFAR 三个数据集上攻击位置敏感性的差异。

该组共 `3 * 2 = 6` 个配置。

## 六、实验矩阵规模

按当前 `.sh` 文件统计：

| 实验类别 | 脚本数量 | 每个脚本配置数 | 总配置数 |
| --- | --- | --- | --- |
| `gama` 攻击强度扫描 | 4 | 5 | 20 |
| `num_steal` 目标数量扫描 | 3 | 6 | 18 |
| 攻击参数位置消融 | 3 | 2 | 6 |
| 合计 | 10 | - | 44 |

数据集与模型覆盖如下：

| 数据集 | 模型 | 覆盖实验 |
| --- | --- | --- |
| MNIST | CNN | `gama` 扫描、`num_steal` 扫描、攻击参数位置消融 |
| Fashion-MNIST | CNN | `gama` 扫描、`num_steal` 扫描、攻击参数位置消融 |
| CIFAR-10 | CNN | `gama` 扫描、`num_steal` 扫描、攻击参数位置消融 |
| CIFAR-10 | ResNet18 | `gama` 扫描 |

## 七、可用于论文或后续分析的实验逻辑

当前实验设计可以概括为三条主线：

1. 攻击强度维度：固定 `num_steal=5`、每客户端 1 张目标图，比较 `gama=0/0.05/0.2/0.5/1.0`。
2. 攻击规模维度：固定 `gama=0.5`、每客户端 1 张目标图，比较 `num_steal=1/2/3/4/5/10`。
3. 攻击参数位置维度：固定 `gama=0.5`、`num_steal=5`、每客户端 1 张目标图和默认 `segmented` 聚合，比较 `front` 与 `spread`。

整体上，当前实验已经覆盖数据集、攻击强度、攻击目标数量、攻击参数位置选择和 CIFAR 模型结构五个维度，适合后续按准确率、loss、MAPE 和恢复图像质量进行横向总结。
