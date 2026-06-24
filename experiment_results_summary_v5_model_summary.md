# experiment_results_summary_v5.md 使用模型总结

本文档总结 `experiment_results_summary_v5.md` 中实际使用到的模型结构、参数量、激活函数、归一化方式和与实验配置相关的注意点。统计对象以 `src/federated_main.py` 的联邦实验模型选择逻辑为准，因为 v5 报告对应的是 `src/federated_main.py` 生成的联邦训练结果。

## 1. v5 报告中使用到的模型

`experiment_results_summary_v5.md` 覆盖 MNIST、Fashion-MNIST、CIFAR-10 三个数据集。在报告中的模型名和代码中实际实例化的模型类如下。

| 数据集 | 报告/脚本中的 `--model` | 实际模型类 | 输入尺寸 | 类别数 | 参数量 |
| --- | --- | --- | --- | --- | --- |
| MNIST | `cnn` | `CNNFashion_Enhanced` | `1 x 28 x 28` | 10 | 3,373,434 |
| Fashion-MNIST | `cnn` | `CNNFashion_Enhanced` | `1 x 28 x 28` | 10 | 3,373,434 |
| CIFAR-10 | `resnet18_cifar` | `ResNet18Cifar` | `3 x 32 x 32` in v5 | 10 | 11,173,962 |

需要注意：代码库中也保留了旧版 `CNNMnist`、`CNNFashion_Mnist`、`CNNCifar`、`CNNCifar_Enhanced_V3`、`CNNCifar_ResNet_V1`、`CNNFashion_ResNet18`、`WideResNetCifar` 等模型，但它们不是 `experiment_results_summary_v5.md` 最终统计表中使用的模型。尤其是 `--model=cnn` 在 `baseline_main.py` 和 `federated_main.py` 中映射不同；v5 联邦实验中，MNIST 与 Fashion-MNIST 的 `cnn` 都映射到 `CNNFashion_Enhanced`。

## 2. MNIST / Fashion-MNIST: `CNNFashion_Enhanced`

### 2.1 结构概览

`CNNFashion_Enhanced` 是一个两层卷积加三层全连接的轻量 CNN。它虽然命名为 Fashion 版本，但在 v5 的联邦实验中同时用于 MNIST 和 Fashion-MNIST。

| 阶段 | 层 | 输出形状 | 激活/归一化/池化 | 说明 |
| --- | --- | --- | --- | --- |
| Input | - | `1 x 28 x 28` | - | 灰度图输入 |
| Conv block 1 | `Conv2d(1, 80, kernel=5, padding=2, bias=False)` | `80 x 28 x 28` | `GroupNorm(8, 80)` + `LeakyReLU(0.1)` + `AvgPool2d(2, 2)` | 池化后为 `80 x 14 x 14` |
| Conv block 2 | `Conv2d(80, 128, kernel=3, padding=1)` | `128 x 14 x 14` | `GroupNorm(8, 128)` + `LeakyReLU(0.1)` + `AvgPool2d(2, 2)` | 池化后为 `128 x 7 x 7` |
| Flatten | - | `6272` | - | `128 * 7 * 7` |
| FC 1 | `Linear(6272, 512)` | `512` | `ReLU` + `Dropout(0.3)` | 最大参数来源 |
| FC 2 | `Linear(512, 128)` | `128` | `ReLU` | 使用函数式 `F.relu` |
| FC 3 | `Linear(128, 10)` | `10` | `log_softmax(dim=1)` | 输出 10 类 log-probability |

### 2.2 参数量分解

| 模块 | 参数量 | 备注 |
| --- | ---: | --- |
| `conv1.weight` | 2,000 | `80 * 1 * 5 * 5`，无 bias |
| `gn1.weight + gn1.bias` | 160 | `80 + 80` |
| `conv2.weight + conv2.bias` | 92,288 | `128 * 80 * 3 * 3 + 128` |
| `gn2.weight + gn2.bias` | 256 | `128 + 128` |
| `fc1.weight + fc1.bias` | 3,211,776 | `512 * 6272 + 512` |
| `fc2.weight + fc2.bias` | 65,664 | `128 * 512 + 128` |
| `fc3.weight + fc3.bias` | 1,290 | `10 * 128 + 10` |
| **合计** | **3,373,434** | 全部为可训练参数 |

### 2.3 激活函数与正则化

- 激活函数：前两个卷积块使用 `LeakyReLU(negative_slope=0.1)`，全连接头使用 `ReLU`。
- 归一化：使用 `GroupNorm`，不是 `BatchNorm`。第一层分 8 组归一化 80 通道，第二层分 8 组归一化 128 通道。
- 池化：两个卷积块均使用 `AvgPool2d(kernel_size=2, stride=2)`。
- Dropout：仅在 `fc1` 后使用 `Dropout(0.3)`。
- 输出：最后返回 `F.log_softmax(x, dim=1)`，即每一类的 log-probability。

### 2.4 与 CVEA / FedCVESA 实验的关系

- 第一层卷积设置为 `bias=False`，代码注释说明这是为了保留更适合梯度泄露攻击的早期梯度结构。
- v5 中 MNIST 与 Fashion-MNIST 都是 `1 x 28 x 28` 输入，因此该模型的固定 flatten 维度 `6272` 与两者兼容。
- 该模型的参数主要集中在 `fc1`，`fc1` 单层占约 95.2% 的总参数量。这意味着按 flattened 参数位置进行 `front` 或 `spread` 选择时，大量候选位置会落在全连接层参数中，除非攻击位置长度限制在更前面的参数段内。

## 3. CIFAR-10: `ResNet18Cifar`

### 3.1 结构概览

`ResNet18Cifar` 是面向 CIFAR 小图改造的 ResNet-18。它保留标准 ResNet-18 的 `[2, 2, 2, 2]` 残差块配置，但将 ImageNet 常见的 `7 x 7` stem 和初始 max-pool 改为更适合小图的 `3 x 3` 卷积，并使用 `GroupNorm + SiLU`。

v5 配置中 CIFAR 使用：

```text
--model=resnet18_cifar
--dataset=cifar
--cifar_crop_size=32
--cifar_normalize=1
```

因此 v5 中该模型输入为标准 CIFAR-10 尺寸 `3 x 32 x 32`。

| 阶段 | 层/模块 | 输出形状 | 激活/归一化/池化 | 说明 |
| --- | --- | --- | --- | --- |
| Input | - | `3 x 32 x 32` | CIFAR mean/std normalize | v5 启用 `cifar_normalize=1` |
| Stem | `Conv2d(3, 64, kernel=3, stride=1, padding=1, bias=False)` | `64 x 32 x 32` | `GroupNorm(8, 64)` + `SiLU` | 无初始 max-pool |
| Layer 1 | 2 个 `BasicBlock(64 -> 64, stride=1)` | `64 x 32 x 32` | 每块 `GroupNorm + SiLU` | 不降采样 |
| Layer 2 | 2 个 block，首块 `64 -> 128, stride=2` | `128 x 16 x 16` | shortcut 用 `1 x 1 conv + GroupNorm` | 降采样 |
| Layer 3 | 2 个 block，首块 `128 -> 256, stride=2` | `256 x 8 x 8` | shortcut 用 `1 x 1 conv + GroupNorm` | 降采样 |
| Layer 4 | 2 个 block，首块 `256 -> 512, stride=2` | `512 x 4 x 4` | shortcut 用 `1 x 1 conv + GroupNorm` | 降采样 |
| Pool | `AdaptiveAvgPool2d((1, 1))` | `512 x 1 x 1` | - | 全局平均池化 |
| Classifier | `Linear(512, 10)` | `10` | `log_softmax(dim=1)` | 输出 10 类 log-probability |

### 3.2 `BasicBlock` 结构

每个 `BasicBlock` 的主分支是：

```text
Conv2d(in_planes, planes, 3x3, stride=s, padding=1, bias=False)
GroupNorm(8, planes)
SiLU
Conv2d(planes, planes, 3x3, stride=1, padding=1, bias=False)
GroupNorm(8, planes)
Residual add
SiLU
```

当 `stride != 1` 或通道数变化时，shortcut 分支使用：

```text
Conv2d(in_planes, planes, 1x1, stride=s, bias=False)
GroupNorm(8, planes)
```

代码变量名中残差块的归一化层叫 `bn1` / `bn2`，但实际类型是 `GroupNorm`，不是 BatchNorm。

### 3.3 参数量分解

| 模块 | 参数量 | 备注 |
| --- | ---: | --- |
| Stem `conv1` | 1,728 | `64 * 3 * 3 * 3`，无 bias |
| Stem `gn1` | 128 | `64 + 64` |
| `layer1` | 147,968 | 2 个 `64 -> 64` blocks |
| `layer2` | 525,568 | `64 -> 128` downsample block + `128 -> 128` block |
| `layer3` | 2,099,712 | `128 -> 256` downsample block + `256 -> 256` block |
| `layer4` | 8,393,728 | `256 -> 512` downsample block + `512 -> 512` block |
| Classifier `fc` | 5,130 | `512 * 10 + 10` |
| **合计** | **11,173,962** | 全部为可训练参数 |

更细的残差块参数量如下。

| 残差块 | 参数量 |
| --- | ---: |
| `layer1.0` | 73,984 |
| `layer1.1` | 73,984 |
| `layer2.0` | 230,144 |
| `layer2.1` | 295,424 |
| `layer3.0` | 919,040 |
| `layer3.1` | 1,180,672 |
| `layer4.0` | 3,673,088 |
| `layer4.1` | 4,720,640 |

### 3.4 激活函数、归一化与初始化

- 激活函数：全模型使用 `SiLU`，包括 stem 和每个残差块的两处激活。
- 归一化：全模型使用 `GroupNorm(8, channels)`，包括 stem、残差块主分支和 shortcut 分支。
- 池化：没有初始 max-pool；末尾使用 `AdaptiveAvgPool2d((1, 1))`。
- Dropout：`ResNet18Cifar` 本身不使用 Dropout。
- 卷积 bias：所有卷积层均为 `bias=False`；线性分类层带 bias。
- 初始化：卷积层使用 Kaiming normal；GroupNorm weight 初始化为 1、bias 初始化为 0；Linear weight 使用均值 0、标准差 0.01 的 normal 初始化，Linear bias 为 0。
- 输出：最后返回 `F.log_softmax(x, dim=1)`。

## 4. 与 v5 训练配置的对应关系

v5 的核心训练配置如下。

| 数据集 | 模型 | 学习率 | local epoch | local batch size | 优化相关设置 |
| --- | --- | ---: | ---: | ---: | --- |
| MNIST | `cnn` / `CNNFashion_Enhanced` | 0.01 | 10 | 16 | `momentum=0.9`, `weight_decay=0.0005`, cosine scheduler, `min_lr=0.0001` |
| Fashion-MNIST | `cnn` / `CNNFashion_Enhanced` | 0.01 | 10 | 16 | 同 MNIST |
| CIFAR-10 | `resnet18_cifar` / `ResNet18Cifar` | 0.03 | 1 | 64 | 同上，并启用 `cifar_crop_size=32`, `cifar_normalize=1` |

v5 所有实验统一使用：

```text
--num_users=10
--frac=1.0
--iid=0
--noniid_mode=dirichlet
--dirichlet_alpha=0.5
--dirichlet_min_size=100
--gama_warmup_epochs=0
```

攻击实验中还根据不同表格扫描或固定：

```text
--gama
--num_steal
--num_img_per_client
--agg_mode=segmented
--attack_position_mode=front/spread
```

## 5. 总体对比

| 维度 | MNIST/Fashion-MNIST `CNNFashion_Enhanced` | CIFAR-10 `ResNet18Cifar` |
| --- | --- | --- |
| 网络类型 | 两层 CNN + MLP 分类头 | ResNet-18 风格残差网络 |
| 参数量 | 3.37M | 11.17M |
| 主激活函数 | LeakyReLU + ReLU | SiLU |
| 归一化 | GroupNorm | GroupNorm |
| 池化 | 两次 AvgPool | 无初始池化，末尾 AdaptiveAvgPool |
| Dropout | 有，`p=0.3` | 无 |
| 输出 | `log_softmax` | `log_softmax` |
| 参数集中位置 | `fc1` 占绝大多数 | `layer4` 占绝大多数 |
| v5 角色 | MNIST/Fashion-MNIST 主实验模型 | CIFAR-10 主实验模型 |

## 6. 参数统计方法

参数量由当前代码直接实例化模型后统计：

```python
sum(p.numel() for p in model.parameters())
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

两个模型的 total parameters 与 trainable parameters 相同，说明所有参数都参与训练。

