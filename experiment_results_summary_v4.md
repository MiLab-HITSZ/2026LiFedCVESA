# 实验结果总结 v4：iid=0、50 轮、batch size 16、无 gama warm-up

本文档基于 `save/results` 中保存的 `C[1.0]`、`iid=0`、`epochs=50`、`local_bs=16` 结果文件整理。相对 `experiment_results_summary_v3.md`，本版保留相同统计口径，并加入本轮固定 `num_steal=10`、扫描 `num_img_per_client=1/5/10/50` 的位置消融结果。

对应 10 客户端全参与配置：

```text
--num_users=10 --frac=1.0 --iid=0 --epochs=50 --local_bs=16 --gama_warmup_epochs=0
```

完整性检查如下：

- `scripts_10clients/logs_multi_per_gpu` 中共有 65 个 log。
- 65 个 log 均包含 `Results after 50 global rounds of training`。
- 未匹配到 `Traceback`、`Failed job`、`RuntimeError`、`CUDA out of memory` 等失败信息。
- `save/results` 中按 v4 口径共有 56 个去重实验配置、164 个 `.npy` 文件，符合 4 个无攻击任务仅保存 `acc/loss`、52 个攻击任务保存 `acc/loss/mape` 的预期。

统计口径与 `experiment_results_summary_v3.md` 保持一致：

- `Acc`、`Loss`、`MAPE` 均取训练结束第 50 轮的最终值。
- `Best Acc` 表示 50 轮内最高准确率。
- `Best MAPE` 表示 50 轮内最低 MAPE，数值越低代表恢复误差越小。
- `gama=0` 是无攻击基线，因此没有 MAPE 文件。

## 总体结论

v4 结果可以概括为以下几点：

1. 非 IID 下整体准确率明显低于 IID：MNIST-CNN 约 98.4%-98.6%，Fashion-MNIST-CNN 约 84.9%-88.1%，CIFAR-10-CNN 约 46%-49%。
2. 攻击强度扫描中，MNIST-CNN 和 Fashion-MNIST-CNN 的最终 MAPE 随 `gama` 增大整体下降；CIFAR-CNN 的最低最终 MAPE 出现在 `gama=0.5`，为 0.0909。
3. CIFAR-10 ResNet18 在非 IID 下波动较大，`gama=1.0` 最终准确率降到 23.70%，但 Best Acc 仍达到 41.98%；最终 MAPE 在 0.0818-0.0821 之间，非常接近。
4. `num_steal` 扫描中，MNIST-CNN 最低最终 MAPE 出现在 `num_steal=3`，为 0.1953；Fashion-MNIST-CNN 最低最终 MAPE 出现在 `num_steal=5`，为 0.0284；CIFAR-CNN 最低最终 MAPE 出现在 `num_steal=1`，为 0.0279。
5. 位置消融中，`spread` 在所有设置里都提高最终准确率，尤其 Fashion-MNIST 和 CIFAR-10 在较大 `num_img_per_client` 下优势明显。
6. 从最终 MAPE 看，`front` 在 MNIST 的四个位置消融设置中都更低；Fashion-MNIST 在 `num_img_per_client=50` 时 `spread` 更低；CIFAR-10 在 `num_img_per_client=5/10` 时 `spread` 略低，其余设置 `front` 更低。

## 实验一：攻击强度 `gama` 扫描

设置：固定 `num_steal=5`、`num_img_per_client=1`、默认 `segmented` 聚合和 `spread` 攻击位置，扫描 `gama=0/0.05/0.2/0.5/1.0`。

### MNIST

| gama | Acc | Loss | MAPE | Best Acc | Best MAPE |
| --- | --- | --- | --- | --- | --- |
| 0 | 98.42% | 0.5331 | - | 98.50% | - |
| 0.05 | 98.52% | 0.4982 | 0.2278 | 98.55% | 0.2193 |
| 0.2 | 98.53% | 0.4975 | 0.2256 | 98.58% | 0.2167 |
| 0.5 | 98.52% | 0.4978 | 0.2226 | 98.58% | 0.2116 |
| 1.0 | 98.53% | 0.4992 | 0.2165 | 98.57% | 0.2037 |

- MNIST-CNN 在 iid=0 下最终准确率保持在 98.42%-98.53%。
- 最低最终 MAPE 出现在 `gama=1.0`，为 0.2165；Best MAPE 也在 `gama=1.0` 最低，为 0.2037。

### Fashion-MNIST

| gama | Acc | Loss | MAPE | Best Acc | Best MAPE |
| --- | --- | --- | --- | --- | --- |
| 0 | 87.73% | 3.8037 | - | 87.77% | - |
| 0.05 | 85.02% | 4.4164 | 0.0306 | 87.48% | 0.0297 |
| 0.2 | 85.13% | 4.4142 | 0.0306 | 87.28% | 0.0280 |
| 0.5 | 85.27% | 4.3647 | 0.0284 | 87.43% | 0.0265 |
| 1.0 | 84.88% | 4.4175 | 0.0266 | 87.32% | 0.0247 |

- Fashion-MNIST-CNN 的攻击组最终准确率约为 84.88%-85.27%，低于无攻击基线 87.73%。
- 最低最终 MAPE 出现在 `gama=1.0`，为 0.0266；Best MAPE 也在 `gama=1.0` 最低，为 0.0247。

### CIFAR-10 CNN

| gama | Acc | Loss | MAPE | Best Acc | Best MAPE |
| --- | --- | --- | --- | --- | --- |
| 0 | 47.18% | 14.1772 | - | 47.74% | - |
| 0.05 | 48.68% | 13.6980 | 0.1011 | 49.06% | 0.0952 |
| 0.2 | 48.54% | 13.5419 | 0.0977 | 49.08% | 0.0957 |
| 0.5 | 47.14% | 13.5113 | 0.0909 | 48.52% | 0.0909 |
| 1.0 | 47.18% | 13.5457 | 0.0967 | 48.56% | 0.0917 |

- CIFAR-CNN 的最终准确率在 47.14%-48.68% 之间。
- 最低最终 MAPE 出现在 `gama=0.5`，为 0.0909。

### CIFAR-10 ResNet18

| gama | Acc | Loss | MAPE | Best Acc | Best MAPE |
| --- | --- | --- | --- | --- | --- |
| 0 | 30.70% | 19.7911 | - | 35.04% | - |
| 0.05 | 42.76% | 13.9524 | 0.0820 | 43.50% | 0.0818 |
| 0.2 | 39.52% | 14.4580 | 0.0820 | 40.90% | 0.0819 |
| 0.5 | 41.32% | 15.0968 | 0.0821 | 45.70% | 0.0819 |
| 1.0 | 23.70% | 17.8354 | 0.0818 | 41.98% | 0.0816 |

- CIFAR-ResNet18 的最终准确率波动较大，最高出现在 `gama=0.05`，为 42.76%；最低出现在 `gama=1.0`，为 23.70%。
- 最低最终 MAPE 出现在 `gama=1.0`，为 0.0818；但各攻击强度的最终 MAPE 差距很小。

## 实验二：攻击目标数量 `num_steal` 扫描

设置：固定 `gama=0.5`、`num_img_per_client=1`、默认 `segmented` 聚合和 `spread` 攻击位置，扫描 `num_steal=1/2/3/4/5/10`。本实验只统计 CNN。

### MNIST

| num_steal | Acc | Loss | MAPE | Best Acc | Best MAPE |
| --- | --- | --- | --- | --- | --- |
| 1 | 98.52% | 0.5014 | 0.3801 | 98.57% | 0.3801 |
| 2 | 98.57% | 0.5060 | 0.2031 | 98.57% | 0.2014 |
| 3 | 98.40% | 0.5218 | 0.1953 | 98.60% | 0.1883 |
| 4 | 98.43% | 0.5183 | 0.2328 | 98.57% | 0.2321 |
| 5 | 98.52% | 0.4978 | 0.2226 | 98.58% | 0.2116 |
| 10 | 98.37% | 0.5089 | 0.2183 | 98.53% | 0.2183 |

- MNIST-CNN 在不同 `num_steal` 下最终准确率保持在 98.37%-98.57%。
- 最低最终 MAPE 出现在 `num_steal=3`，为 0.1953；Best MAPE 也在 `num_steal=3` 最低，为 0.1883。

### Fashion-MNIST

| num_steal | Acc | Loss | MAPE | Best Acc | Best MAPE |
| --- | --- | --- | --- | --- | --- |
| 1 | 87.63% | 3.9508 | 0.3189 | 88.20% | 0.3186 |
| 2 | 86.37% | 4.1232 | 0.1140 | 87.73% | 0.0800 |
| 3 | 88.07% | 3.6294 | 0.1117 | 88.88% | 0.1034 |
| 4 | 85.20% | 4.3982 | 0.1173 | 87.60% | 0.0999 |
| 5 | 85.27% | 4.3647 | 0.0284 | 87.43% | 0.0265 |
| 10 | 86.87% | 3.9479 | 0.0982 | 87.77% | 0.0782 |

- Fashion-MNIST-CNN 最终准确率保持在 85.20%-88.07%。
- 最低最终 MAPE 出现在 `num_steal=5`，为 0.0284；Best MAPE 也在 `num_steal=5` 最低，为 0.0265。

### CIFAR-10 CNN

| num_steal | Acc | Loss | MAPE | Best Acc | Best MAPE |
| --- | --- | --- | --- | --- | --- |
| 1 | 46.60% | 13.8426 | 0.0279 | 47.26% | 0.0279 |
| 2 | 47.36% | 13.7946 | 0.0467 | 47.36% | 0.0460 |
| 3 | 48.62% | 13.3742 | 0.0535 | 49.12% | 0.0340 |
| 4 | 47.52% | 13.9372 | 0.1131 | 48.70% | 0.0546 |
| 5 | 47.14% | 13.5113 | 0.0909 | 48.52% | 0.0909 |
| 10 | 46.20% | 14.3241 | 0.0612 | 48.40% | 0.0596 |

- CIFAR-CNN 最终准确率保持在 46.20%-48.62%。
- 最低最终 MAPE 出现在 `num_steal=1`，为 0.0279；最高最终准确率出现在 `num_steal=3`，为 48.62%。

## 实验三：攻击参数位置选择消融

设置：固定 `gama=0.5`、默认 `segmented` 聚合，比较 `front` 与 `spread`。本实验只统计 CNN，并固定 `num_steal=10`，扫描 `num_img_per_client=1/5/10/50`。

| 设置 | num_steal | num_img_per_client | 攻击参数长度 |
| --- | ---: | ---: | ---: |
| A | 10 | 1 | `10 * 1 * 576 = 5760` |
| B | 10 | 5 | `10 * 5 * 576 = 28800` |
| C | 10 | 10 | `10 * 10 * 576 = 57600` |
| D | 10 | 50 | `10 * 50 * 576 = 288000` |

### 设置 A：`num_steal=10`、`num_img_per_client=1`

| 数据集 | 位置方式 | Acc | Loss | MAPE | Best Acc | Best MAPE |
| --- | --- | --- | --- | --- | --- | --- |
| MNIST | front | 98.00% | 0.6876 | 0.0345 | 98.02% | 0.0345 |
| MNIST | spread | 98.37% | 0.5089 | 0.2183 | 98.53% | 0.2183 |
| Fashion-MNIST | front | 84.27% | 4.5647 | 0.0173 | 87.95% | 0.0173 |
| Fashion-MNIST | spread | 86.87% | 3.9479 | 0.0982 | 87.77% | 0.0782 |
| CIFAR-10 CNN | front | 44.00% | 14.3753 | 0.0546 | 45.54% | 0.0538 |
| CIFAR-10 CNN | spread | 46.20% | 14.3241 | 0.0612 | 48.40% | 0.0596 |

spread 相对 front 的差值：

| 数据集 | Acc 差值 | MAPE 差值 | 解释 |
| --- | ---: | ---: | --- |
| MNIST | +0.37pp | +0.1838 | `spread` 准确率略高，但最终 MAPE 明显差于 `front`。 |
| Fashion-MNIST | +2.60pp | +0.0808 | `spread` 准确率更高，`front` 窃取误差更低。 |
| CIFAR-10 CNN | +2.20pp | +0.0066 | `spread` 准确率更高，最终 MAPE 略差。 |

### 设置 B：`num_steal=10`、`num_img_per_client=5`

| 数据集 | 位置方式 | Acc | Loss | MAPE | Best Acc | Best MAPE |
| --- | --- | --- | --- | --- | --- | --- |
| MNIST | front | 97.12% | 1.0521 | 0.0339 | 97.12% | 0.0336 |
| MNIST | spread | 98.43% | 0.5174 | 0.1401 | 98.53% | 0.1301 |
| Fashion-MNIST | front | 76.28% | 7.0028 | 0.0138 | 81.35% | 0.0138 |
| Fashion-MNIST | spread | 86.67% | 3.9290 | 0.0547 | 87.88% | 0.0387 |
| CIFAR-10 CNN | front | 44.32% | 14.8775 | 0.0807 | 45.20% | 0.0709 |
| CIFAR-10 CNN | spread | 47.70% | 13.6819 | 0.0752 | 48.24% | 0.0678 |

spread 相对 front 的差值：

| 数据集 | Acc 差值 | MAPE 差值 | 解释 |
| --- | ---: | ---: | --- |
| MNIST | +1.32pp | +0.1062 | `spread` 准确率更高，`front` 最终 MAPE 更低。 |
| Fashion-MNIST | +10.38pp | +0.0409 | `front` 准确率损失明显，但最终 MAPE 更低。 |
| CIFAR-10 CNN | +3.38pp | -0.0055 | `spread` 同时取得更高准确率和略低最终 MAPE。 |

### 设置 C：`num_steal=10`、`num_img_per_client=10`

| 数据集 | 位置方式 | Acc | Loss | MAPE | Best Acc | Best MAPE |
| --- | --- | --- | --- | --- | --- | --- |
| MNIST | front | 93.78% | 2.0068 | 0.0865 | 94.73% | 0.0865 |
| MNIST | spread | 98.48% | 0.5107 | 0.1449 | 98.55% | 0.1322 |
| Fashion-MNIST | front | 61.82% | 9.3432 | 0.0381 | 75.45% | 0.0381 |
| Fashion-MNIST | spread | 86.05% | 4.0942 | 0.0525 | 87.23% | 0.0424 |
| CIFAR-10 CNN | front | 43.36% | 14.7071 | 0.0672 | 45.28% | 0.0657 |
| CIFAR-10 CNN | spread | 44.78% | 14.5659 | 0.0667 | 47.42% | 0.0667 |

spread 相对 front 的差值：

| 数据集 | Acc 差值 | MAPE 差值 | 解释 |
| --- | ---: | ---: | --- |
| MNIST | +4.70pp | +0.0584 | `spread` 准确率明显更高，`front` 最终 MAPE 更低。 |
| Fashion-MNIST | +24.23pp | +0.0144 | `front` 准确率损失很大，但最终 MAPE 略低。 |
| CIFAR-10 CNN | +1.42pp | -0.0005 | `spread` 准确率更高，最终 MAPE 基本持平且略低。 |

### 设置 D：`num_steal=10`、`num_img_per_client=50`

| 数据集 | 位置方式 | Acc | Loss | MAPE | Best Acc | Best MAPE |
| --- | --- | --- | --- | --- | --- | --- |
| MNIST | front | 89.98% | 3.2017 | 0.1237 | 96.28% | 0.1237 |
| MNIST | spread | 98.38% | 0.5374 | 0.1303 | 98.50% | 0.1285 |
| Fashion-MNIST | front | 70.13% | 9.1446 | 0.0894 | 74.62% | 0.0894 |
| Fashion-MNIST | spread | 85.02% | 4.3022 | 0.0589 | 86.72% | 0.0589 |
| CIFAR-10 CNN | front | 6.90% | 27.5652 | 0.0771 | 18.30% | 0.0731 |
| CIFAR-10 CNN | spread | 42.12% | 14.4248 | 0.0979 | 43.72% | 0.0803 |

spread 相对 front 的差值：

| 数据集 | Acc 差值 | MAPE 差值 | 解释 |
| --- | ---: | ---: | --- |
| MNIST | +8.40pp | +0.0066 | `spread` 准确率明显更高，最终 MAPE 略差。 |
| Fashion-MNIST | +14.88pp | -0.0305 | `spread` 同时取得更高准确率和更低最终 MAPE。 |
| CIFAR-10 CNN | +35.22pp | +0.0209 | `front` 准确率崩塌，`spread` 明显更稳。 |

## spread vs front 重点判断

从准确率看，`spread` 在非 IID 下明显更稳：

- MNIST 四组中 `spread` 均高于 `front`，优势从 +0.37pp 增至 +8.40pp。
- Fashion-MNIST 四组中 `spread` 均高于 `front`，尤其 `num_img_per_client=10` 时高 +24.23pp。
- CIFAR-10 CNN 四组中 `spread` 均高于 `front`；`num_img_per_client=50` 时 `front` 最终准确率只有 6.90%，`spread` 为 42.12%。

从窃取效果看，`front` 常有更低 MAPE，但代价明显：

- MNIST 中 `front` 的最终 MAPE 四组都低于 `spread`。
- Fashion-MNIST 中 `front` 在 `num_img_per_client=1/5/10` 下最终 MAPE 更低，但准确率损失很大；`num_img_per_client=50` 时 `spread` 的最终 MAPE 更低。
- CIFAR-10 CNN 中 `spread` 在 `num_img_per_client=5/10` 时最终 MAPE 略低；在 `num_img_per_client=1/50` 时最终 MAPE 高于 `front`，但准确率更稳。

综合判断：

- 如果只看窃取误差，`front` 在 MNIST 和部分 Fashion-MNIST 设置中更强。
- 如果同时考虑分类效用，非 IID 设置下 `spread` 更适合作为主结果，因为 `front` 在 Fashion-MNIST 和 CIFAR-10 上会造成明显准确率损伤。
- 对 CIFAR-10 而言，`front` 在大 payload 设置下风险最高；`num_img_per_client=50` 时最终准确率仅 6.90%，不适合作为稳健主配置。
