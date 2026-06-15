# 位置选择消融实验结果总结：num_steal=10, num_img_per_client=1/5/10/50

本文档首先总结 non-IID 设置下 `front` 与 `spread` 两种攻击参数位置选择方式的对比结果，并追加 IID 设置下 `num_steal=10, num_img_per_client=50` 的最新结果。non-IID 实验设置如下：

```text
--num_users=10 --frac=1.0 --iid=0 --epochs=50 --local_bs=16 --gama_warmup_epochs=0
--gama=0.5 --agg_mode=segmented
```

non-IID 数据划分采用 label-sorted shard 方式，每个客户端分配 2 个 shard。当前 10 客户端设置下，MNIST/Fashion-MNIST 每个客户端 6000 张图，CIFAR-10 每个客户端 5000 张图，训练数据全部使用。

本次对比固定 `num_steal=10`，考察四种 payload 规模：

| 设置 | num_steal | num_img_per_client | 攻击参数长度 |
| --- | ---: | ---: | ---: |
| A | 10 | 1 | 5760 |
| B | 10 | 5 | 28800 |
| C | 10 | 10 | 57600 |
| D | 10 | 50 | 288000 |

## Non-IID 总体结论

1. non-IID 下 `spread` 的分类效用优势非常明显，尤其是 Fashion-MNIST 和 CIFAR-10。随着 `num_img_per_client` 增大，`front` 的准确率下降更严重，而 `spread` 通常能保持更高准确率和更低 loss。
2. Fashion-MNIST 是最能体现 `spread` 整体优势的数据集。`num_img_per_client=10/50` 时，`spread` 同时取得更高准确率、更低 loss 和更低最终 MAPE。
3. CIFAR-10 上 `spread` 在效用保持方面优势最大。`num_img_per_client=50` 时，`front` 最终准确率只有 6.90%，而 `spread` 为 42.12%，差距达到 +35.22pp；但 CIFAR-10 的最终 MAPE 不总是 `spread` 更低。
4. MNIST 上 `spread` 明显提升大 payload 下的分类效用，但恢复误差多数仍由 `front` 更低。只有 `num_img_per_client=50` 的中期轮次中，`spread` 曾取得更低 MAPE。

## 设置 A：num_steal=10, num_img_per_client=1

| 数据集 | 位置方式 | Final Acc | Final Loss | Final MAPE | Best Acc | Best MAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | front | 98.00% | 0.6876 | 0.0345 | 98.02% | 0.0345 |
| MNIST | spread | 98.37% | 0.5089 | 0.2183 | 98.53% | 0.2183 |
| Fashion-MNIST | front | 84.27% | 4.5647 | 0.0173 | 87.95% | 0.0173 |
| Fashion-MNIST | spread | 86.87% | 3.9479 | 0.0982 | 87.77% | 0.0782 |
| CIFAR-10 CNN | front | 44.00% | 14.3753 | 0.0546 | 45.54% | 0.0538 |
| CIFAR-10 CNN | spread | 46.20% | 14.3241 | 0.0612 | 48.40% | 0.0596 |

`spread` 相对 `front` 的差值：

| 数据集 | Final Acc 差值 | Final Loss 差值 | Final MAPE 差值 | Best Acc 差值 | Best MAPE 差值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | +0.37pp | -0.1787 | +0.1838 | +0.52pp | +0.1838 |
| Fashion-MNIST | +2.60pp | -0.6169 | +0.0808 | -0.18pp | +0.0609 |
| CIFAR-10 CNN | +2.20pp | -0.0512 | +0.0066 | +2.86pp | +0.0058 |

设置 A 下，`spread` 在三个数据集上都提升最终准确率并降低 final loss，但 MAPE 均高于 `front`。这说明小 payload 下 `spread` 主要体现为效用优势，不体现恢复误差优势。

### 设置 A 的中期轮次

| 数据集 | Round | Acc 差值 | Loss 差值 | MAPE 差值 |
| --- | ---: | ---: | ---: | ---: |
| MNIST | 20 | +0.50pp | -0.1266 | +0.1761 |
| MNIST | 25 | +0.62pp | -0.1613 | +0.1783 |
| MNIST | 30 | +0.60pp | -0.2040 | +0.1796 |
| Fashion-MNIST | 20 | +2.85pp | -0.6849 | +0.0674 |
| Fashion-MNIST | 25 | -1.13pp | +0.1343 | +0.0732 |
| Fashion-MNIST | 30 | +0.85pp | -0.2845 | +0.0755 |
| CIFAR-10 CNN | 20 | +2.46pp | -0.7021 | -0.0017 |
| CIFAR-10 CNN | 25 | +2.80pp | -1.2270 | +0.0021 |
| CIFAR-10 CNN | 30 | +2.08pp | -1.4258 | +0.0055 |

## 设置 B：num_steal=10, num_img_per_client=5

| 数据集 | 位置方式 | Final Acc | Final Loss | Final MAPE | Best Acc | Best MAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | front | 97.12% | 1.0521 | 0.0339 | 97.12% | 0.0336 |
| MNIST | spread | 98.43% | 0.5174 | 0.1401 | 98.53% | 0.1301 |
| Fashion-MNIST | front | 76.28% | 7.0028 | 0.0138 | 81.35% | 0.0138 |
| Fashion-MNIST | spread | 86.67% | 3.9290 | 0.0547 | 87.88% | 0.0387 |
| CIFAR-10 CNN | front | 44.32% | 14.8775 | 0.0807 | 45.20% | 0.0709 |
| CIFAR-10 CNN | spread | 47.70% | 13.6819 | 0.0752 | 48.24% | 0.0678 |

`spread` 相对 `front` 的差值：

| 数据集 | Final Acc 差值 | Final Loss 差值 | Final MAPE 差值 | Best Acc 差值 | Best MAPE 差值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | +1.32pp | -0.5347 | +0.1062 | +1.42pp | +0.0966 |
| Fashion-MNIST | +10.38pp | -3.0738 | +0.0409 | +6.53pp | +0.0249 |
| CIFAR-10 CNN | +3.38pp | -1.1956 | -0.0055 | +3.04pp | -0.0031 |

设置 B 下，`spread` 的效用优势显著增强。Fashion-MNIST 的最终准确率提升达到 +10.38pp，CIFAR-10 也达到 +3.38pp。CIFAR-10 是本组中 `spread` 同时改善准确率、loss 和 MAPE 的数据集。

### 设置 B 的中期轮次

| 数据集 | Round | Acc 差值 | Loss 差值 | MAPE 差值 |
| --- | ---: | ---: | ---: | ---: |
| MNIST | 20 | +3.82pp | -1.6048 | +0.0975 |
| MNIST | 25 | +3.98pp | -1.4149 | +0.1028 |
| MNIST | 30 | +10.78pp | -3.4240 | +0.1018 |
| Fashion-MNIST | 20 | +12.08pp | -2.7306 | +0.0252 |
| Fashion-MNIST | 25 | +16.13pp | -3.6116 | +0.0305 |
| Fashion-MNIST | 30 | +8.95pp | -2.4098 | +0.0344 |
| CIFAR-10 CNN | 20 | +0.84pp | -1.3265 | -0.0090 |
| CIFAR-10 CNN | 25 | -1.26pp | -0.9413 | -0.0095 |
| CIFAR-10 CNN | 30 | +2.10pp | -0.8882 | -0.0077 |

## 设置 C：num_steal=10, num_img_per_client=10

| 数据集 | 位置方式 | Final Acc | Final Loss | Final MAPE | Best Acc | Best MAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | front | 93.78% | 2.0068 | 0.0865 | 94.73% | 0.0865 |
| MNIST | spread | 98.48% | 0.5107 | 0.1449 | 98.55% | 0.1322 |
| Fashion-MNIST | front | 61.82% | 9.3432 | 0.0381 | 75.45% | 0.0381 |
| Fashion-MNIST | spread | 86.05% | 4.0942 | 0.0525 | 87.23% | 0.0424 |
| CIFAR-10 CNN | front | 43.36% | 14.7071 | 0.0672 | 45.28% | 0.0657 |
| CIFAR-10 CNN | spread | 44.78% | 14.5659 | 0.0667 | 47.42% | 0.0667 |

`spread` 相对 `front` 的差值：

| 数据集 | Final Acc 差值 | Final Loss 差值 | Final MAPE 差值 | Best Acc 差值 | Best MAPE 差值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | +4.70pp | -1.4960 | +0.0584 | +3.82pp | +0.0456 |
| Fashion-MNIST | +24.23pp | -5.2490 | +0.0144 | +11.78pp | +0.0042 |
| CIFAR-10 CNN | +1.42pp | -0.1411 | -0.0005 | +2.14pp | +0.0009 |

设置 C 下，Fashion-MNIST 的 `front` 准确率明显崩塌，`spread` 最终准确率高 +24.23pp。CIFAR-10 中 `spread` 的最终 MAPE 略低，但 best MAPE 基本持平且略高。MNIST 中 `spread` 维持模型效用，但 MAPE 仍高于 `front`。

### 设置 C 的中期轮次

| 数据集 | Round | Acc 差值 | Loss 差值 | MAPE 差值 |
| --- | ---: | ---: | ---: | ---: |
| MNIST | 20 | +4.58pp | -1.7984 | +0.0490 |
| MNIST | 25 | +5.57pp | -1.8927 | +0.0550 |
| MNIST | 30 | +9.13pp | -3.0590 | +0.0554 |
| Fashion-MNIST | 20 | +20.05pp | -5.0447 | -0.0156 |
| Fashion-MNIST | 25 | +30.27pp | -7.3700 | -0.0038 |
| Fashion-MNIST | 30 | +14.65pp | -3.7823 | +0.0028 |
| CIFAR-10 CNN | 20 | +9.54pp | -2.4117 | +0.0029 |
| CIFAR-10 CNN | 25 | +2.66pp | -2.0436 | +0.0021 |
| CIFAR-10 CNN | 30 | +5.60pp | -3.0668 | +0.0009 |

## 设置 D：num_steal=10, num_img_per_client=50

| 数据集 | 位置方式 | Final Acc | Final Loss | Final MAPE | Best Acc | Best MAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | front | 89.98% | 3.2017 | 0.1237 | 96.28% | 0.1237 |
| MNIST | spread | 98.38% | 0.5374 | 0.1303 | 98.50% | 0.1285 |
| Fashion-MNIST | front | 70.13% | 9.1446 | 0.0894 | 74.62% | 0.0894 |
| Fashion-MNIST | spread | 85.02% | 4.3022 | 0.0589 | 86.72% | 0.0589 |
| CIFAR-10 CNN | front | 6.90% | 27.5652 | 0.0771 | 18.30% | 0.0731 |
| CIFAR-10 CNN | spread | 42.12% | 14.4248 | 0.0979 | 43.72% | 0.0803 |

`spread` 相对 `front` 的差值：

| 数据集 | Final Acc 差值 | Final Loss 差值 | Final MAPE 差值 | Best Acc 差值 | Best MAPE 差值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | +8.40pp | -2.6643 | +0.0066 | +2.22pp | +0.0048 |
| Fashion-MNIST | +14.88pp | -4.8424 | -0.0305 | +12.10pp | -0.0305 |
| CIFAR-10 CNN | +35.22pp | -13.1404 | +0.0209 | +25.42pp | +0.0072 |

设置 D 是最大 payload 场景，最能体现 `front` 对分类效用的破坏。CIFAR-10 中 `front` 最终准确率降到 6.90%，而 `spread` 仍保持 42.12%；Fashion-MNIST 中 `spread` 同时提升准确率并降低 MAPE；MNIST 中 `spread` 大幅提升准确率，但最终 MAPE 仍略高。

### 设置 D 的中期轮次

| 数据集 | Round | Acc 差值 | Loss 差值 | MAPE 差值 |
| --- | ---: | ---: | ---: | ---: |
| MNIST | 20 | +9.08pp | -2.8729 | -0.0373 |
| MNIST | 25 | +5.80pp | -1.8712 | -0.0225 |
| MNIST | 30 | +7.53pp | -2.1681 | -0.0126 |
| Fashion-MNIST | 20 | +14.87pp | -4.9161 | -0.0648 |
| Fashion-MNIST | 25 | +35.70pp | -7.5905 | -0.0560 |
| Fashion-MNIST | 30 | +36.97pp | -8.8845 | -0.0491 |
| CIFAR-10 CNN | 20 | +27.72pp | -9.7000 | +0.0127 |
| CIFAR-10 CNN | 25 | +19.58pp | -10.7376 | +0.0161 |
| CIFAR-10 CNN | 30 | +28.78pp | -16.3621 | +0.0168 |

## 新增 IID 结果：num_steal=10, num_img_per_client=50

新增实验固定 10 个目标客户端、每个目标客户端记录 50 张图，用于在 IID 设置下对比 `front` 与 `spread`。实验设置如下：

```text
--num_users=10 --frac=1.0 --iid=1 --epochs=50 --local_bs=16 --gama_warmup_epochs=0
--gama=0.5 --agg_mode=segmented --num_steal=10 --num_img_per_client=50
```

对应的 epoch 曲线图已输出到 `save/plots/ablation/`：

- `mnist_iid1_numsteal10_numimg50_position_ablation_curves.png`
- `fmnist_iid1_numsteal10_numimg50_position_ablation_curves.png`
- `cifar_iid1_numsteal10_numimg50_position_ablation_curves.png`

### 最终结果

| 数据集 | 位置方式 | Final Acc | Final Loss | Final MAPE | Best Acc | Best MAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | front | 99.22% | 0.2577 | 0.1030 | 99.30% | 0.1030 |
| MNIST | spread | 99.17% | 0.2999 | 0.1470 | 99.27% | 0.1466 |
| Fashion-MNIST | front | 92.65% | 3.1824 | 0.1802 | 92.75% | 0.1802 |
| Fashion-MNIST | spread | 93.00% | 3.9333 | 0.1484 | 93.18% | 0.1484 |
| CIFAR-10 CNN | front | 78.52% | 7.7625 | 0.1006 | 78.64% | 0.0892 |
| CIFAR-10 CNN | spread | 80.44% | 7.3056 | 0.1549 | 81.78% | 0.0934 |

`spread` 相对 `front` 的差值：

| 数据集 | Final Acc 差值 | Final Loss 差值 | Final MAPE 差值 | Best Acc 差值 | Best MAPE 差值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | -0.05pp | +0.0422 | +0.0439 | -0.03pp | +0.0436 |
| Fashion-MNIST | +0.35pp | +0.7509 | -0.0317 | +0.43pp | -0.0317 |
| CIFAR-10 CNN | +1.92pp | -0.4569 | +0.0542 | +3.14pp | +0.0042 |

IID 大 payload 下，`spread` 不再像 non-IID 设置那样呈现单边优势。MNIST 中 `front` 的最终准确率、loss 和 MAPE 都略好；Fashion-MNIST 中 `spread` 带来小幅准确率提升并明显降低 MAPE，但 loss 更高；CIFAR-10 中 `spread` 明显提高准确率并降低 loss，但最终 MAPE 高于 `front`。

### 关键轮次

| 数据集 | Round | Acc 差值 | Loss 差值 | MAPE 差值 |
| --- | ---: | ---: | ---: | ---: |
| MNIST | 20 | +0.00pp | +0.0252 | -0.0083 |
| MNIST | 30 | +0.13pp | +0.0170 | +0.0196 |
| MNIST | 40 | -0.08pp | +0.0355 | +0.0345 |
| MNIST | 50 | -0.05pp | +0.0422 | +0.0439 |
| Fashion-MNIST | 20 | +0.92pp | +0.2980 | -0.0432 |
| Fashion-MNIST | 30 | +0.50pp | +0.4315 | -0.0343 |
| Fashion-MNIST | 40 | +0.53pp | +0.5524 | -0.0308 |
| Fashion-MNIST | 50 | +0.35pp | +0.7509 | -0.0317 |
| CIFAR-10 CNN | 20 | +4.64pp | -1.1248 | +0.0332 |
| CIFAR-10 CNN | 30 | +3.82pp | -0.8056 | +0.0429 |
| CIFAR-10 CNN | 40 | +3.04pp | -0.6876 | +0.0496 |
| CIFAR-10 CNN | 50 | +1.92pp | -0.4569 | +0.0542 |

从曲线趋势看，MNIST 两种位置方式的准确率几乎重合，差异主要体现在 MAPE；Fashion-MNIST 的 `spread` 在全程保持更低 MAPE；CIFAR-10 的 `spread` 从中期开始稳定保持更高准确率和更低 loss，但 MAPE 逐步高于 `front`。

## Non-IID 重点判断

从分类效用看，non-IID 下 `spread` 的优势比 IID 更强：

- MNIST：随着 payload 增大，`spread` 最终准确率优势从 +0.37pp 增至 +8.40pp。
- Fashion-MNIST：`spread` 在四组设置中最终准确率均高于 `front`，最大优势出现在 `num_img_per_client=10`，达到 +24.23pp。
- CIFAR-10 CNN：`spread` 始终有更高最终准确率；在 `num_img_per_client=50` 时优势达到 +35.22pp。

从恢复误差看，`spread` 的优势更依赖数据集和 payload：

- MNIST：最终 MAPE 仍由 `front` 更低，但 `num_img_per_client=50` 的中期轮次里 `spread` 曾有更低 MAPE。
- Fashion-MNIST：小 payload 下 `front` MAPE 更低；大 payload 下 `spread` 更好，尤其 `num_img_per_client=50` 时最终 MAPE 低 0.0305。
- CIFAR-10 CNN：`num_img_per_client=5/10` 时 `spread` 的最终 MAPE 更低或基本持平；`num_img_per_client=50` 时 `front` 最终 MAPE 更低，但其分类准确率已经严重失效。

综合来看，non-IID 设置更有利于突出 `spread` 的核心优势：在客户端数据分布更异质、payload 更大的情况下，分散攻击位置能显著降低对模型效用的破坏。若论文叙述强调“效用-恢复折中”，最有力的证据是 Fashion-MNIST 的大 payload 设置和 CIFAR-10 的效用保持结果。
