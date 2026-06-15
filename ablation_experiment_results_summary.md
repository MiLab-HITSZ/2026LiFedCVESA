# 消融实验结果总结：num_steal=10, num_img_per_client=5/10

本文档总结 `scripts_10clients/position_ablation` 中 `front` 与 `spread` 两种攻击参数位置选择方式的对比结果。实验设置如下：

```text
--num_users=10 --frac=1.0 --epochs=50 --local_bs=16 --gama_warmup_epochs=0
--gama=0.5 --agg_mode=segmented
```

本次对比固定 `num_steal=10`，分别考察两种 payload 规模：

| 设置 | num_steal | num_img_per_client | 攻击参数长度 |
| --- | ---: | ---: | ---: |
| A | 10 | 5 | 28800 |
| B | 10 | 10 | 57600 |

## 总体结论

1. CIFAR-10 是最能体现 `spread` 优势的数据集。`num_img_per_client=5` 时，`spread` 同时取得更高最终准确率、更低 loss 和更低最终 MAPE；`num_img_per_client=10` 时，`spread` 最终准确率高 2.18pp，best accuracy 高 2.30pp，best MAPE 也更低。
2. Fashion-MNIST 上，`spread` 在两组 payload 下都带来更高分类准确率。`num_img_per_client=10` 时，`spread` 还取得更低最终 MAPE 和更低 best MAPE，是 Fashion-MNIST 上更均衡的选择。
3. MNIST 上，`front` 的恢复误差始终更低；`spread` 的优势主要体现在部分轮次 loss 更低，分类准确率基本持平或略低。因此 MNIST 不适合作为强调 `spread` 恢复优势的主要证据。
4. 从第 20/25/30 轮的中期结果看，CIFAR-10 的 `spread` 优势在训练中期已经稳定出现；Fashion-MNIST 的 `spread` 优势主要体现为准确率更高；MNIST 中期仍更支持 `front` 的恢复效果。

## 设置 A：num_steal=10, num_img_per_client=5

| 数据集 | 位置方式 | Final Acc | Final Loss | Final MAPE | Best Acc | Best MAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | front | 99.17% | 0.3166 | 0.0438 | 99.30% | 0.0438 |
| MNIST | spread | 99.17% | 0.2935 | 0.1586 | 99.23% | 0.1426 |
| Fashion-MNIST | front | 92.65% | 3.8586 | 0.0579 | 92.90% | 0.0579 |
| Fashion-MNIST | spread | 93.17% | 3.9633 | 0.1157 | 93.48% | 0.0741 |
| CIFAR-10 CNN | front | 80.54% | 7.4170 | 0.0790 | 80.60% | 0.0786 |
| CIFAR-10 CNN | spread | 81.58% | 6.8824 | 0.0737 | 81.84% | 0.0687 |

`spread` 相对 `front` 的差值：

| 数据集 | Final Acc 差值 | Final Loss 差值 | Final MAPE 差值 | Best Acc 差值 | Best MAPE 差值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | +0.00pp | -0.0230 | +0.1148 | -0.07pp | +0.0989 |
| Fashion-MNIST | +0.52pp | +0.1047 | +0.0578 | +0.58pp | +0.0162 |
| CIFAR-10 CNN | +1.04pp | -0.5346 | -0.0053 | +1.24pp | -0.0099 |

设置 A 下，`spread` 的优势集中在 CIFAR-10：最终准确率高 1.04pp，loss 低 0.5346，最终 MAPE 低 0.0053。Fashion-MNIST 上 `spread` 准确率更高，但 MAPE 更高；MNIST 上两者最终准确率持平，但 `front` 的 MAPE 明显更低。

### 设置 A 的中期轮次

| 数据集 | Round | Acc 差值 | Loss 差值 | MAPE 差值 |
| --- | ---: | ---: | ---: | ---: |
| MNIST | 20 | +0.02pp | -0.0044 | +0.0828 |
| MNIST | 25 | -0.10pp | +0.0011 | +0.0923 |
| MNIST | 30 | -0.03pp | -0.0225 | +0.0981 |
| Fashion-MNIST | 20 | +0.38pp | +0.0274 | +0.0025 |
| Fashion-MNIST | 25 | +0.62pp | +0.1019 | +0.0160 |
| Fashion-MNIST | 30 | +0.87pp | +0.1090 | +0.0293 |
| CIFAR-10 CNN | 20 | +2.22pp | -0.5887 | -0.0023 |
| CIFAR-10 CNN | 25 | +1.56pp | -0.3410 | -0.0023 |
| CIFAR-10 CNN | 30 | +1.96pp | -0.4139 | -0.0034 |

中期轮次进一步支持 CIFAR-10 上的 `spread` 结论：第 20/25/30 轮均同时表现为更高准确率、更低 loss 和更低 MAPE。Fashion-MNIST 上 `spread` 的准确率优势随轮次增加而扩大，但 MAPE 仍高于 `front`。MNIST 上 `spread` 没有体现恢复误差优势。

## 设置 B：num_steal=10, num_img_per_client=10

| 数据集 | 位置方式 | Final Acc | Final Loss | Final MAPE | Best Acc | Best MAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | front | 99.25% | 0.2922 | 0.0942 | 99.32% | 0.0942 |
| MNIST | spread | 99.17% | 0.2940 | 0.1621 | 99.23% | 0.1428 |
| Fashion-MNIST | front | 92.52% | 3.7528 | 0.1540 | 92.88% | 0.1540 |
| Fashion-MNIST | spread | 93.33% | 3.8939 | 0.1305 | 93.33% | 0.0921 |
| CIFAR-10 CNN | front | 79.34% | 7.6068 | 0.0792 | 80.18% | 0.0789 |
| CIFAR-10 CNN | spread | 81.52% | 6.6403 | 0.0927 | 82.48% | 0.0688 |

`spread` 相对 `front` 的差值：

| 数据集 | Final Acc 差值 | Final Loss 差值 | Final MAPE 差值 | Best Acc 差值 | Best MAPE 差值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | -0.08pp | +0.0019 | +0.0678 | -0.08pp | +0.0486 |
| Fashion-MNIST | +0.82pp | +0.1411 | -0.0236 | +0.45pp | -0.0619 |
| CIFAR-10 CNN | +2.18pp | -0.9666 | +0.0135 | +2.30pp | -0.0101 |

设置 B 下，payload 更大，`spread` 在 Fashion-MNIST 和 CIFAR-10 上的分类效用优势更明显。Fashion-MNIST 中 `spread` 同时获得更高最终准确率和更低最终 MAPE。CIFAR-10 中 `spread` 的最终准确率高 2.18pp、loss 低 0.9666，best MAPE 也更低；虽然最终 MAPE 略高 0.0135，但训练过程中曾达到更好的恢复误差。

### 设置 B 的中期轮次

| 数据集 | Round | Acc 差值 | Loss 差值 | MAPE 差值 |
| --- | ---: | ---: | ---: | ---: |
| MNIST | 20 | -0.03pp | +0.0044 | +0.0370 |
| MNIST | 25 | -0.07pp | -0.0077 | +0.0438 |
| MNIST | 30 | -0.02pp | -0.0266 | +0.0507 |
| Fashion-MNIST | 20 | +0.25pp | +0.2982 | -0.0661 |
| Fashion-MNIST | 25 | +0.32pp | +0.2463 | -0.0576 |
| Fashion-MNIST | 30 | +0.67pp | +0.2559 | -0.0497 |
| CIFAR-10 CNN | 20 | +2.16pp | -0.3994 | +0.0027 |
| CIFAR-10 CNN | 25 | +2.30pp | -0.3654 | +0.0078 |
| CIFAR-10 CNN | 30 | +2.56pp | -0.4745 | +0.0070 |

设置 B 的中期结果中，Fashion-MNIST 的 `spread` 表现最均衡：第 20/25/30 轮都同时拥有更高准确率和更低 MAPE。CIFAR-10 的 `spread` 在中期主要体现为更强分类效用，准确率优势达到 2.16-2.56pp，loss 也持续更低；MAPE 中期略高于 `front`，但 best MAPE 更低。MNIST 仍然不体现 `spread` 的恢复优势。

## 重点判断

从分类效用看，`spread` 在 Fashion-MNIST 和 CIFAR-10 上更稳：

- Fashion-MNIST：设置 A/B 中 `spread` 最终准确率分别高 +0.52pp 和 +0.82pp。
- CIFAR-10 CNN：设置 A/B 中 `spread` 最终准确率分别高 +1.04pp 和 +2.18pp，payload 更大时优势更明显。
- MNIST：两种设置下差距很小，设置 A 持平，设置 B 中 `spread` 低 0.08pp。

从恢复误差看，结论依数据集不同：

- MNIST：`front` 更强，两组设置中 `spread` 的最终 MAPE 分别高 0.1148 和 0.0678。
- Fashion-MNIST：设置 A 中 `front` 最终 MAPE 更低；设置 B 中 `spread` 最终 MAPE 低 0.0236，best MAPE 低 0.0619。
- CIFAR-10 CNN：设置 A 中 `spread` 同时拥有更低最终 MAPE 和 best MAPE；设置 B 中 `spread` 的最终 MAPE 略高，但 best MAPE 更低。

综合来看，`spread` 的主要优势是随着 payload 增大能够更好保持模型效用，尤其在 Fashion-MNIST 和 CIFAR-10 上明显。若论文叙述强调“攻击位置分散带来更好的效用-恢复折中”，当前结果最支持 Fashion-MNIST 设置 B 和 CIFAR-10 两组设置；若强调“最低最终 MAPE”，MNIST 仍更支持 `front`。
