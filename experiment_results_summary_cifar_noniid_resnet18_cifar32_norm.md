# CIFAR Non-IID ResNet18 32x32 Normalize 结果总结

本文档整理 CIFAR-10 non-IID 无攻击 ResNet18 在 `32x32 + CIFAR Normalize` 设置下的 refine 结果。实验目标是验证：在上一轮 ResNet18 已达到 60% 左右后，打开 CIFAR 标准归一化并保留完整 32x32 输入，是否能继续提升最终测试准确率。

## 实验设置

运行配置：

```text
--model=resnet18 --dataset=cifar
--num_users=10 --frac=1.0 --iid=0
--epochs=200
--local_bs=32 --local_ep=1
--cifar_crop_size=32 --cifar_normalize=1
--gama=0 --gama_warmup_epochs=0
--agg_mode=avg --attack_position_mode=spread
--num_steal=5 --num_img_per_client=1
```

数据增强与归一化：

```text
train: RandomCrop(32, padding=4) + RandomHorizontalFlip + ColorJitter + ToTensor + Normalize(CIFAR_MEAN, CIFAR_STD)
test:  CenterCrop(32) + ToTensor + Normalize(CIFAR_MEAN, CIFAR_STD)
```

参数扫描范围：

```text
lr=0.005/0.015/0.02/0.03/0.04/0.05
lr_decay=0.995/0.998
```

日志目录：

```text
scripts_10clients/logs_cifar_noniid_resnet18_cifar32_norm_refine200
```

完整性检查：

- 共 12 个 ResNet18 任务。
- 12 个日志均包含 `Results after 200 global rounds of training`。
- 未匹配到 `Traceback`、`Failed job`、`RuntimeError`、`CUDA out of memory`。
- 主指标 `Test Acc` 来自日志末尾的最终测试集推理。
- `Train Acc` 来自日志末尾的 `Avg Train Accuracy`。
- `Client-Val Acc` 来自保存的 `_acc.npy`，是每轮所有客户端本地 holdout 平均准确率；当前代码中它与日志中的 `Avg Train Accuracy` 是同一口径，不等同于最终全局测试准确率。

## 结果表

按最终 `Test Acc` 从高到低排序：

| Rank | LR | LR Decay | Batch | Local Ep | Train Acc | Test Acc | Client-Val Final | Client-Val Best | Final Loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.02 | 0.998 | 32 | 1 | 56.34% | 58.71% | 56.34% | 60.96% | 11.6668 |
| 2 | 0.015 | 0.995 | 32 | 1 | 58.44% | 57.87% | 58.44% | 59.48% | 11.6555 |
| 3 | 0.02 | 0.995 | 32 | 1 | 59.06% | 57.35% | 59.06% | 59.16% | 11.5529 |
| 4 | 0.04 | 0.995 | 32 | 1 | 59.04% | 57.18% | 59.04% | 59.04% | 11.0525 |
| 5 | 0.04 | 0.998 | 32 | 1 | 56.20% | 56.70% | 56.20% | 57.88% | 11.3275 |
| 6 | 0.05 | 0.998 | 32 | 1 | 55.82% | 56.20% | 55.82% | 58.14% | 12.0365 |
| 7 | 0.03 | 0.995 | 32 | 1 | 57.42% | 56.18% | 57.42% | 59.30% | 11.8469 |
| 8 | 0.015 | 0.998 | 32 | 1 | 58.42% | 55.23% | 58.42% | 60.74% | 11.6372 |
| 9 | 0.05 | 0.995 | 32 | 1 | 54.40% | 55.22% | 54.40% | 58.14% | 12.1496 |
| 10 | 0.03 | 0.998 | 32 | 1 | 58.08% | 55.08% | 58.08% | 59.66% | 11.1824 |
| 11 | 0.005 | 0.998 | 32 | 1 | 57.70% | 50.30% | 57.70% | 58.08% | 11.5665 |
| 12 | 0.005 | 0.995 | 32 | 1 | 56.26% | 48.53% | 56.26% | 56.78% | 12.0355 |

## 对比结论

1. 这轮 `32x32 + Normalize` 没有继续提升最终测试准确率。最高 `Test Acc` 为 58.71%，低于上一轮 24x24 无 Normalize ResNet18 refine 的最高 60.68%。
2. 相比 v4 CIFAR-CNN non-IID 无攻击基线 47.18%，当前最优 58.71% 仍有 +11.53pp 提升。
3. 相比 CNN quick sweep 最优 54.78%，当前最优 58.71% 仍有 +3.93pp 提升。
4. 需要注意，`Client-Val Best` 最高达到 60.96%，但这是客户端本地 holdout 平均准确率，不是最终全局测试准确率。论文和主结论应优先使用日志中的最终 `Test Acc`。
5. `lr=0.02, lr_decay=0.998` 是本轮最终测试准确率最高的组合，为 58.71%。
6. `lr=0.005` 明显偏低，最终测试准确率只有 48.53%-50.30%，不适合作为后续主配置。
7. 32x32+Normalize 让保存曲线中的 Client-Val 指标整体更好，但最终 test set 泛化没有同步提升。这说明当前瓶颈仍然主要来自极端 non-IID/FedAvg 训练，而不只是输入分辨率或标准化。

## 推荐判断

如果目标是选择当前最强 CIFAR non-IID 无攻击 baseline，建议仍优先使用上一轮 24x24 无 Normalize 的 ResNet18 配置：

```text
--model=resnet18
--epochs=200
--local_bs=32 --local_ep=1
--lr=0.015 --lr_decay=0.998
--cifar_crop_size=24 --cifar_normalize=0
```

该配置上一轮最终测试准确率为 60.68%。

如果继续冲 80%，下一步不建议继续只微调 crop/normalize。更值得尝试的是：

```text
1. FedProx 或 FedAvgM，抑制 non-IID client drift
2. 复跑 60%+ top 配置换 seed，确认稳定性
3. 在稳定 baseline 上再加 CVEA attack warm-up
```

当前证据不支持“仅开启 32x32+Normalize 就能明显提升到 80%”。
