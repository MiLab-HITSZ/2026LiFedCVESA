# CIFAR Non-IID ResNet18 Refine 结果总结

本文档整理 CIFAR-10 non-IID 无攻击 ResNet18 精扫结果。实验目标是在 quick sweep 的 CNN 最优结果基础上，改用 `resnet18` 并继续调学习率，观察是否能把最终测试准确率提升到 60% 以上。

## 实验设置

运行配置：

```text
--model=resnet18 --dataset=cifar
--num_users=10 --frac=1.0 --iid=0
--epochs=200
--local_bs=32 --local_ep=1
--gama=0 --gama_warmup_epochs=0
--agg_mode=avg --attack_position_mode=spread
--num_steal=5 --num_img_per_client=1
```

参数扫描范围：

```text
lr=0.005/0.015/0.02/0.03/0.04/0.05
lr_decay=0.995/0.998
```

日志目录：

```text
scripts_10clients/logs_cifar_noniid_resnet18_refine200
```

完整性检查：

- 共 12 个 ResNet18 refine 任务。
- 12 个日志均包含 `Results after 200 global rounds of training`。
- 未匹配到 `Traceback`、`Failed job`、`RuntimeError`、`CUDA out of memory`。
- 主指标 `Test Acc` 来自日志末尾的最终测试集推理。
- `Client-Val Acc` 来自保存的 `_acc.npy`，是每轮所有客户端本地 holdout 平均准确率，不等同于最终全局测试准确率。

## 结果表

按最终 `Test Acc` 从高到低排序：

| Rank | LR | LR Decay | Batch | Local Ep | Test Acc | Client-Val Final | Client-Val Best | Final Loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.015 | 0.998 | 32 | 1 | 60.68% | 51.34% | 55.74% | 13.1605 |
| 2 | 0.015 | 0.995 | 32 | 1 | 60.63% | 52.22% | 53.88% | 13.1617 |
| 3 | 0.04 | 0.995 | 32 | 1 | 60.56% | 52.50% | 54.42% | 12.5645 |
| 4 | 0.005 | 0.998 | 32 | 1 | 59.93% | 50.74% | 51.92% | 13.6260 |
| 5 | 0.02 | 0.998 | 32 | 1 | 59.62% | 49.78% | 56.00% | 13.9332 |
| 6 | 0.05 | 0.995 | 32 | 1 | 59.38% | 48.04% | 54.12% | 13.9873 |
| 7 | 0.03 | 0.998 | 32 | 1 | 59.30% | 49.68% | 54.06% | 14.3280 |
| 8 | 0.02 | 0.995 | 32 | 1 | 59.06% | 50.02% | 55.46% | 15.1397 |
| 9 | 0.05 | 0.998 | 32 | 1 | 59.04% | 48.72% | 50.94% | 13.4154 |
| 10 | 0.04 | 0.998 | 32 | 1 | 58.80% | 51.06% | 51.98% | 12.6865 |
| 11 | 0.005 | 0.995 | 32 | 1 | 58.58% | 49.92% | 51.12% | 13.9017 |
| 12 | 0.03 | 0.995 | 32 | 1 | 57.85% | 48.64% | 55.32% | 15.6671 |

## 主要结论

1. ResNet18 refine 已经达到 60% 以上。最高测试准确率为 60.68%，对应 `lr=0.015, lr_decay=0.998, local_bs=32, local_ep=1, epochs=200`。
2. 有 3 组超过 60%：`lr=0.015, decay=0.998` 为 60.68%；`lr=0.015, decay=0.995` 为 60.63%；`lr=0.04, decay=0.995` 为 60.56%。
3. 相比 v4 CIFAR-CNN non-IID 无攻击基线 47.18%，当前最优结果提升约 +13.50pp。
4. 相比 CNN quick sweep 最优 54.78%，ResNet18 refine 最优提升约 +5.90pp。
5. 当前 12 组结果整体集中在 57.85%-60.68%，说明 `resnet18 + batch=32 + local_ep=1 + 200轮` 比原 CNN 配置稳定很多。
6. `lr=0.015` 最稳，两个 decay 都超过 60%；`lr=0.04, decay=0.995` 也超过 60%，但同一 lr 下 `decay=0.998` 降到 58.80%，说明学习率衰减和初始学习率存在交互。

## 推荐后续配置

如果需要一个主 baseline 配置，建议使用：

```bash
python src/federated_main.py \
    --model=resnet18 \
    --dataset=cifar \
    --num_users=10 \
    --frac=1.0 \
    --iid=0 \
    --epochs=200 \
    --lr=0.015 \
    --lr_decay=0.998 \
    --local_bs=32 \
    --local_ep=1 \
    --gama=0 \
    --gama_warmup_epochs=0 \
    --agg_mode=avg \
    --attack_position_mode=spread \
    --num_steal=5 \
    --num_img_per_client=1
```

如果要进一步稳住 60% 以上，可以复跑 top 3 配置换 seed 验证稳定性：

```text
lr=0.015, lr_decay=0.998
lr=0.015, lr_decay=0.995
lr=0.04,  lr_decay=0.995
```

若这些配置在不同 seed 下仍稳定超过 60%，再基于最稳的无攻击 baseline 加回 CVEA 攻击项和 warm-up，评估攻击条件下的效用损失。
