# CIFAR Non-IID Baseline Quick Sweep 结果总结

本文档整理 `scripts_10clients/cifar_noniid_baseline_sweep.sh` 的 `quick` 模式结果。实验目标是先调 CIFAR-10 non-IID 的无攻击分类基线，验证是否能把 v4 中约 47% 的准确率提高到 60% 以上。

## 实验设置

运行配置：

```text
--model=cnn --dataset=cifar
--num_users=10 --frac=1.0 --iid=0
--epochs=100
--gama=0 --gama_warmup_epochs=0
--agg_mode=avg --attack_position_mode=spread
--num_steal=5 --num_img_per_client=1
```

参数扫描范围：

```text
local_ep=1, lr_decay=0.995:
  local_bs=32/64/128, lr=0.03/0.05/0.08

local_bs=64, local_ep=2, lr_decay=0.995:
  lr=0.03/0.05/0.08

local_bs=64, local_ep=1, lr=0.05:
  lr_decay=0.990/0.998
```

日志目录：

```text
scripts_10clients/logs_cifar_noniid_baseline_sweep_quick100
```

完整性检查：

- 共 14 个 quick sweep 任务。
- 14 个日志均包含 `Results after 100 global rounds of training`。
- 未匹配到 `Traceback`、`Failed job`、`RuntimeError`、`CUDA out of memory`。
- 主指标 `Test Acc` 来自日志末尾的最终测试集推理。
- `Client-Val Acc` 来自保存的 `_acc.npy`，是每轮所有客户端本地 holdout 平均准确率，不等同于最终全局测试准确率。

## 结果表

按最终 `Test Acc` 从高到低排序：

| Rank | LR | LR Decay | Batch | Local Ep | Test Acc | Client-Val Final | Client-Val Best | Final Loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.03 | 0.995 | 32 | 1 | 54.78% | 46.18% | 46.18% | 14.3614 |
| 2 | 0.05 | 0.995 | 32 | 1 | 54.46% | 44.98% | 45.82% | 14.5167 |
| 3 | 0.08 | 0.995 | 32 | 1 | 54.06% | 44.86% | 45.78% | 14.7960 |
| 4 | 0.05 | 0.995 | 64 | 2 | 53.30% | 46.14% | 47.16% | 13.7256 |
| 5 | 0.08 | 0.995 | 64 | 2 | 52.26% | 42.60% | 48.84% | 14.7871 |
| 6 | 0.03 | 0.995 | 64 | 2 | 51.85% | 44.94% | 47.52% | 13.9828 |
| 7 | 0.05 | 0.990 | 64 | 1 | 51.68% | 42.52% | 44.26% | 15.2341 |
| 8 | 0.03 | 0.995 | 64 | 1 | 51.48% | 42.40% | 45.00% | 15.4294 |
| 9 | 0.05 | 0.998 | 64 | 1 | 50.72% | 41.66% | 45.72% | 15.4876 |
| 10 | 0.05 | 0.995 | 64 | 1 | 49.71% | 43.68% | 43.68% | 14.8060 |
| 11 | 0.08 | 0.995 | 64 | 1 | 46.73% | 41.46% | 42.32% | 15.1006 |
| 12 | 0.05 | 0.995 | 128 | 1 | 44.37% | 36.36% | 39.10% | 16.9681 |
| 13 | 0.08 | 0.995 | 128 | 1 | 42.19% | 35.24% | 37.40% | 17.0263 |
| 14 | 0.03 | 0.995 | 128 | 1 | 40.80% | 38.12% | 38.12% | 15.8473 |

## 主要结论

1. quick sweep 已经明显优于 v4 的 CIFAR non-IID 无攻击基线。v4 中 `cifar_cnn_50_C[1.0]_iid[0]_E[5]_B[16]_Gama[0.0]` 最终准确率为 47.18%；本轮最高为 54.78%，提升约 +7.60pp。
2. 当前还没有达到 60%。最好的三组全部是 `local_bs=32, local_ep=1, lr_decay=0.995`，说明降低本地 epoch 和使用较小 batch 对这个极端 shard non-IID 设置更有利。
3. `local_bs=128` 明显不适合当前配置，测试准确率只有 40.80%-44.37%。
4. `local_ep=2` 在 `batch=64` 下比 `local_ep=1` 更好，例如 `lr=0.05,batch=64` 从 49.71% 提升到 53.30%；但仍低于 `batch=32,local_ep=1` 的前三组。
5. `lr=0.03/0.05/0.08` 在 `batch=32,local_ep=1` 下差距很小，最高是 `lr=0.03`，为 54.78%。

## 后续建议

下一步建议围绕当前最优区域精扫，而不是直接跑 full：

```text
batch=32
local_ep=1
lr=0.015/0.02/0.03/0.04/0.05
lr_decay=0.995/0.998
epochs=200
```

如果目标是冲 60% 以上，优先测试 200 轮，因为当前 100 轮 best 已经接近 55%，但仍不足以说明收敛上限。若 200 轮仍停在 55%-57%，再考虑放宽 CIFAR non-IID shard 强度或加入更适合 non-IID 的聚合策略。
