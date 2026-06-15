# CIFAR FedAvg 80% Sweep 结果总结

本文档整理 `scripts_10clients/cifar_fedavg80_sweep.sh` 的 CIFAR-10 non-IID FedAvg baseline sweep 结果。实验目标是在之前 `resnet18 + 原始 2-shard non-IID` 约 60% 测试准确率的基础上，保持标准 FedAvg 聚合，尝试通过更合理的 CIFAR 模型、优化器参数、学习率调度和更温和的 non-IID 划分把准确率提升到 80% 左右。

## 实验设置

运行命令：

```bash
GPU_LIST="0 1 2 3 4 5 6 7" JOBS_PER_GPU=1 EPOCHS=300 \
    ./scripts_10clients/cifar_fedavg80_sweep.sh
```

公共配置：

```text
--dataset=cifar
--num_users=10 --frac=1.0 --iid=0
--epochs=300
--local_bs=64 --local_ep=1
--cifar_crop_size=32 --cifar_normalize=1
--optimizer=sgd --momentum=0.9 --weight_decay=0.0005
--lr_scheduler=cosine --min_lr=0.0001
--gama=0 --gama_warmup_epochs=0
--agg_mode=avg --attack_position_mode=spread
--num_steal=5 --num_img_per_client=1
```

扫描范围：

```text
models:
  resnet18_cifar
  wrn28_2
  wrn28_4

learning rates:
  lr=0.03 / 0.05

non-IID settings:
  Dirichlet alpha=0.5 / 1.0
  sorted shards_per_user=5 / 10
```

日志目录：

```text
scripts_10clients/logs_cifar_fedavg80_sweep
```

完整性检查：

- 共 18 个任务。
- 18 个日志均包含 `Results after 300 global rounds of training`。
- 未匹配到 `Traceback`、`RuntimeError`、`CUDA out of memory`、`Failed job`。
- 主指标 `Test Acc` 来自日志末尾的最终全局测试集推理。
- `Avg Train Accuracy` 是代码每轮对各客户端本地 split 的平均推理准确率，不等同于最终全局 test accuracy。

## Non-IID 配置说明

原始 CIFAR non-IID baseline 使用 sorted shards：

```text
num_users=10
shards_per_user=2
num_shards=20
每个 shard=2500 张
每个 client=2 shards=5000 张
```

因为数据先按 label 排序，再切成 shard，所以原始设置通常导致每个 client 只覆盖极少类别，是很强的 label-skew non-IID。之前在这个设置下，FedAvg 测试准确率约停在 60%。

本轮 sweep 包含两类更温和的 non-IID：

```text
1. Dirichlet:
   alpha=0.5 或 1.0
   每个类别按 Dirichlet 比例分给 10 个 clients。
   alpha 越大越接近 IID。

2. Sorted shards:
   shards_per_user=5 或 10
   每个 client 仍拿 5000 张，但由更多、更小的 shard 组成，类别覆盖更广。
```

因此，本轮 80%+ 结果不能直接等同于原始 `20 shards / client 2 shards` 的极端 non-IID 设置。

## 最终结果

按最终 `Test Acc` 从高到低排序：

| Rank | Job | Model | Non-IID | LR | Final Train Acc | Test Acc | Best Client Acc | First >=80 | First >=85 | First >=88 |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | wrn28_4 | Dirichlet alpha=1.0 | 0.03 | 91.08% | **90.22%** | 91.11%@240 | 60 | 80 | 110 |
| 2 | 17 | wrn28_4 | Dirichlet alpha=1.0 | 0.05 | 90.89% | **89.76%** | 91.27%@230 | 60 | 90 | 120 |
| 3 | 5 | resnet18_cifar | Dirichlet alpha=1.0 | 0.05 | 91.05% | **89.67%** | 91.14%@270 | 60 | 90 | 140 |
| 4 | 1 | resnet18_cifar | Dirichlet alpha=1.0 | 0.03 | 91.27% | **89.66%** | 91.27%@300 | 50 | 80 | 120 |
| 5 | 13 | wrn28_2 | Dirichlet alpha=1.0 | 0.05 | 90.06% | **89.20%** | 90.49%@240 | 60 | 100 | 150 |
| 6 | 4 | resnet18_cifar | Dirichlet alpha=0.5 | 0.05 | 90.57% | **89.10%** | 90.83%@270 | 70 | 100 | 150 |
| 7 | 9 | wrn28_2 | Dirichlet alpha=1.0 | 0.03 | 89.70% | **88.90%** | 90.29%@280 | 60 | 100 | 150 |
| 8 | 12 | wrn28_2 | Dirichlet alpha=0.5 | 0.05 | 89.97% | **88.88%** | 89.97%@300 | 80 | 120 | 170 |
| 9 | 8 | wrn28_2 | Dirichlet alpha=0.5 | 0.03 | 89.61% | **88.64%** | 90.07%@270 | 70 | 100 | 160 |
| 10 | 0 | resnet18_cifar | Dirichlet alpha=0.5 | 0.03 | 90.22% | **88.58%** | 90.79%@270 | 70 | 90 | 130 |
| 11 | 3 | resnet18_cifar | shards_per_user=10 | 0.03 | 84.26% | **87.49%** | 84.26%@300 | 160 | - | - |
| 12 | 7 | resnet18_cifar | shards_per_user=10 | 0.05 | 83.48% | **87.25%** | 83.48%@300 | 190 | - | - |
| 13 | 15 | wrn28_2 | shards_per_user=10 | 0.05 | 81.94% | **85.88%** | 81.94%@300 | 270 | - | - |
| 14 | 11 | wrn28_2 | shards_per_user=10 | 0.03 | 81.18% | **84.54%** | 81.26%@290 | 250 | - | - |
| 15 | 2 | resnet18_cifar | shards_per_user=5 | 0.03 | 69.50% | **81.20%** | 70.06%@280 | - | - | - |
| 16 | 6 | resnet18_cifar | shards_per_user=5 | 0.05 | 67.48% | **78.49%** | 68.30%@210 | - | - | - |
| 17 | 10 | wrn28_2 | shards_per_user=5 | 0.03 | 61.02% | **75.36%** | 64.54%@260 | - | - | - |
| 18 | 14 | wrn28_2 | shards_per_user=5 | 0.05 | 57.84% | **72.58%** | 62.24%@240 | - | - | - |

注：`First >=80/85/88` 使用每 10 轮日志中的 `Avg Train Accuracy` 作为收敛代理，不是每轮全局测试准确率。

## 主要结论

1. 本轮 sweep 已经明显超过 80% 目标。最高最终测试准确率为 **90.22%**，对应：

```text
--model=wrn28_4
--lr=0.03
--cifar_noniid_mode=dirichlet
--cifar_dirichlet_alpha=1.0
--epochs=300
```

2. Dirichlet 设置整体最强且稳定。10 个 Dirichlet 任务全部达到 **88.58%-90.22%**，明显高于原始 2-shard 约 60% baseline。

3. `wrn28_4` 成为本轮最优模型，但 `resnet18_cifar` 性价比也很好。`resnet18_cifar + Dirichlet alpha=1.0` 已达到 **89.66%-89.67%**，训练成本低于 `wrn28_4`。

4. `shards_per_user=10` 也能达到 84% 以上，最高 **87.49%**。说明在保持 shard non-IID 口径的前提下，只要每个 client 类别覆盖更广，FedAvg 的准确率会大幅提升。

5. `shards_per_user=5` 明显更难，最高只有 **81.20%**，且 WRN28_2 在该设置下表现更差。该设置不建议作为冲高准确率的主配置。

6. 当前结果主要来自三类变化共同作用：

```text
更温和的 non-IID 划分
更合适的 CIFAR 模型
SGD momentum=0.9 + weight_decay=5e-4 + cosine learning-rate schedule
```

## 收敛轮数判断

当前代码只在训练结束后做一次全局 test，因此无法精确判断“第几轮 Test Acc 达到最高”。但从每 10 轮的客户端平均准确率看：

- Dirichlet 配置大约 **50-80 轮**达到 80% client accuracy。
- Dirichlet 配置大约 **80-120 轮**达到 85% client accuracy。
- Dirichlet 配置大约 **110-170 轮**达到 88% client accuracy。
- 多数 Dirichlet 配置在 **180-220 轮**已经接近最终 client accuracy。
- `300` 轮对 Dirichlet 主配置偏多，后期收益较小。

推荐后续主配置把 `epochs` 改为 **200**：

```bash
python src/federated_main.py \
    --model=wrn28_4 \
    --dataset=cifar \
    --num_users=10 \
    --frac=1.0 \
    --iid=0 \
    --epochs=200 \
    --lr=0.03 \
    --lr_scheduler=cosine \
    --min_lr=0.0001 \
    --momentum=0.9 \
    --weight_decay=0.0005 \
    --local_bs=64 \
    --local_ep=1 \
    --cifar_crop_size=32 \
    --cifar_normalize=1 \
    --cifar_noniid_mode=dirichlet \
    --cifar_dirichlet_alpha=1.0 \
    --gama=0 \
    --gama_warmup_epochs=0 \
    --agg_mode=avg \
    --attack_position_mode=spread \
    --num_steal=5 \
    --num_img_per_client=1
```

如果希望降低训练成本且保持接近 89% 的结果，推荐：

```bash
python src/federated_main.py \
    --model=resnet18_cifar \
    --dataset=cifar \
    --num_users=10 \
    --frac=1.0 \
    --iid=0 \
    --epochs=200 \
    --lr=0.03 \
    --lr_scheduler=cosine \
    --min_lr=0.0001 \
    --momentum=0.9 \
    --weight_decay=0.0005 \
    --local_bs=64 \
    --local_ep=1 \
    --cifar_crop_size=32 \
    --cifar_normalize=1 \
    --cifar_noniid_mode=dirichlet \
    --cifar_dirichlet_alpha=1.0 \
    --gama=0 \
    --gama_warmup_epochs=0 \
    --agg_mode=avg \
    --attack_position_mode=spread \
    --num_steal=5 \
    --num_img_per_client=1
```

## 后续建议

1. 增加 `--test_every`，每 10 或 20 轮保存一次真实 global test accuracy。这样可以准确判断 150/180/200/300 轮之间的 test 收敛差异。

2. 复跑 top 配置换 seed，建议至少验证：

```text
wrn28_4, lr=0.03, Dirichlet alpha=1.0
resnet18_cifar, lr=0.03, Dirichlet alpha=1.0
resnet18_cifar, lr=0.05, Dirichlet alpha=1.0
```

3. 如果论文需要和原始 shard non-IID 保持可比，建议明确区分：

```text
Extreme shard non-IID: shards_per_user=2
Moderate shard non-IID: shards_per_user=5/10
Dirichlet non-IID: alpha=0.5/1.0
```

否则 90% 结果容易被误解为原始极端 2-shard 设置下的 FedAvg 结果。
