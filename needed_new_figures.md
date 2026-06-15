# 新版实验图片需求清单

以下图片建议统一放入 `examples/hitbook/chinese/figures/`。文件名尽量使用英文、数字和下划线，避免继续使用带 `[`、`]` 的长实验参数名，后续 LaTeX 引用更稳定。

## 必需图片

1. `fedcvesa_gamma_tradeoff_50r.png`

   用途：新版第五章“攻击强度分析”总览图，对应 FedCVESA.pdf 的 Fig. 3。

   内容：在 `gamma = {0, 0.05, 0.2, 0.5, 1.0}` 下展示 MNIST、Fashion-MNIST、CIFAR-CNN、CIFAR-ResNet18 的最终准确率变化和最终 MAPE。

2. `fedcvesa_recovery_examples_gamma0.5_n5_q1.png`

   用途：新版第五章“定性恢复结果”展示图，对应 FedCVESA.pdf 的 Fig. 4。

   内容：默认代表性配置 `gamma=0.5`、`n=5`、每目标客户端 1 张图、segmented aggregation、dispersed placement。每个数据集上方放原始私有图像，下方放恢复图像。

3. `fedcvesa_target_scale_tradeoff_gamma0.5.png`

   用途：新版第五章“目标客户端数量分析”总览图，对应 FedCVESA.pdf 的 Fig. 5。

   内容：在 `num_steal = {1, 2, 3, 4, 5, 10}`、`gamma=0.5`、每目标客户端 1 张图下，展示 MNIST、Fashion-MNIST、CIFAR-CNN 的最终准确率变化和最终 MAPE。

## 建议补充图片

4. `fedcvesa_mnist_gamma_curves_50r.png`

   用途：如果第五章仍保留当前“每个数据集单独分析 Acc / Loss / MAPE 曲线”的写法，用于替换旧的 MNIST 曲线图。

   内容：50 轮、`local_bs=16`、无 warm-up、全参与配置下，MNIST 的 Acc、Loss、MAPE 随通信轮次变化曲线。

5. `fedcvesa_fmnist_gamma_curves_50r.png`

   用途：替换旧的 Fashion-MNIST 曲线图。

   内容：50 轮、`local_bs=16`、无 warm-up、全参与配置下，Fashion-MNIST 的 Acc、Loss、MAPE 随通信轮次变化曲线。

6. `fedcvesa_cifar_cnn_gamma_curves_50r.png`

   用途：替换旧的 CIFAR-CNN 曲线图。

   内容：50 轮、`local_bs=16`、无 warm-up、全参与配置下，CIFAR-CNN 的 Acc、Loss、MAPE 随通信轮次变化曲线。

7. `fedcvesa_cifar_resnet18_gamma_curves_50r.png`

   用途：新增 CIFAR-ResNet18 攻击强度分析曲线图。

   内容：50 轮、`local_bs=16`、无 warm-up、全参与配置下，CIFAR-ResNet18 的 Acc、Loss、MAPE 随通信轮次变化曲线。

## 可选图片

8. `fedcvesa_placement_ablation_summary.png`

   用途：如果不想只用表格展示“连续布置 vs 分散布置”消融，可以增加一张可视化汇总图。

   内容：设置 A `(n=5, q=1)`、设置 B `(n=5, q=10)`、设置 C `(n=10, q=1)` 下，展示 dispersed 相对 contiguous 的 `Delta Acc` 和 `Delta MAPE`。

9. `fedcvesa_mnist_recovery_gamma1.0.png`

   用途：新版 MNIST 攻击强度扫描中，最终 MAPE 最低为 `gamma=1.0`，如果第五章需要展示 MNIST 推荐配置，可使用此图。

   内容：MNIST 在 `gamma=1.0` 下的原图与恢复图对比。

10. `fedcvesa_fmnist_recovery_gamma1.0.png`

    用途：新版 Fashion-MNIST 攻击强度扫描中，最终 MAPE 最低为 `gamma=1.0`，如果第五章需要展示 Fashion-MNIST 推荐配置，可使用此图。

    内容：Fashion-MNIST 在 `gamma=1.0` 下的原图与恢复图对比。

11. `fedcvesa_cifar_cnn_recovery_gamma0.2.png`

    用途：新版 CIFAR-CNN 攻击强度扫描中，最终 MAPE 最低为 `gamma=0.2`，如果第五章需要展示 CIFAR-CNN 推荐配置，可使用此图。

    内容：CIFAR-CNN 在 `gamma=0.2` 下的原图与恢复图对比。

12. `fedcvesa_cifar_resnet18_recovery_gamma0.2.png`

    用途：新版 CIFAR-ResNet18 攻击强度扫描中，最终 MAPE 最低为 `gamma=0.2`，如果第五章需要展示 CIFAR-ResNet18 推荐配置，可使用此图。

    内容：CIFAR-ResNet18 在 `gamma=0.2` 下的原图与恢复图对比。

## 旧图片替换提醒

当前 `examples/hitbook/chinese/figures/` 中这些图片属于旧实验口径，后续改第五章时不建议继续引用：

- `mnist_acc_by_gamma.png`
- `mnist_loss_by_gamma.png`
- `mnist_mape_by_gamma.png`
- `fmnist_acc_by_gamma.png`
- `fmnist_loss_by_gamma.png`
- `fmnist_mape_by_gamma.png`
- `cifar_acc_by_gamma.png`
- `cifar_loss_by_gamma.png`
- `cifar_mape_by_gamma.png`
- `mnist_cnn_200_C[0.1]_iid[1]_E[10]_B[10]_Gama[0.2]_numSteal[5]_final_comparison.png`
- `fmnist_cnn_200_C[0.1]_iid[1]_E[10]_B[10]_Gama[0.05]_numSteal[5]_final_comparison.png`
- `cifar_cnn_200_C[0.1]_iid[1]_E[5]_B[50]_Gama[0.5]_numSteal[5]_final_comparison.png`

## 最小交付建议

如果只想先完成新版论文正文，最少准备前三张：

- `fedcvesa_gamma_tradeoff_50r.png`
- `fedcvesa_recovery_examples_gamma0.5_n5_q1.png`
- `fedcvesa_target_scale_tradeoff_gamma0.5.png`

位置消融结果可以先用表格呈现，不一定必须画图。
