# 10 客户端全参与实验脚本

这个目录保存新的实验脚本。核心配置是每轮 10 个客户端全部参与：

```bash
--num_users=10 --frac=1.0
```

也就是替代旧的 `100` 个客户端、每轮采样 `10%` 客户端的设置。

## 目录分类

- `gamma_sweep/`：对应实验一，扫描 `gama=0, 0.05, 0.2, 0.5, 1.0`。
- `num_steal_sweep/`：对应实验二，扫描 `num_steal=1, 2, 3, 4, 5, 10`。
- `position_ablation/`：对应实验三，比较 `front` 与 `spread`，覆盖四组设置：
  `num_steal=10, num_img_per_client=1`、
  `num_steal=10, num_img_per_client=5`、
  `num_steal=10, num_img_per_client=10`、
  `num_steal=10, num_img_per_client=50`。

## 运行方式

运行全部实验：

```bash
bash scripts_10clients/run_all_10clients.sh
```

使用 0-7 号 GPU 并行运行全部实验：

```bash
bash scripts_10clients/run_all_8gpu_parallel.sh
```

如果单个任务显存占用较低，可以在同一张 GPU 上同时跑多个任务：

```bash
bash scripts_10clients/run_all_8gpu_multi_per_gpu.sh
```

只运行某一类实验：

```bash
RUN_GAMMA=1 RUN_NUM_STEAL=0 RUN_POSITION=0 bash scripts_10clients/run_all_10clients.sh
RUN_GAMMA=0 RUN_NUM_STEAL=1 RUN_POSITION=0 bash scripts_10clients/run_all_10clients.sh
RUN_GAMMA=0 RUN_NUM_STEAL=0 RUN_POSITION=1 bash scripts_10clients/run_all_10clients.sh
```

并行脚本也支持同样的分类开关，并且可以指定 GPU 列表：

```bash
GPU_LIST="0 1 2 3 4 5 6 7" RUN_GAMMA=1 RUN_NUM_STEAL=0 RUN_POSITION=0 bash scripts_10clients/run_all_8gpu_parallel.sh
GPU_LIST="0 1 2 3" bash scripts_10clients/run_all_8gpu_parallel.sh
GPU_LIST="0 1 2 3 4 5 6 7" JOBS_PER_GPU=2 bash scripts_10clients/run_all_8gpu_multi_per_gpu.sh
```

正式运行前可以先做 dry run，只打印任务分配，不启动训练：

```bash
DRY_RUN=1 bash scripts_10clients/run_all_8gpu_parallel.sh
DRY_RUN=1 JOBS_PER_GPU=2 bash scripts_10clients/run_all_8gpu_multi_per_gpu.sh
```

运行单个脚本：

```bash
bash scripts_10clients/gamma_sweep/mnist_cnn_gamma.sh
```

常用环境变量覆盖：

```bash
GPU=0 EPOCHS=10 SEED=1 bash scripts_10clients/gamma_sweep/mnist_cnn_gamma.sh
```

新结果文件名会包含 `C[1.0]`，用于和旧的 `C[0.1]` 结果区分。
并行脚本的日志默认保存到 `scripts_10clients/logs/`。
同卡多任务脚本的日志默认保存到 `scripts_10clients/logs_multi_per_gpu/`。
