# Reproducing v5 Experiments

This directory keeps only the final 10-client Dirichlet non-IID experiment runner used for `experiment_results_summary_v5.md`.

Run a dry check:

```bash
DRY_RUN=1 bash scripts_10clients/run_all_rank10_dirichlet05.sh
```

Run all v5 jobs:

```bash
GPU_LIST="0 1 2 3 4 5 6 7" bash scripts_10clients/run_all_rank10_dirichlet05.sh
```

Useful switches:

```bash
RUN_GAMMA=1 RUN_NUM_STEAL=0 RUN_POSITION=0 bash scripts_10clients/run_all_rank10_dirichlet05.sh
RUN_GAMMA=0 RUN_NUM_STEAL=1 RUN_POSITION=0 bash scripts_10clients/run_all_rank10_dirichlet05.sh
RUN_GAMMA=0 RUN_NUM_STEAL=0 RUN_POSITION=1 bash scripts_10clients/run_all_rank10_dirichlet05.sh
```

Logs are written to `scripts_10clients/logs_rank10_dirichlet05/`, which is ignored by git.
