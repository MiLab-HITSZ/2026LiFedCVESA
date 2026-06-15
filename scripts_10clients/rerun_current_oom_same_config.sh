#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPOCHS="${EPOCHS:-200}"
GAMA_WARMUP_EPOCHS="${GAMA_WARMUP_EPOCHS:-0}"
IID="${IID:-0}"
source "${SCRIPT_DIR}/common.sh"

GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/scripts_10clients/logs_current_oom_rerun}"
DRY_RUN="${DRY_RUN:-0}"
SLEEP_BETWEEN_JOBS="${SLEEP_BETWEEN_JOBS:-30}"
WAIT_FOR_GPU_FREE="${WAIT_FOR_GPU_FREE:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-8000}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
RANK10_EPOCHS="${RANK10_EPOCHS:-100}"
RANK11_EPOCHS="${RANK11_EPOCHS:-200}"

FMNIST_MODEL="${FMNIST_MODEL:-resnet18}"

BASE_OPT_ARGS=(
    --momentum=0.9
    --weight_decay=0.0005
    --lr_scheduler=cosine
    --min_lr=0.0001
)

RANK10_NONIID_ARGS=(
    --noniid_mode=dirichlet
    --shards_per_user=0
    --dirichlet_alpha=0.5
    --dirichlet_min_size=100
)

RANK11_NONIID_ARGS=(
    --noniid_mode=shards
    --shards_per_user=10
    --dirichlet_alpha=0.0
    --dirichlet_min_size=100
)

JOBS_CONFIG=()
JOBS_ORIGINAL_INDEX=()
JOBS_NAME=()
JOBS_ARGS=()

add_job() {
    config_tag="$1"
    original_index="$2"
    job_name="$3"
    shift 3

    JOBS_CONFIG+=("${config_tag}")
    JOBS_ORIGINAL_INDEX+=("${original_index}")
    JOBS_NAME+=("${job_name}")
    JOBS_ARGS+=("$*")
}

add_fmnist_job() {
    config_tag="$1"
    original_index="$2"
    job_name="$3"
    shift 3

    noniid_args=()
    epoch_args=()
    if [ "${config_tag}" = "rank10_dirichlet05" ]; then
        noniid_args=("${RANK10_NONIID_ARGS[@]}")
        epoch_args=(--epochs="${RANK10_EPOCHS}")
    elif [ "${config_tag}" = "rank11_shards10" ]; then
        noniid_args=("${RANK11_NONIID_ARGS[@]}")
        epoch_args=(--epochs="${RANK11_EPOCHS}")
    else
        echo "Unknown config tag: ${config_tag}" >&2
        exit 1
    fi

    add_job "${config_tag}" "${original_index}" "${job_name}" \
        --model="${FMNIST_MODEL}" \
        --dataset=fmnist \
        --lr=0.01 \
        --local_bs=16 \
        --local_ep=10 \
        "${epoch_args[@]}" \
        "${BASE_OPT_ARGS[@]}" \
        "${noniid_args[@]}" \
        --result_tag="${config_tag}_${job_name}" \
        "$@"
}

add_current_oom_jobs() {
    # Current OOM list from experiment_results_summary_v5.md / v6.md.
    # Keep experiment settings unchanged: Fashion-MNIST ResNet18, local_bs=16.
    add_fmnist_job "rank10_dirichlet05" 40 "position_fmnist_${FMNIST_MODEL}_n10_img5_front" \
        --gama=0.5 --num_steal=10 --num_img_per_client=5 \
        --agg_mode=segmented --attack_position_mode=front

    add_fmnist_job "rank11_shards10" 4 "gamma_fmnist_${FMNIST_MODEL}_g0.05" \
        --gama=0.05 --num_steal=5 --num_img_per_client=1 \
        --agg_mode=segmented --attack_position_mode=spread

    add_fmnist_job "rank11_shards10" 7 "gamma_fmnist_${FMNIST_MODEL}_g0.2" \
        --gama=0.2 --num_steal=5 --num_img_per_client=1 \
        --agg_mode=segmented --attack_position_mode=spread

    add_fmnist_job "rank11_shards10" 10 "gamma_fmnist_${FMNIST_MODEL}_g0.5" \
        --gama=0.5 --num_steal=5 --num_img_per_client=1 \
        --agg_mode=segmented --attack_position_mode=spread

    add_fmnist_job "rank11_shards10" 19 "numsteal_fmnist_${FMNIST_MODEL}_n2" \
        --gama=0.5 --num_steal=2 --num_img_per_client=1 \
        --agg_mode=segmented --attack_position_mode=spread

    add_fmnist_job "rank11_shards10" 31 "numsteal_fmnist_${FMNIST_MODEL}_n10" \
        --gama=0.5 --num_steal=10 --num_img_per_client=1 \
        --agg_mode=segmented --attack_position_mode=spread

    add_fmnist_job "rank11_shards10" 34 "position_fmnist_${FMNIST_MODEL}_n10_img1_front" \
        --gama=0.5 --num_steal=10 --num_img_per_client=1 \
        --agg_mode=segmented --attack_position_mode=front
}

gpu_free_mb() {
    gpu="$1"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "999999"
        return 0
    fi
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | awk 'NR==1 {gsub(/ /, "", $1); print $1}'
}

wait_for_gpu_free() {
    gpu="$1"
    if [ "${WAIT_FOR_GPU_FREE}" != "1" ]; then
        return 0
    fi

    while true; do
        free_mb="$(gpu_free_mb "${gpu}")"
        if [ "${free_mb}" -ge "${MIN_FREE_MB}" ]; then
            echo "==> GPU ${gpu} free memory ${free_mb} MB >= ${MIN_FREE_MB} MB"
            return 0
        fi
        echo "==> GPU ${gpu} free memory ${free_mb} MB < ${MIN_FREE_MB} MB; wait ${GPU_POLL_SECONDS}s"
        sleep "${GPU_POLL_SECONDS}"
    done
}

run_worker() {
    gpu="$1"
    gpu_slot="$2"
    worker_index="$3"
    worker_count="$4"
    total_jobs="${#JOBS_NAME[@]}"

    job_index="${worker_index}"
    while [ "${job_index}" -lt "${total_jobs}" ]; do
        config_tag="${JOBS_CONFIG[${job_index}]}"
        original_index="${JOBS_ORIGINAL_INDEX[${job_index}]}"
        job_name="${JOBS_NAME[${job_index}]}"
        job_args="${JOBS_ARGS[${job_index}]}"
        log_path="${LOG_DIR}/${config_tag}_${original_index}_${job_name}_gpu${gpu}_slot${gpu_slot}.log"

        echo "==> [GPU ${gpu} slot ${gpu_slot}] Start OOM rerun ${job_index}/${total_jobs}: ${config_tag} original ${original_index} ${job_name}"
        wait_for_gpu_free "${gpu}"

        read -r -a args_array <<< "${job_args}"
        if run_fed --gpu="${gpu}" "${args_array[@]}" > "${log_path}" 2>&1; then
            echo "==> [GPU ${gpu} slot ${gpu_slot}] Done OOM rerun ${job_index}: ${config_tag} original ${original_index} ${job_name}"
        else
            echo "!! [GPU ${gpu} slot ${gpu_slot}] Failed OOM rerun ${job_index}: ${config_tag} original ${original_index} ${job_name}. See ${log_path}" >&2
            return 1
        fi

        if [ "${SLEEP_BETWEEN_JOBS}" -gt 0 ]; then
            sleep "${SLEEP_BETWEEN_JOBS}"
        fi
        job_index=$((job_index + worker_count))
    done
}

main() {
    mkdir -p "${LOG_DIR}"
    add_current_oom_jobs

    read -r -a gpus <<< "${GPU_LIST}"
    if [ "${#gpus[@]}" -eq 0 ]; then
        echo "GPU_LIST is empty." >&2
        exit 1
    fi
    if ! [[ "${JOBS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
        echo "JOBS_PER_GPU must be a positive integer." >&2
        exit 1
    fi

    gpu_count="${#gpus[@]}"
    worker_count=$((gpu_count * JOBS_PER_GPU))

    echo "Total current OOM rerun jobs: ${#JOBS_NAME[@]}"
    echo "GPUs: ${GPU_LIST}"
    echo "Jobs per GPU: ${JOBS_PER_GPU}"
    echo "Total workers: ${worker_count}"
    echo "Default epochs: ${EPOCHS}"
    echo "Rank10 epochs override: ${RANK10_EPOCHS}"
    echo "Rank11 epochs override: ${RANK11_EPOCHS}"
    echo "IID: ${IID}"
    echo "Fashion-MNIST config: model=${FMNIST_MODEL}, lr=0.01, local_bs=16, local_ep=10"
    echo "Wait for GPU free: ${WAIT_FOR_GPU_FREE}, min free MB: ${MIN_FREE_MB}"
    echo "Logs: ${LOG_DIR}"

    if [ "${DRY_RUN}" = "1" ]; then
        for job_index in "${!JOBS_NAME[@]}"; do
            worker_index=$((job_index % worker_count))
            gpu="${gpus[$((worker_index % gpu_count))]}"
            gpu_slot=$((worker_index / gpu_count))
            echo "DRY_RUN job ${job_index}: GPU ${gpu} slot ${gpu_slot} ${JOBS_CONFIG[${job_index}]} original ${JOBS_ORIGINAL_INDEX[${job_index}]} ${JOBS_NAME[${job_index}]} ${JOBS_ARGS[${job_index}]}"
        done
        exit 0
    fi

    pids=()
    for worker_index in $(seq 0 $((worker_count - 1))); do
        gpu="${gpus[$((worker_index % gpu_count))]}"
        gpu_slot=$((worker_index / gpu_count))
        run_worker "${gpu}" "${gpu_slot}" "${worker_index}" "${worker_count}" &
        pids+=("$!")
    done

    exit_code=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            exit_code=1
        fi
    done

    exit "${exit_code}"
}

main
