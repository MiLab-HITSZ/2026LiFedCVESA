#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPOCHS="${EPOCHS:-100}"
GAMA_WARMUP_EPOCHS="${GAMA_WARMUP_EPOCHS:-0}"
IID="${IID:-0}"
source "${SCRIPT_DIR}/common.sh"

RUN_GAMMA="${RUN_GAMMA:-1}"
RUN_NUM_STEAL="${RUN_NUM_STEAL:-1}"
RUN_POSITION="${RUN_POSITION:-1}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/scripts_10clients/logs_rank10_dirichlet05_fmnist_cnn}"
DRY_RUN="${DRY_RUN:-0}"
SLEEP_BETWEEN_JOBS="${SLEEP_BETWEEN_JOBS:-0}"
WAIT_FOR_GPU_FREE="${WAIT_FOR_GPU_FREE:-0}"
MIN_FREE_MB="${MIN_FREE_MB:-8000}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"

CONFIG_TAG="${CONFIG_TAG:-rank10_dirichlet05}"
FMNIST_MODEL="${FMNIST_MODEL:-cnn}"
NONIID_MODE="${NONIID_MODE:-dirichlet}"
SHARDS_PER_USER="${SHARDS_PER_USER:-0}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.5}"
DIRICHLET_MIN_SIZE="${DIRICHLET_MIN_SIZE:-100}"

BASE_ARGS=(
    --model="${FMNIST_MODEL}"
    --dataset=fmnist
    --lr=0.01
    --local_bs=16
    --local_ep=10
    --momentum=0.9
    --weight_decay=0.0005
    --lr_scheduler=cosine
    --min_lr=0.0001
    --noniid_mode="${NONIID_MODE}"
    --shards_per_user="${SHARDS_PER_USER}"
    --dirichlet_alpha="${DIRICHLET_ALPHA}"
    --dirichlet_min_size="${DIRICHLET_MIN_SIZE}"
)

JOBS_NAME=()
JOBS_ARGS=()

add_job() {
    job_name="$1"
    shift

    JOBS_NAME+=("${job_name}")
    JOBS_ARGS+=("$*")
}

add_fmnist_job() {
    job_name="$1"
    shift

    add_job "${job_name}" \
        "${BASE_ARGS[@]}" \
        --result_tag="${CONFIG_TAG}_${job_name}" \
        "$@"
}

add_gamma_jobs() {
    for gama in 0 0.05 0.2 0.5 1.0; do
        add_fmnist_job "gamma_fmnist_${FMNIST_MODEL}_g${gama}" \
            --gama="${gama}" --num_steal=5 --num_img_per_client=1 \
            --agg_mode=segmented --attack_position_mode=spread
    done
}

add_num_steal_jobs() {
    for num_steal in 1 2 3 4 5 10; do
        add_fmnist_job "numsteal_fmnist_${FMNIST_MODEL}_n${num_steal}" \
            --gama=0.5 --num_steal="${num_steal}" --num_img_per_client=1 \
            --agg_mode=segmented --attack_position_mode=spread
    done
}

add_position_jobs() {
    for num_img_per_client in 1 5 10 50; do
        for attack_position_mode in front spread; do
            add_fmnist_job "position_fmnist_${FMNIST_MODEL}_n10_img${num_img_per_client}_${attack_position_mode}" \
                --gama=0.5 --num_steal=10 --num_img_per_client="${num_img_per_client}" \
                --agg_mode=segmented --attack_position_mode="${attack_position_mode}"
        done
    done
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
        job_name="${JOBS_NAME[${job_index}]}"
        job_args="${JOBS_ARGS[${job_index}]}"
        log_path="${LOG_DIR}/${job_index}_${job_name}_gpu${gpu}_slot${gpu_slot}.log"

        echo "==> [GPU ${gpu} slot ${gpu_slot}] Start job ${job_index}/${total_jobs}: ${job_name}"
        wait_for_gpu_free "${gpu}"

        read -r -a args_array <<< "${job_args}"
        if run_fed --gpu="${gpu}" "${args_array[@]}" > "${log_path}" 2>&1; then
            echo "==> [GPU ${gpu} slot ${gpu_slot}] Done job ${job_index}: ${job_name}"
        else
            echo "!! [GPU ${gpu} slot ${gpu_slot}] Failed job ${job_index}: ${job_name}. See ${log_path}" >&2
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

    if [ "${RUN_GAMMA}" = "1" ]; then
        add_gamma_jobs
    fi
    if [ "${RUN_NUM_STEAL}" = "1" ]; then
        add_num_steal_jobs
    fi
    if [ "${RUN_POSITION}" = "1" ]; then
        add_position_jobs
    fi

    read -r -a gpus <<< "${GPU_LIST}"
    if [ "${#gpus[@]}" -eq 0 ]; then
        echo "GPU_LIST is empty." >&2
        exit 1
    fi
    if [ "${#JOBS_NAME[@]}" -eq 0 ]; then
        echo "No jobs selected." >&2
        exit 1
    fi
    if ! [[ "${JOBS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
        echo "JOBS_PER_GPU must be a positive integer." >&2
        exit 1
    fi

    gpu_count="${#gpus[@]}"
    worker_count=$((gpu_count * JOBS_PER_GPU))

    echo "Config: ${CONFIG_TAG}"
    echo "Total Fashion-MNIST CNN jobs: ${#JOBS_NAME[@]}"
    echo "GPUs: ${GPU_LIST}"
    echo "Jobs per GPU: ${JOBS_PER_GPU}"
    echo "Total workers: ${worker_count}"
    echo "Epochs: ${EPOCHS}"
    echo "IID: ${IID}"
    echo "Non-IID: mode=${NONIID_MODE}, alpha=${DIRICHLET_ALPHA}, min_size=${DIRICHLET_MIN_SIZE}"
    echo "Model config: model=${FMNIST_MODEL}, lr=0.01, local_bs=16, local_ep=10"
    echo "Wait for GPU free: ${WAIT_FOR_GPU_FREE}, min free MB: ${MIN_FREE_MB}"
    echo "Logs: ${LOG_DIR}"

    if [ "${DRY_RUN}" = "1" ]; then
        for job_index in "${!JOBS_NAME[@]}"; do
            worker_index=$((job_index % worker_count))
            gpu="${gpus[$((worker_index % gpu_count))]}"
            gpu_slot=$((worker_index / gpu_count))
            echo "DRY_RUN job ${job_index}: GPU ${gpu} slot ${gpu_slot} ${JOBS_NAME[${job_index}]} ${JOBS_ARGS[${job_index}]}"
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
