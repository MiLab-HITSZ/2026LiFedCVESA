#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPOCHS="${EPOCHS:-200}"
GAMA_WARMUP_EPOCHS="${GAMA_WARMUP_EPOCHS:-0}"
IID="${IID:-0}"
source "${SCRIPT_DIR}/common.sh"

GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/scripts_10clients/logs_rank11_shards10_fmnist_rerun}"
DRY_RUN="${DRY_RUN:-0}"
SLEEP_BETWEEN_JOBS="${SLEEP_BETWEEN_JOBS:-20}"

CONFIG_TAG="${CONFIG_TAG:-rank11_shards10}"
FMNIST_MODEL="${FMNIST_MODEL:-resnet18}"

RUN_GAMMA="${RUN_GAMMA:-1}"
RUN_NUM_STEAL="${RUN_NUM_STEAL:-1}"
RUN_POSITION="${RUN_POSITION:-1}"

COMMON_OPT_ARGS=(
    --momentum=0.9
    --weight_decay=0.0005
    --lr_scheduler=cosine
    --min_lr=0.0001
    --noniid_mode=shards
    --shards_per_user=10
    --dirichlet_alpha=0.0
    --dirichlet_min_size=100
)

JOBS_NAME=()
JOBS_ARGS=()
JOBS_ORIGINAL_INDEX=()

add_job() {
    original_index="$1"
    job_name="$2"
    shift 2
    JOBS_ORIGINAL_INDEX+=("${original_index}")
    JOBS_NAME+=("${job_name}")
    JOBS_ARGS+=("$*")
}

add_fmnist_job() {
    original_index="$1"
    job_name="$2"
    shift 2

    add_job "${original_index}" "${job_name}" \
        --model="${FMNIST_MODEL}" \
        --dataset=fmnist \
        --lr=0.01 \
        --local_bs=16 \
        --local_ep=10 \
        "${COMMON_OPT_ARGS[@]}" \
        --result_tag="${CONFIG_TAG}_${job_name}" \
        "$@"
}

add_gamma_jobs() {
    original_index=1
    for gama in 0 0.05 0.2 0.5 1.0; do
        add_fmnist_job "${original_index}" "gamma_fmnist_${FMNIST_MODEL}_g${gama}" \
            --gama="${gama}" --num_steal=5 --num_img_per_client=1 \
            --agg_mode=segmented --attack_position_mode=spread
        original_index=$((original_index + 3))
    done
}

add_num_steal_jobs() {
    original_index=16
    for num_steal in 1 2 3 4 5 10; do
        add_fmnist_job "${original_index}" "numsteal_fmnist_${FMNIST_MODEL}_n${num_steal}" \
            --gama=0.5 --num_steal="${num_steal}" --num_img_per_client=1 \
            --agg_mode=segmented --attack_position_mode=spread
        original_index=$((original_index + 3))
    done
}

add_position_jobs() {
    original_index=34
    for num_img_per_client in 1 5 10 50; do
        for attack_position_mode in front spread; do
            add_fmnist_job "${original_index}" "position_fmnist_${FMNIST_MODEL}_n10_img${num_img_per_client}_${attack_position_mode}" \
                --gama=0.5 --num_steal=10 --num_img_per_client="${num_img_per_client}" \
                --agg_mode=segmented --attack_position_mode="${attack_position_mode}"
            original_index=$((original_index + 3))
        done
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
        original_index="${JOBS_ORIGINAL_INDEX[${job_index}]}"
        job_name="${JOBS_NAME[${job_index}]}"
        job_args="${JOBS_ARGS[${job_index}]}"
        log_path="${LOG_DIR}/${original_index}_${job_name}_gpu${gpu}_slot${gpu_slot}.log"

        echo "==> [GPU ${gpu} slot ${gpu_slot}] Start job ${job_index}/${total_jobs} original ${original_index}: ${job_name}"
        read -r -a args_array <<< "${job_args}"
        if run_fed --gpu="${gpu}" "${args_array[@]}" > "${log_path}" 2>&1; then
            echo "==> [GPU ${gpu} slot ${gpu_slot}] Done job ${job_index} original ${original_index}: ${job_name}"
        else
            echo "!! [GPU ${gpu} slot ${gpu_slot}] Failed job ${job_index} original ${original_index}: ${job_name}. See ${log_path}" >&2
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
    echo "Total Fashion-MNIST jobs: ${#JOBS_NAME[@]}"
    echo "GPUs: ${GPU_LIST}"
    echo "Jobs per GPU: ${JOBS_PER_GPU}"
    echo "Total workers: ${worker_count}"
    echo "Epochs: ${EPOCHS}"
    echo "IID: ${IID}"
    echo "Fashion-MNIST config: model=${FMNIST_MODEL}, lr=0.01, local_bs=16, local_ep=10"
    echo "Logs: ${LOG_DIR}"

    if [ "${DRY_RUN}" = "1" ]; then
        for job_index in "${!JOBS_NAME[@]}"; do
            worker_index=$((job_index % worker_count))
            gpu="${gpus[$((worker_index % gpu_count))]}"
            gpu_slot=$((worker_index / gpu_count))
            echo "DRY_RUN job ${job_index} original ${JOBS_ORIGINAL_INDEX[${job_index}]}: GPU ${gpu} slot ${gpu_slot} ${JOBS_NAME[${job_index}]} ${JOBS_ARGS[${job_index}]}"
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
