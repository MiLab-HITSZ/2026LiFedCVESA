#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPOCHS="${EPOCHS:-300}"
GAMA_WARMUP_EPOCHS="${GAMA_WARMUP_EPOCHS:-0}"
IID="${IID:-0}"
source "${SCRIPT_DIR}/common.sh"

GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/scripts_10clients/logs_cifar_fedavg80_sweep}"
DRY_RUN="${DRY_RUN:-0}"

CIFAR_CROP_SIZE="${CIFAR_CROP_SIZE:-32}"
CIFAR_NORMALIZE="${CIFAR_NORMALIZE:-1}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
MIN_LR="${MIN_LR:-0.0001}"

JOBS_NAME=()
JOBS_ARGS=()
declare -A SEEN_JOB_ARGS=()

sanitize_float() {
    value="$1"
    echo "${value//./p}"
}

add_job() {
    job_name="$1"
    shift
    job_args="$*"
    if [ -n "${SEEN_JOB_ARGS[${job_args}]:-}" ]; then
        return 0
    fi
    SEEN_JOB_ARGS["${job_args}"]=1
    JOBS_NAME+=("${job_name}")
    JOBS_ARGS+=("${job_args}")
}

add_fedavg_job() {
    model="$1"
    lr="$2"
    local_bs="$3"
    noniid_mode="$4"
    noniid_value="$5"

    lr_tag="$(sanitize_float "${lr}")"
    value_tag="$(sanitize_float "${noniid_value}")"
    tag="fedavg80_${model}_lr${lr_tag}_b${local_bs}_${noniid_mode}${value_tag}"
    name="cifar_${model}_lr${lr_tag}_b${local_bs}_${noniid_mode}${value_tag}"

    common_args=(
        --model="${model}" --dataset=cifar
        --lr="${lr}" --lr_decay=1.0 --lr_scheduler=cosine --min_lr="${MIN_LR}"
        --momentum="${MOMENTUM}" --weight_decay="${WEIGHT_DECAY}"
        --local_bs="${local_bs}" --local_ep=1
        --cifar_crop_size="${CIFAR_CROP_SIZE}" --cifar_normalize="${CIFAR_NORMALIZE}"
        --gama=0 --num_steal=5 --num_img_per_client=1
        --agg_mode=avg --attack_position_mode=spread
        --result_tag="${tag}"
    )

    if [ "${noniid_mode}" = "dirichlet" ]; then
        add_job "${name}" "${common_args[@]}" \
            --cifar_noniid_mode=dirichlet \
            --cifar_dirichlet_alpha="${noniid_value}"
    else
        add_job "${name}" "${common_args[@]}" \
            --cifar_noniid_mode=shards \
            --cifar_shards_per_user="${noniid_value}"
    fi
}

build_jobs() {
    for model in resnet18_cifar wrn28_2; do
        for lr in 0.03 0.05; do
            for alpha in 0.5 1.0; do
                add_fedavg_job "${model}" "${lr}" 64 dirichlet "${alpha}"
            done
            for shards_per_user in 5 10; do
                add_fedavg_job "${model}" "${lr}" 64 shards "${shards_per_user}"
            done
        done
    done

    for lr in 0.03 0.05; do
        add_fedavg_job wrn28_4 "${lr}" 64 dirichlet 1.0
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
        read -r -a args_array <<< "${job_args}"
        if run_fed --gpu="${gpu}" "${args_array[@]}" > "${log_path}" 2>&1; then
            echo "==> [GPU ${gpu} slot ${gpu_slot}] Done job ${job_index}: ${job_name}"
        else
            echo "!! [GPU ${gpu} slot ${gpu_slot}] Failed job ${job_index}: ${job_name}. See ${log_path}" >&2
            return 1
        fi

        job_index=$((job_index + worker_count))
    done
}

main() {
    mkdir -p "${LOG_DIR}"
    build_jobs

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

    echo "Total jobs: ${#JOBS_NAME[@]}"
    echo "GPUs: ${GPU_LIST}"
    echo "Jobs per GPU: ${JOBS_PER_GPU}"
    echo "Epochs: ${EPOCHS}"
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
