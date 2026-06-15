#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPOCHS="${EPOCHS:-200}"
GAMA_WARMUP_EPOCHS="${GAMA_WARMUP_EPOCHS:-0}"
IID="${IID:-0}"
source "${SCRIPT_DIR}/common.sh"

MODEL="${MODEL:-resnet18}"
CIFAR_CROP_SIZE="${CIFAR_CROP_SIZE:-32}"
CIFAR_NORMALIZE="${CIFAR_NORMALIZE:-1}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/scripts_10clients/logs_cifar_noniid_baseline_sweep}"
DRY_RUN="${DRY_RUN:-0}"
SWEEP_MODE="${SWEEP_MODE:-quick}"

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

add_cifar_baseline_job() {
    lr="$1"
    local_bs="$2"
    local_ep="$3"
    lr_decay="$4"

    lr_tag="$(sanitize_float "${lr}")"
    decay_tag="$(sanitize_float "${lr_decay}")"
    tag="tune_cifar_noniid_${MODEL}_lr${lr_tag}_b${local_bs}_e${local_ep}_d${decay_tag}_c${CIFAR_CROP_SIZE}_n${CIFAR_NORMALIZE}"
    name="cifar_${MODEL}_lr${lr_tag}_b${local_bs}_e${local_ep}_d${decay_tag}_c${CIFAR_CROP_SIZE}_n${CIFAR_NORMALIZE}"

    add_job "${name}" \
        --model="${MODEL}" --dataset=cifar \
        --lr="${lr}" --lr_decay="${lr_decay}" \
        --local_bs="${local_bs}" --local_ep="${local_ep}" \
        --cifar_crop_size="${CIFAR_CROP_SIZE}" --cifar_normalize="${CIFAR_NORMALIZE}" \
        --gama=0 --num_steal=5 --num_img_per_client=1 \
        --agg_mode=avg --attack_position_mode=spread \
        --result_tag="${tag}"
}

add_quick_jobs() {
    for local_bs in 32 64 128; do
        for lr in 0.03 0.05 0.08; do
            add_cifar_baseline_job "${lr}" "${local_bs}" 1 0.995
        done
    done

    for lr in 0.03 0.05 0.08; do
        add_cifar_baseline_job "${lr}" 64 2 0.995
    done

    add_cifar_baseline_job 0.05 64 1 0.990
    add_cifar_baseline_job 0.05 64 1 0.998
}

add_full_jobs() {
    for local_ep in 1 2; do
        for local_bs in 32 64 128; do
            for lr in 0.02 0.03 0.05 0.08 0.10; do
                add_cifar_baseline_job "${lr}" "${local_bs}" "${local_ep}" 0.995
            done
        done
    done

    for lr_decay in 0.990 0.998; do
        for lr in 0.03 0.05 0.08; do
            add_cifar_baseline_job "${lr}" 64 1 "${lr_decay}"
            add_cifar_baseline_job "${lr}" 64 2 "${lr_decay}"
        done
    done
}

add_refine_jobs() {
    for lr_decay in 0.995 0.998; do
        for lr in 0.005 0.015 0.02 0.03 0.04 0.05; do
            add_cifar_baseline_job "${lr}" 32 1 "${lr_decay}"
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

    case "${SWEEP_MODE}" in
        quick)
            add_quick_jobs
            ;;
        full)
            add_full_jobs
            ;;
        refine)
            add_refine_jobs
            ;;
        *)
            echo "SWEEP_MODE must be quick, full, or refine." >&2
            exit 1
            ;;
    esac

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

    echo "Sweep mode: ${SWEEP_MODE}"
    echo "Model: ${MODEL}"
    echo "CIFAR crop size: ${CIFAR_CROP_SIZE}"
    echo "CIFAR normalize: ${CIFAR_NORMALIZE}"
    echo "Total jobs: ${#JOBS_NAME[@]}"
    echo "GPUs: ${GPU_LIST}"
    echo "Jobs per GPU: ${JOBS_PER_GPU}"
    echo "Total workers: ${worker_count}"
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
