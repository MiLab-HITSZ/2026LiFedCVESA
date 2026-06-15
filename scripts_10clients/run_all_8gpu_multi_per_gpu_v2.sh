#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPOCHS="${EPOCHS:-50}"
GAMA_WARMUP_EPOCHS="${GAMA_WARMUP_EPOCHS:-0}"
IID="${IID:-1}"
source "${SCRIPT_DIR}/common.sh"

RUN_GAMMA="${RUN_GAMMA:-1}"
RUN_NUM_STEAL="${RUN_NUM_STEAL:-1}"
RUN_POSITION="${RUN_POSITION:-1}"
GPU_LIST="${GPU_LIST:-0 1 2 4 5 6}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/scripts_10clients/logs_multi_per_gpu_v2}"
DRY_RUN="${DRY_RUN:-0}"
AUTO_GPU="${AUTO_GPU:-1}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-7100}"
GPU_RETRY_SECONDS="${GPU_RETRY_SECONDS:-10}"
GPU_WAIT_TIMEOUT_SECONDS="${GPU_WAIT_TIMEOUT_SECONDS:-0}"
MAX_DYNAMIC_JOBS_PER_GPU="${MAX_DYNAMIC_JOBS_PER_GPU:-${JOBS_PER_GPU}}"
RUN_ID="${RUN_ID:-$$}"
GPU_STATE_DIR="${GPU_STATE_DIR:-${LOG_DIR}/gpu_state_${RUN_ID}}"

JOBS_NAME=()
JOBS_ARGS=()
declare -A SEEN_JOB_ARGS=()

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

add_gamma_jobs() {
    for gama in 0 0.05 0.2 0.5 1.0; do
        add_job "gamma_mnist_cnn_g${gama}" \
            --model=cnn --dataset=mnist --lr=0.01 --local_bs=16 --local_ep=10 \
            --gama="${gama}" --num_steal=5 --num_img_per_client=1 \
            --agg_mode=segmented --attack_position_mode=spread
        add_job "gamma_fmnist_resnet18_g${gama}" \
            --model=resnet18 --dataset=fmnist --lr=0.01 --local_bs=16 --local_ep=10 \
            --gama="${gama}" --num_steal=5 --num_img_per_client=1 \
            --agg_mode=segmented --attack_position_mode=spread
        add_job "gamma_cifar_resnet18_g${gama}" \
            --model=resnet18 --dataset=cifar --lr=0.05 --local_bs=16 --local_ep=5 \
            --gama="${gama}" --num_steal=5 --num_img_per_client=1 \
            --agg_mode=segmented --attack_position_mode=spread
    done
}

add_num_steal_jobs() {
    for num_steal in 1 2 3 4 5 10; do
        add_job "numsteal_mnist_cnn_n${num_steal}" \
            --model=cnn --dataset=mnist --lr=0.01 --local_bs=16 --local_ep=10 \
            --gama=0.5 --num_steal="${num_steal}" --num_img_per_client=1 \
            --agg_mode=segmented --attack_position_mode=spread
        add_job "numsteal_fmnist_resnet18_n${num_steal}" \
            --model=resnet18 --dataset=fmnist --lr=0.01 --local_bs=16 --local_ep=10 \
            --gama=0.5 --num_steal="${num_steal}" --num_img_per_client=1 \
            --agg_mode=segmented --attack_position_mode=spread
        add_job "numsteal_cifar_resnet18_n${num_steal}" \
            --model=resnet18 --dataset=cifar --lr=0.05 --local_bs=16 --local_ep=5 \
            --gama=0.5 --num_steal="${num_steal}" --num_img_per_client=1 \
            --agg_mode=segmented --attack_position_mode=spread
    done
}

add_position_jobs() {
    for setting in "10 50"; do
        read -r num_steal num_img_per_client <<< "${setting}"
        for attack_position_mode in front spread; do
            add_job "position_mnist_cnn_n${num_steal}_img${num_img_per_client}_${attack_position_mode}" \
                --model=cnn --dataset=mnist --lr=0.01 --local_bs=16 --local_ep=10 \
                --gama=0.5 --num_steal="${num_steal}" --num_img_per_client="${num_img_per_client}" \
                --agg_mode=segmented --attack_position_mode="${attack_position_mode}"
            add_job "position_fmnist_resnet18_n${num_steal}_img${num_img_per_client}_${attack_position_mode}" \
                --model=resnet18 --dataset=fmnist --lr=0.01 --local_bs=16 --local_ep=10 \
                --gama=0.5 --num_steal="${num_steal}" --num_img_per_client="${num_img_per_client}" \
                --agg_mode=segmented --attack_position_mode="${attack_position_mode}"
            add_job "position_cifar_resnet18_n${num_steal}_img${num_img_per_client}_${attack_position_mode}" \
                --model=resnet18 --dataset=cifar --lr=0.05 --local_bs=16 --local_ep=5 \
                --gama=0.5 --num_steal="${num_steal}" --num_img_per_client="${num_img_per_client}" \
                --agg_mode=segmented --attack_position_mode="${attack_position_mode}"
        done
    done
}

get_gpu_free_mem_mb() {
    gpu="$1"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo 999999
        return 0
    fi

    free_mem="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')"
    if ! [[ "${free_mem}" =~ ^[0-9]+$ ]]; then
        echo 0
        return 0
    fi
    echo "${free_mem}"
}

acquire_gpu_lock() {
    lock_dir="${GPU_STATE_DIR}/lock"
    while ! mkdir "${lock_dir}" 2>/dev/null; do
        sleep 0.2
    done
}

release_gpu_lock() {
    rmdir "${GPU_STATE_DIR}/lock" 2>/dev/null || true
}

reserved_gpu_count() {
    gpu="$1"
    count=0
    for reservation_file in "${GPU_STATE_DIR}"/reserve_*; do
        if [ ! -e "${reservation_file}" ]; then
            continue
        fi
        reserved_gpu="$(cat "${reservation_file}" 2>/dev/null || true)"
        if [ "${reserved_gpu}" = "${gpu}" ]; then
            count=$((count + 1))
        fi
    done
    echo "${count}"
}

try_select_gpu_locked() {
    preferred_gpu="$1"
    best_gpu=""
    best_free=-1

    for candidate_gpu in "${gpus[@]}"; do
        free_mem="$(get_gpu_free_mem_mb "${candidate_gpu}")"
        reserved_count="$(reserved_gpu_count "${candidate_gpu}")"
        if [ "${free_mem}" -lt "${MIN_FREE_MEM_MB}" ]; then
            continue
        fi
        if [ "${reserved_count}" -ge "${MAX_DYNAMIC_JOBS_PER_GPU}" ]; then
            continue
        fi
        if [ "${candidate_gpu}" = "${preferred_gpu}" ]; then
            echo "${candidate_gpu}"
            return 0
        fi
        if [ "${free_mem}" -gt "${best_free}" ]; then
            best_free="${free_mem}"
            best_gpu="${candidate_gpu}"
        fi
    done

    echo "${best_gpu}"
}

select_and_reserve_gpu() {
    preferred_gpu="$1"
    reservation_id="$2"

    if [ "${AUTO_GPU}" != "1" ]; then
        echo "${preferred_gpu}"
        return 0
    fi

    mkdir -p "${GPU_STATE_DIR}"
    start_time="$(date +%s)"
    while true; do
        acquire_gpu_lock
        selected_gpu="$(try_select_gpu_locked "${preferred_gpu}")"
        if [ -n "${selected_gpu}" ]; then
            echo "${selected_gpu}" > "${GPU_STATE_DIR}/reserve_${reservation_id}"
            release_gpu_lock
            echo "${selected_gpu}"
            return 0
        fi
        release_gpu_lock

        now="$(date +%s)"
        elapsed=$((now - start_time))
        if [ "${GPU_WAIT_TIMEOUT_SECONDS}" -gt 0 ] && [ "${elapsed}" -ge "${GPU_WAIT_TIMEOUT_SECONDS}" ]; then
            echo "Timed out waiting for GPU with at least ${MIN_FREE_MEM_MB} MB free memory." >&2
            return 1
        fi
        echo "No GPU has at least ${MIN_FREE_MEM_MB} MB free memory and an open dynamic slot. Retrying in ${GPU_RETRY_SECONDS}s..." >&2
        sleep "${GPU_RETRY_SECONDS}"
    done
}

release_gpu_reservation() {
    reservation_id="$1"
    if [ "${AUTO_GPU}" = "1" ]; then
        acquire_gpu_lock
        rm -f "${GPU_STATE_DIR}/reserve_${reservation_id}"
        release_gpu_lock
    fi
}

run_worker() {
    preferred_gpu="$1"
    gpu_slot="$2"
    worker_index="$3"
    worker_count="$4"
    total_jobs="${#JOBS_NAME[@]}"

    job_index="${worker_index}"
    while [ "${job_index}" -lt "${total_jobs}" ]; do
        job_name="${JOBS_NAME[${job_index}]}"
        job_args="${JOBS_ARGS[${job_index}]}"
        reservation_id="${worker_index}_${job_index}"
        gpu="$(select_and_reserve_gpu "${preferred_gpu}" "${reservation_id}")"
        log_path="${LOG_DIR}/${job_index}_${job_name}_gpu${gpu}_slot${gpu_slot}.log"

        if [ "${gpu}" != "${preferred_gpu}" ]; then
            echo "==> [GPU ${preferred_gpu} slot ${gpu_slot}] Redirect job ${job_index}/${total_jobs}: ${job_name} to GPU ${gpu}"
        fi
        echo "==> [GPU ${gpu} slot ${gpu_slot}] Start job ${job_index}/${total_jobs}: ${job_name}"
        read -r -a args_array <<< "${job_args}"
        if run_fed --gpu="${gpu}" "${args_array[@]}" > "${log_path}" 2>&1; then
            echo "==> [GPU ${gpu} slot ${gpu_slot}] Done job ${job_index}: ${job_name}"
            release_gpu_reservation "${reservation_id}"
        else
            echo "!! [GPU ${gpu} slot ${gpu_slot}] Failed job ${job_index}: ${job_name}. See ${log_path}" >&2
            release_gpu_reservation "${reservation_id}"
            return 1
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
    if ! [[ "${MAX_DYNAMIC_JOBS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
        echo "MAX_DYNAMIC_JOBS_PER_GPU must be a positive integer." >&2
        exit 1
    fi
    if ! [[ "${MIN_FREE_MEM_MB}" =~ ^[0-9]+$ ]]; then
        echo "MIN_FREE_MEM_MB must be a non-negative integer." >&2
        exit 1
    fi
    if ! [[ "${GPU_RETRY_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "GPU_RETRY_SECONDS must be a positive integer." >&2
        exit 1
    fi
    if ! [[ "${GPU_WAIT_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]]; then
        echo "GPU_WAIT_TIMEOUT_SECONDS must be a non-negative integer." >&2
        exit 1
    fi

    gpu_count="${#gpus[@]}"
    worker_count=$((gpu_count * JOBS_PER_GPU))

    echo "Total jobs: ${#JOBS_NAME[@]}"
    echo "GPUs: ${GPU_LIST}"
    echo "Jobs per GPU: ${JOBS_PER_GPU}"
    echo "Total workers: ${worker_count}"
    echo "IID: ${IID}"
    echo "Logs: ${LOG_DIR}"
    echo "Auto GPU: ${AUTO_GPU}"
    if [ "${AUTO_GPU}" = "1" ]; then
        echo "Min free GPU memory: ${MIN_FREE_MEM_MB} MB"
        echo "Max dynamic jobs per GPU: ${MAX_DYNAMIC_JOBS_PER_GPU}"
        echo "GPU retry seconds: ${GPU_RETRY_SECONDS}"
        echo "GPU wait timeout seconds: ${GPU_WAIT_TIMEOUT_SECONDS}"
    fi

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
