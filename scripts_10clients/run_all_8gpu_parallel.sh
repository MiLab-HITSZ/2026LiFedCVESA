#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPOCHS="${EPOCHS:-50}"
GAMA_WARMUP_EPOCHS="${GAMA_WARMUP_EPOCHS:-0}"
source "${SCRIPT_DIR}/common.sh"

RUN_GAMMA="${RUN_GAMMA:-1}"
RUN_NUM_STEAL="${RUN_NUM_STEAL:-1}"
RUN_POSITION="${RUN_POSITION:-1}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/scripts_10clients/logs}"
DRY_RUN="${DRY_RUN:-0}"

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
        add_job "gamma_fmnist_cnn_g${gama}" \
            --model=cnn --dataset=fmnist --lr=0.01 --local_bs=16 --local_ep=10 \
            --gama="${gama}" --num_steal=5 --num_img_per_client=1 \
            --agg_mode=segmented --attack_position_mode=spread
        add_job "gamma_cifar_cnn_g${gama}" \
            --model=cnn --dataset=cifar --lr=0.15 --local_bs=16 --local_ep=5 \
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
        add_job "numsteal_fmnist_cnn_n${num_steal}" \
            --model=cnn --dataset=fmnist --lr=0.01 --local_bs=16 --local_ep=10 \
            --gama=0.5 --num_steal="${num_steal}" --num_img_per_client=1 \
            --agg_mode=segmented --attack_position_mode=spread
        add_job "numsteal_cifar_cnn_n${num_steal}" \
            --model=cnn --dataset=cifar --lr=0.15 --local_bs=16 --local_ep=5 \
            --gama=0.5 --num_steal="${num_steal}" --num_img_per_client=1 \
            --agg_mode=segmented --attack_position_mode=spread
    done
}

add_position_jobs() {
    for setting in "10 1" "10 5" "10 10" "10 50"; do
        read -r num_steal num_img_per_client <<< "${setting}"
        for attack_position_mode in front spread; do
            add_job "position_mnist_cnn_n${num_steal}_img${num_img_per_client}_${attack_position_mode}" \
                --model=cnn --dataset=mnist --lr=0.01 --local_bs=16 --local_ep=10 \
                --gama=0.5 --num_steal="${num_steal}" --num_img_per_client="${num_img_per_client}" \
                --agg_mode=segmented --attack_position_mode="${attack_position_mode}"
            add_job "position_fmnist_cnn_n${num_steal}_img${num_img_per_client}_${attack_position_mode}" \
                --model=cnn --dataset=fmnist --lr=0.01 --local_bs=16 --local_ep=10 \
                --gama=0.5 --num_steal="${num_steal}" --num_img_per_client="${num_img_per_client}" \
                --agg_mode=segmented --attack_position_mode="${attack_position_mode}"
            add_job "position_cifar_cnn_n${num_steal}_img${num_img_per_client}_${attack_position_mode}" \
                --model=cnn --dataset=cifar --lr=0.15 --local_bs=16 --local_ep=5 \
                --gama=0.5 --num_steal="${num_steal}" --num_img_per_client="${num_img_per_client}" \
                --agg_mode=segmented --attack_position_mode="${attack_position_mode}"
        done
    done
}

run_worker() {
    gpu="$1"
    worker_index="$2"
    worker_count="$3"
    total_jobs="${#JOBS_NAME[@]}"

    job_index="${worker_index}"
    while [ "${job_index}" -lt "${total_jobs}" ]; do
        job_name="${JOBS_NAME[${job_index}]}"
        job_args="${JOBS_ARGS[${job_index}]}"
        log_path="${LOG_DIR}/${job_index}_${job_name}_gpu${gpu}.log"

        echo "==> [GPU ${gpu}] Start job ${job_index}/${total_jobs}: ${job_name}"
        read -r -a args_array <<< "${job_args}"
        if run_fed --gpu="${gpu}" "${args_array[@]}" > "${log_path}" 2>&1; then
            echo "==> [GPU ${gpu}] Done job ${job_index}: ${job_name}"
        else
            echo "!! [GPU ${gpu}] Failed job ${job_index}: ${job_name}. See ${log_path}" >&2
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

    echo "Total jobs: ${#JOBS_NAME[@]}"
    echo "GPUs: ${GPU_LIST}"
    echo "Logs: ${LOG_DIR}"

    if [ "${DRY_RUN}" = "1" ]; then
        for job_index in "${!JOBS_NAME[@]}"; do
            gpu="${gpus[$((job_index % ${#gpus[@]}))]}"
            echo "DRY_RUN job ${job_index}: GPU ${gpu} ${JOBS_NAME[${job_index}]} ${JOBS_ARGS[${job_index}]}"
        done
        exit 0
    fi

    pids=()
    for worker_index in "${!gpus[@]}"; do
        run_worker "${gpus[${worker_index}]}" "${worker_index}" "${#gpus[@]}" &
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
