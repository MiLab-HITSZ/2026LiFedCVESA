#!/usr/bin/env bash

set -eu

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${COMMON_DIR}/.." && pwd)"

NUM_USERS="${NUM_USERS:-10}"
FRAC="${FRAC:-1.0}"
EPOCHS="${EPOCHS:-50}"
GAMA_WARMUP_EPOCHS="${GAMA_WARMUP_EPOCHS:-0}"
IID="${IID:-1}"
SEED="${SEED:-1}"

run_fed() {
    (
        cd "${PROJECT_ROOT}"
        python src/federated_main.py \
            --num_users="${NUM_USERS}" \
            --frac="${FRAC}" \
            --iid="${IID}" \
            --epochs="${EPOCHS}" \
            --gama_warmup_epochs="${GAMA_WARMUP_EPOCHS}" \
            --seed="${SEED}" \
            "$@"
    )
}
