#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_TAG="${CONFIG_TAG:-rank11_shards10}"
export LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs_rank11_shards10}"
export NONIID_MODE="${NONIID_MODE:-shards}"
export SHARDS_PER_USER="${SHARDS_PER_USER:-10}"
export DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.0}"
export DIRICHLET_MIN_SIZE="${DIRICHLET_MIN_SIZE:-100}"

exec "${SCRIPT_DIR}/run_all_rank10_dirichlet05.sh"
