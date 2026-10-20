#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
exec "${ORACLE_PYTHON:-$root/.venv-oracle/bin/python}" "$root/tests/ridgefill/run_gate1_qwen35.py" "$@"
