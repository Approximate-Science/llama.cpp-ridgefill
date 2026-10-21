#!/usr/bin/env bash
# Run outside the single-card bridge using the clean Vulkan archive build.
set -euo pipefail
if [[ $# != 5 && $# != 7 ]]; then
    echo "usage: $0 BIN_DIR COMMIT MODEL PROJECTOR OUT [--device-name-pattern REGEX]" >&2
    exit 2
fi
options=()
if [[ $# == 7 ]]; then
    [[ $6 == --device-name-pattern ]] || { echo "expected --device-name-pattern REGEX" >&2; exit 2; }
    options+=("$6" "$7")
fi
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$script_dir/check_gpu_smoke.py" --bin-dir "$1" --expect-commit "$2" --model "$3" \
    --projector "$4" --out "$5" --backend vulkan --two-card "${options[@]}"
