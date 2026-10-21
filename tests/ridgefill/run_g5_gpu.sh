#!/usr/bin/env bash
# Run on the host through lane Q's bridge. The bridge selects the single GPU.
set -euo pipefail
if [[ $# -lt 6 ]]; then
    echo "usage: $0 vulkan|hip COMMIT FIXTURE MODEL PROJECTOR OUT [ROCM_SDK] [--hip-targets TARGETS] [--sdk-min-version VERSION] [--device-name-pattern REGEX]" >&2
    exit 2
fi
backend=$1
ref=$2
fixture=$3
model=$4
projector=$5
output=$6
shift 6
case "$backend" in vulkan|hip) ;; *) echo "backend must be vulkan or hip" >&2; exit 2 ;; esac
options=()
if [[ $# -gt 0 && $1 != --* ]]; then
    options+=(--rocm-sdk "$1")
    shift
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rocm-sdk|--hip-targets|--sdk-min-version|--device-name-pattern)
            [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
            options+=("$1" "$2")
            shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$script_dir/run_g5.py" --ref "$ref" --backend "$backend" --fixture "$fixture" --out "$output" \
    --gpu-layers 99 --flash-attn off --smoke-model "$model" --smoke-projector "$projector" "${options[@]}"
