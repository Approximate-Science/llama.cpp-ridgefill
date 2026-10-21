#!/usr/bin/env bash
# Run on the host through lane Q's bridge. The bridge selects the single GPU.
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
    echo "usage: $0 vulkan|hip COMMIT FIXTURE MODEL PROJECTOR OUT [ROCM_SDK]" >&2
    exit 2
fi
backend=$1
cmake_args=()
case "$backend" in
    vulkan)
        [[ $# == 6 ]] || { echo "Vulkan does not accept ROCM_SDK" >&2; exit 2; }
        ;;
    hip)
        [[ $# == 7 ]] || { echo "HIP requires the ROCm 7.10 SDK directory" >&2; exit 2; }
        rf_sdk=$(realpath "$7")
        [[ -x "$rf_sdk/bin/hipcc" && -f "$rf_sdk/.info/version" ]] || { echo "ROCm SDK lacks hipcc or .info/version" >&2; exit 2; }
        [[ $(cat "$rf_sdk/.info/version") == 7.10.0* ]] || { echo "HIP smoke requires the ROCm 7.10 SDK" >&2; exit 2; }
        export ROCM_PATH="$rf_sdk" HIP_PATH="$rf_sdk" CMAKE_PREFIX_PATH="$rf_sdk"
        export PATH="$rf_sdk/bin:$rf_sdk/llvm/bin:$PATH" LD_LIBRARY_PATH="$rf_sdk/lib"
        cmake_args+=(--cmake-arg=-DAMDGPU_TARGETS=gfx1201)
        ;;
    *) echo "backend must be vulkan or hip" >&2; exit 2 ;;
esac
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$script_dir/run_g5.py" --ref "$2" --backend "$backend" --fixture "$3" --out "$6" \
    --gpu-layers 99 --flash-attn off --smoke-model "$4" --smoke-projector "$5" "${cmake_args[@]}"
