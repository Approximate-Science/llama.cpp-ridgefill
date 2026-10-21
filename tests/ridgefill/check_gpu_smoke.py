#!/usr/bin/env python3
"""Run one short Qwen3.8-27B speed prefill; refuse CPU fallback or unused GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def select_devices(inventory: str, backend: str, two_card: bool) -> list[str]:
    prefix = r"Vulkan" if backend == "vulkan" else r"(?:ROCm|HIP)"
    rows = re.findall(r"^\s*(" + prefix + r"\d+): (.+) \((\d+) MiB, (\d+) MiB free\)\s*$", inventory, re.M)
    if two_card:
        rows = [row for row in rows if re.search(r"\b(?:R9700|Radeon\s+AI\s+PRO\s+9700)\b", row[1], re.I)]
    expected = 2 if two_card else 1
    if len(rows) != expected or len({row[0] for row in rows}) != expected:
        raise ValueError(f"GPU smoke requires exactly {expected} visible {'R9700 Vulkan' if two_card else backend} device(s)")
    if any(int(row[3]) < 3072 for row in rows):
        raise ValueError("GPU smoke requires at least 3072 MiB free on each selected device")
    return [row[0] for row in rows]


def validate_bench(rows: list[dict], log: str, devices: list[str], commit: str,
                   backend: str, prompt: int = 8192, tail: int = 4096) -> dict:
    if len(rows) != 1:
        raise ValueError("GPU smoke requires exactly one benchmark measurement")
    row = rows[0]
    expected = dict(n_prompt=prompt, n_gen=0, n_depth=0, n_gpu_layers=99, split_mode="layer",
                    type_k="f16", type_v="f16", flash_attn=0)
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError("GPU smoke benchmark configuration differs from the requested smoke")
    if row.get("build_commit") != commit or row.get("devices") != "/".join(devices):
        raise ValueError("GPU smoke benchmark revision/devices differ from the requested smoke")
    supported = ("VULKAN",) if backend == "vulkan" else ("ROCM", "HIP")
    if not any(value in row.get("backends", "").upper().split(",") for value in supported):
        raise ValueError("GPU smoke backend is missing; CPU fallback is refused")
    for key in ("avg_ts", "avg_ns"):
        if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) or row[key] <= 0:
            raise ValueError("GPU smoke requires finite positive timing and throughput")
    if len(row.get("samples_ns", [])) != 1 or len(row.get("samples_ts", [])) != 1:
        raise ValueError("GPU smoke requires one measured repetition")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
           for key in ("samples_ns", "samples_ts") for value in row[key]):
        raise ValueError("GPU smoke requires finite positive measured samples")
    marker = f"RidgeFill: mode speed completed {prompt - tail} prefix rows, {tail} exact tail rows, 1 measured repetitions"
    if marker not in log:
        raise ValueError("GPU smoke did not confirm an active completed speed prefix")
    buffers = {}
    for device in devices:
        sizes = re.findall(r"\b" + re.escape(device) + r"\s+model buffer size\s*=\s*([0-9.]+) MiB", log)
        if not sizes or not all(math.isfinite(float(size)) for size in sizes) or sum(map(float, sizes)) <= 1:
            raise ValueError(f"GPU smoke did not place model tensors on {device}")
        buffers[device] = sum(map(float, sizes))
    return dict(benchmark=row, model_buffer_mib=buffers, prefix_rows=prompt - tail, exact_tail_rows=tail)


def check_model(model: Path, projector: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
    import gguf

    model_reader, projector_reader = gguf.GGUFReader(model), gguf.GGUFReader(projector)
    def integer(reader, key):
        field = reader.fields.get(key)
        if field is None or field.types != [gguf.GGUFValueType.UINT32] or len(field.data) != 1:
            raise ValueError(f"GPU smoke requires scalar UINT32 {key}")
        return int(field.parts[field.data[0]][0])
    field = model_reader.fields.get("general.architecture")
    if field is None or field.types != [gguf.GGUFValueType.STRING] or field.parts[field.data[0]].tobytes() != b"qwen35":
        raise ValueError("GPU smoke requires the qwen35 27B model")
    nextn = integer(model_reader, "qwen35.nextn_predict_layers") if "qwen35.nextn_predict_layers" in model_reader.fields else 0
    if nextn not in (0, 1) or integer(model_reader, "qwen35.block_count") - nextn != 64 or integer(model_reader, "qwen35.embedding_length") != 5120:
        raise ValueError("GPU smoke requires the 64-layer trunk, width-5120 Qwen3.8-27B model")
    for key, expected in (("split", 16), ("tail_default", 4096), ("tail_min", 512)):
        if integer(projector_reader, "adapter.ridgefill." + key) != expected:
            raise ValueError("GPU smoke requires the split-16 release projector and its tail defaults")


def run(args) -> dict:
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        check_model(args.model, args.projector)
        binary = args.bin_dir / "llama-bench"
        version = subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=True, timeout=30)
        version_text = version.stdout + version.stderr
        (args.out / "version.log").write_text(version_text)
        match = re.search(r"build (\d+), commit ([0-9a-f]+)", version_text)
        if match is None or (args.expect_commit and match[2] != args.expect_commit[:9]):
            raise ValueError("GPU smoke cannot identify the expected benchmark revision")
        inventory = subprocess.run([str(binary), "--list-devices"], capture_output=True, text=True, check=True, timeout=30)
        (args.out / "devices.log").write_text(inventory.stdout + inventory.stderr)
        devices = select_devices(inventory.stdout, args.backend, args.two_card)
        command = [str(binary), "-m", str(args.model), "--ridgefill", str(args.projector),
                   "--ridgefill-mode", "speed", "--ridgefill-alpha", "0", "--ridgefill-tail", "4096",
                   "-p", "8192", "-n", "0", "-r", "1", "--no-warmup", "-b", "512", "-ub", "128", "-t", "4",
                   "-ngl", "99", "-sm", "layer", "-dev", "/".join(devices), "-ctk", "f16", "-ctv", "f16", "-fa", "off",
                   "--offline", "-o", "json", "-v"]
        if args.two_card:
            command += ["-ts", "1/1"]
        (args.out / "bench.command.json").write_text(json.dumps(command, indent=2) + "\n")
        with (args.out / "bench.json").open("w") as stdout, (args.out / "bench.log").open("w") as stderr:
            subprocess.run(command, stdout=stdout, stderr=stderr, check=True, timeout=600)
        details = validate_bench(json.loads((args.out / "bench.json").read_text()), (args.out / "bench.log").read_text(),
                                 devices, match[2], args.backend)
        result = dict(gate="G5-GPU-smoke", result="PASS", backend=args.backend, devices=devices, commit=match[2],
                      scope="short speed-mode prefill; no performance, numerical-equivalence or container-isolation claim",
                      model=str(args.model), model_sha256=digest(args.model), projector=str(args.projector),
                      projector_sha256=digest(args.projector), **details)
        (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    except Exception as error:
        (args.out / "results.json").write_text(json.dumps(dict(gate="G5-GPU-smoke", result="FAIL", reason=str(error)), indent=2) + "\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("vulkan", "hip"), required=True)
    parser.add_argument("--two-card", action="store_true", help="Vulkan only: require both visible R9700 cards")
    parser.add_argument("--expect-commit")
    args = parser.parse_args()
    for key in ("bin_dir", "model", "projector", "out"):
        setattr(args, key, getattr(args, key).resolve())
    if args.two_card and args.backend != "vulkan":
        parser.error("--two-card requires Vulkan")
    try:
        print(json.dumps(run(args)), flush=True)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps(dict(gate="G5-GPU-smoke", result="FAIL", reason=str(error))), flush=True)
        raise SystemExit(1)
