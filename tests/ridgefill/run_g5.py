#!/usr/bin/env python3
"""G5: pinned git-archive build, portability grep, and live named-refusal controls.

The output directory must be new. Git cannot discover the parent worktree from the archive;
build IDs are pinned explicitly. Backend/toolchain choices are arguments, never device IDs.
This script does not provide container isolation: run it inside a clean container when available.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


def patterns() -> list[str]:
    return [r"/(?:var/)?(?:home|mnt|Users)/", r"[A-Z]:[\\/]Users[\\/]",
            r"\b(?:[0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]\b",
            r"\b(?:Vulkan|CUDA|HIP|ROCm)[0-9]+\b", r"\bGPU-[0-9a-f-]{8,}\b",
            r"\bR9[0-9]{3}\b", r"\bRadeon\s+AI\s+PRO\s+[0-9]+\b", r"\bgfx(?:11|12)[0-9a-z]+\b",
            r"\b(?:ROCm|HIP|SDK)[ /-]*(?:version[ :=-]*)?[vV]?[0-9]+\.[0-9]+(?:\.[0-9]+)?\b",
            r"(?:^|[^\w.])(?:[6-9]|[1-9][0-9]+)\.[0-9]+\.[0-9]+(?:[a-z][0-9]+)?(?:$|[^\w.])",
            r"(?:CUDA|HIP|ROCR|GGML_VK)_VISIBLE_DEVICES\s*=\s*[\"']?[0-9]",
            r"\b(?:tp_size|tensor_parallel_size|world_size|n_gpus)\s*(?:==|=|:)\s*[2-9]\b",
            r"[\w-]+\.(?:local|lan)\b", re.escape(platform.node())]


def sdk_version(version: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"\s*(\d+)\.(\d+)(?:\.(\d+))?(?:[a-zA-Z+~.-].*)?\s*", version)
    if match is None:
        raise ValueError("SDK version must start with major.minor[.patch]")
    return int(match[1]), int(match[2]), int(match[3] or 0)


def check_sdk_version(observed: str, cmake: str, requested: str | None = None) -> None:
    match = re.search(r"\$\{hip_VERSION\}\s+VERSION_LESS\s+(\d+\.\d+(?:\.\d+)?)", cmake)
    if match is None:
        raise ValueError("cannot determine the backend's minimum HIP version")
    minimum = max(sdk_version(match[1]), sdk_version(requested) if requested else (0, 0, 0))
    if sdk_version(observed) < minimum:
        raise ValueError("ROCm SDK version is below the backend or --sdk-min-version requirement")


def hip_environment(args, source: Path, env: dict[str, str]) -> dict[str, str]:
    if args.backend != "hip":
        return env
    sdk_path = args.rocm_sdk or env.get("ROCM_PATH") or env.get("HIP_PATH")
    if sdk_path is None:
        sdk_path = subprocess.check_output(["hipconfig", "-R"], text=True, env=env).strip()
    sdk = Path(sdk_path).resolve()
    if not (sdk / "bin/hipcc").is_file():
        raise ValueError("ROCm SDK lacks bin/hipcc; supply --rocm-sdk or configure HIP_PATH")
    check_sdk_version((sdk / ".info/version").read_text(), (source / "ggml/src/ggml-hip/CMakeLists.txt").read_text(), args.sdk_min_version)
    env.update(ROCM_PATH=str(sdk), HIP_PATH=str(sdk), CMAKE_PREFIX_PATH=str(sdk),
               PATH=str(sdk / "bin") + os.pathsep + str(sdk / "llvm/bin") + os.pathsep + env.get("PATH", ""),
               LD_LIBRARY_PATH=str(sdk / "lib") + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""))
    return env


def source_files(root: Path) -> list[Path]:
    files = list((root / "src").glob("*ridgefill*"))
    files += list((root / "tools/server").glob("*ridgefill*"))
    files += [root / path for path in ("src/llama-context.cpp", "src/llama-graph.cpp", "src/llama-graph.h", "src/llama-ext.h",
              "src/models/qwen35.cpp", "common/common.cpp", "common/common.h", "common/arg.cpp",
              "tools/server/server-context.cpp", "tools/server/server-schema.cpp", "tools/server/server-task.cpp",
              "tools/server/server-task.h", "tools/server/server-common.cpp", "tools/server/server-common.h", "tools/llama-bench/llama-bench.cpp", "tools/perplexity/perplexity.cpp",
              "convert_ridgefill_to_gguf.py")]
    for directory in ("tests/ridgefill", "tools/ridgefill"):
        files += [path for path in (root / directory).glob("*") if path.suffix in (".py", ".cpp", ".sh")]
    return sorted(set(path for path in files if path.is_file()))


def grep(files: list[Path], log: Path) -> subprocess.CompletedProcess:
    if not files:
        raise ValueError("SKIPPED: no RidgeFill sources to audit")
    audit_patterns = patterns()
    command = ["rg", "-n", "-i", "--with-filename"]
    for pattern in audit_patterns:
        command += ["-e", pattern]
    command += ["--", *map(str, files)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        expressions = [re.compile(pattern, re.IGNORECASE) for pattern in audit_patterns]
        matches = []
        for path in files:
            with path.open(encoding="utf-8") as source:
                for number, line in enumerate(source, 1):
                    text = line.rstrip("\r\n")
                    if any(pattern.search(text) for pattern in expressions):
                        matches.append(f"{path}:{number}:{text}\n")
        result = subprocess.CompletedProcess("Python portability scan", 0 if matches else 1, "".join(matches), "")
    log.write_text(result.stdout + result.stderr)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"portability rg failed: {result.stderr}")
    return result


def run(args) -> dict:
    root = Path(__file__).resolve().parents[2]
    args.out.mkdir(parents=True, exist_ok=False)
    records = []

    def record(name, passed, **details):
        row = dict(check=name, result="PASS" if passed else "FAIL", **details)
        records.append(row)
        print(json.dumps(row), flush=True)
        if not passed:
            raise AssertionError(name)

    def command(argv, label, env=None):
        (args.out / (label + ".command.json")).write_text(json.dumps(argv, indent=2) + "\n")
        with (args.out / (label + ".log")).open("w") as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)

    try:
        revision = subprocess.check_output(["git", "rev-parse", "--verify", args.ref + "^{commit}"], cwd=root, text=True).strip()
        count = subprocess.check_output(["git", "rev-list", "--count", revision], cwd=root, text=True).strip()
        source, build = args.out / "source", args.out / "build"
        source.mkdir()
        archive = args.out / "source.tar"
        command(["git", "-C", str(root), "archive", "--format=tar", "--output=" + str(archive), revision], "archive")
        command(["tar", "-xf", str(archive), "-C", str(source)], "extract")
        record("archive-has-no-git-metadata", not (source / ".git").exists() and (source / "CMakeLists.txt").is_file(), revision=revision)
        env = os.environ.copy()
        env["GIT_CEILING_DIRECTORIES"] = str(args.out)
        probe = subprocess.run(["git", "-C", str(source), "rev-parse", "--show-toplevel"], capture_output=True, text=True, env=env)
        record("archive-cannot-find-parent-worktree", probe.returncode != 0)
        files = source_files(source)
        result = grep(files, args.out / "portability-grep.log")
        record("portability-grep-empty", result.returncode == 1 and not result.stdout, files=len(files))
        (args.out / "audit.json").write_text(json.dumps(dict(files=[str(path.relative_to(source)) for path in files], patterns=patterns()), indent=2) + "\n")
        bad = args.out / "nonportable.cpp"
        assumptions = ['/' + 'home/user/model.gguf', platform.node(), 'Vulkan' + str(2), 'int tp_size = ' + str(2),
                       'R9' + str(123), 'gfx' + str(1200), 'gfx' + str(1100), 'ROCm ' + '.'.join(map(str, (8, 0, 0))),
                       ':'.join(('abcd', '12', '34.5')), ':'.join(('56', '78.0'))]
        bad.write_text('\n'.join(assumptions) + '\n')
        result = grep([bad], args.out / "positive-grep.log")
        record("positive-grep-detects-assumptions", result.returncode == 0 and len(result.stdout.splitlines()) == len(assumptions))
        clean = args.out / "portable.cpp"
        clean.write_text("int count(int argc) { return argc; }\n")
        result = grep([clean], args.out / "negative-grep.log")
        record("negative-grep-is-empty", result.returncode == 1 and not result.stdout)
        flags = ["-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_TOOLS=ON",
                 "-DLLAMA_BUILD_SERVER=ON", "-DLLAMA_BUILD_UI=OFF", "-DLLAMA_OPENSSL=OFF", "-DGGML_NATIVE=OFF",
                 "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF", "-DGGML_VULKAN=" + ("ON" if args.backend == "vulkan" else "OFF"),
                 "-DGGML_HIP=" + ("ON" if args.backend == "hip" else "OFF"), "-DLLAMA_BUILD_COMMIT=" + revision[:9],
                 "-DLLAMA_BUILD_NUMBER=" + count]
        env = hip_environment(args, source, env)
        if args.hip_targets:
            flags += ["-DAMDGPU_TARGETS=" + args.hip_targets]
        command(["cmake", "-S", str(source), "-B", str(build), *flags, *args.cmake_arg], "configure", env)
        command(["cmake", "--build", str(build), "--target", "llama-server", "llama-bench", "llama-perplexity", "-j" + str(args.jobs)], "build", env)
        record("archive-tools-build", all((build / "bin" / name).is_file() for name in ("llama-server", "llama-bench", "llama-perplexity")))
        versions = {}
        for name in ("llama-server", "llama-bench", "llama-perplexity"):
            result = subprocess.run([str(build / "bin" / name), "--version"], capture_output=True, text=True, check=True, env=env)
            versions[name] = result.stdout + result.stderr
            record(name + "-version-pinned", revision[:9] in versions[name])
        checker = source / "tests/ridgefill/check_refusals.py"
        if not checker.is_file():
            raise ValueError("SKIPPED: archived revision lacks tests/ridgefill/check_refusals.py")
        command([sys.executable, str(checker), "--bin-dir", str(build / "bin"), "--fixture", str(args.fixture),
                 "--work", str(args.out / "refusals"), "--port", str(args.port), "--gpu-layers", str(args.gpu_layers),
                 "--flash-attn", args.flash_attn], "refusals", env)
        refusals = json.loads((args.out / "refusals/results.json").read_text())
        record("named-refusal-controls", refusals["result"] == "PASS" and refusals["checks"] > 0, checks=refusals["checks"])
        smoke = None
        if args.smoke_model:
            checker = source / "tests/ridgefill/check_gpu_smoke.py"
            if not checker.is_file():
                raise ValueError("SKIPPED: archived revision lacks tests/ridgefill/check_gpu_smoke.py")
            if args.backend == "hip":
                command(["hipcc", "--version"], "hipcc-version", env)
                sdk = env.get("ROCM_PATH")
                if sdk:
                    version = Path(sdk) / ".info/version"
                    (args.out / "rocm-sdk-version.log").write_text(version.read_text())
            name_options = ["--device-name-pattern", args.device_name_pattern] if args.device_name_pattern is not None else []
            command([sys.executable, str(checker), "--bin-dir", str(build / "bin"), "--expect-commit", revision,
                     "--backend", args.backend, "--model", str(args.smoke_model), "--projector", str(args.smoke_projector),
                     "--out", str(args.out / "gpu-smoke"), *name_options], "gpu-smoke", env)
            smoke = json.loads((args.out / "gpu-smoke/results.json").read_text())
            record("real-model-speed-smoke", smoke["result"] == "PASS", devices=smoke["devices"])
        result = dict(gate="G5", result="PASS", scope="archive build, source audit, named refusals on selected backend", backend=args.backend,
                      revision=revision, checks=len(records), refusal_checks=refusals["checks"], records=records, versions=versions,
                      isolation="caller environment; container isolation NOT PROVIDED", multi_gpu="NOT RUN", gpu_smoke=smoke)
        (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    except Exception as error:
        (args.out / "results.json").write_text(json.dumps(dict(gate="G5", result="FAIL", reason=str(error), records=records), indent=2) + "\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "vulkan", "hip"), default="cpu")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cmake-arg", action="append", default=[], help="one CMake argument, supplied as --cmake-arg=-DOPTION=VALUE")
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--flash-attn", choices=("off", "on", "auto"), default="off")
    parser.add_argument("--port", type=int, default=18110)
    parser.add_argument("--smoke-model", type=Path, help="GPU only: finish with a short 27B speed-mode benchmark")
    parser.add_argument("--smoke-projector", type=Path)
    parser.add_argument("--rocm-sdk", type=Path, help="HIP SDK directory; default uses ROCM_PATH, HIP_PATH or hipconfig -R")
    parser.add_argument("--hip-targets", help="semicolon-separated HIP targets; default lets CMake detect targets")
    parser.add_argument("--sdk-min-version", help="require this SDK version or the backend's minimum, whichever is newer")
    parser.add_argument("--device-name-pattern", help="optional regex over GPU descriptions; default accepts any GPU")
    args = parser.parse_args()
    args.out, args.fixture = args.out.resolve(), args.fixture.resolve()
    if args.backend != "hip" and any((args.rocm_sdk, args.hip_targets, args.sdk_min_version)):
        parser.error("HIP toolchain options require --backend hip")
    if args.sdk_min_version is not None:
        try:
            sdk_version(args.sdk_min_version)
        except ValueError as error:
            parser.error(str(error))
    if args.device_name_pattern is not None:
        try:
            re.compile(args.device_name_pattern)
        except re.error as error:
            parser.error(f"invalid --device-name-pattern: {error}")
    if bool(args.smoke_model) != bool(args.smoke_projector):
        parser.error("--smoke-model and --smoke-projector must be supplied together")
    if args.smoke_model:
        if args.backend == "cpu" or args.gpu_layers != 99:
            parser.error("real-model smoke requires a GPU backend and --gpu-layers 99")
        args.smoke_model, args.smoke_projector = args.smoke_model.resolve(), args.smoke_projector.resolve()
    if not 1 <= args.jobs <= 4:
        parser.error("--jobs must be in [1,4]")
    reserved = r"-D(?:GGML_(?:HIP|VULKAN|CUDA|METAL|NATIVE)|LLAMA_(?:BUILD_(?:COMMIT|NUMBER|TESTS|EXAMPLES|TOOLS|SERVER|UI)|OPENSSL))(?::|=)"
    if any(re.match(reserved, value) for value in args.cmake_arg):
        parser.error("--cmake-arg cannot override the gate's backend, build identity or target options")
    try:
        print(json.dumps(run(args)), flush=True)
    except (OSError, ValueError, RuntimeError, AssertionError, subprocess.CalledProcessError) as error:
        print(json.dumps(dict(gate="G5", result="FAIL", reason=str(error))), flush=True)
        raise SystemExit(1)
