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
            r"\b[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]\b",
            r"\b(?:Vulkan|CUDA|HIP)[0-9]+\b", r"\bGPU-[0-9a-f-]{8,}\b",
            r"(?:CUDA|HIP|ROCR|GGML_VK)_VISIBLE_DEVICES\s*=\s*[\"']?[0-9]",
            r"\b(?:tp_size|tensor_parallel_size|world_size|n_gpus)\s*(?:==|=|:)\s*[2-9]\b",
            r"[\w-]+\.(?:local|lan)\b", re.escape(platform.node())]


def source_files(root: Path) -> list[Path]:
    files = list((root / "src").glob("*ridgefill*"))
    files += list((root / "tools/server").glob("*ridgefill*"))
    files += [root / path for path in ("src/llama-context.cpp", "src/llama-graph.cpp", "src/llama-graph.h", "src/llama-ext.h",
              "src/models/qwen35.cpp", "common/common.cpp", "common/common.h", "common/arg.cpp",
              "tools/server/server-context.cpp", "tools/server/server-schema.cpp", "tools/server/server-task.cpp",
              "tools/server/server-task.h", "tools/llama-bench/llama-bench.cpp", "tools/perplexity/perplexity.cpp",
              "convert_ridgefill_to_gguf.py")]
    files += [path for path in (root / "tests/ridgefill").glob("*") if path.suffix in (".py", ".cpp", ".sh")]
    return sorted(set(path for path in files if path.is_file()))


def grep(files: list[Path], log: Path) -> subprocess.CompletedProcess:
    if not files:
        raise ValueError("SKIPPED: no RidgeFill sources to audit")
    command = ["rg", "-n", "-i"]
    for pattern in patterns():
        command += ["-e", pattern]
    command += ["--", *map(str, files)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
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
        bad.write_text('const char * path = "/' + 'home/user/model.gguf";\nconst char * host = "' + platform.node() +
                       '";\nconst char * device = "Vulkan' + str(2) + '";\nint tp_size = ' + str(2) + ';\n')
        result = grep([bad], args.out / "positive-grep.log")
        record("positive-grep-detects-assumptions", result.returncode == 0 and len(result.stdout.splitlines()) == 4)
        clean = args.out / "portable.cpp"
        clean.write_text("int count(int argc) { return argc; }\n")
        result = grep([clean], args.out / "negative-grep.log")
        record("negative-grep-is-empty", result.returncode == 1 and not result.stdout)
        flags = ["-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_TOOLS=ON",
                 "-DLLAMA_BUILD_SERVER=ON", "-DLLAMA_BUILD_UI=OFF", "-DLLAMA_OPENSSL=OFF", "-DGGML_NATIVE=OFF",
                 "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF", "-DGGML_VULKAN=" + ("ON" if args.backend == "vulkan" else "OFF"),
                 "-DGGML_HIP=" + ("ON" if args.backend == "hip" else "OFF"), "-DLLAMA_BUILD_COMMIT=" + revision[:9],
                 "-DLLAMA_BUILD_NUMBER=" + count]
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
            command([sys.executable, str(checker), "--bin-dir", str(build / "bin"), "--expect-commit", revision,
                     "--backend", args.backend, "--model", str(args.smoke_model), "--projector", str(args.smoke_projector),
                     "--out", str(args.out / "gpu-smoke")], "gpu-smoke", env)
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
    args = parser.parse_args()
    args.out, args.fixture = args.out.resolve(), args.fixture.resolve()
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
