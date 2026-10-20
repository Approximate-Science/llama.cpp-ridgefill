#!/usr/bin/env python3
"""Build a pinned engine archive and rf_dump inside this lane; never modify engine sources."""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build(ref: str, work: Path) -> Path:
    revision = subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT, text=True).strip()
    directory = work.resolve() / revision[:12]
    source, binary = directory / "source", directory / "build"
    source.mkdir(parents=True, exist_ok=True)
    archive = directory / "source.tar"
    if not (source / "CMakeLists.txt").exists():
        subprocess.run(["git", "archive", revision, "-o", str(archive)], cwd=ROOT, check=True)
        subprocess.run(["tar", "-xf", str(archive), "-C", str(source)], check=True)
    with (directory / "configure.log").open("w") as log:
        subprocess.run(["cmake", "-S", str(source), "-B", str(binary), "-DCMAKE_BUILD_TYPE=Release",
                        "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_TOOLS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
                        "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_COMMON=OFF"], stdout=log, stderr=subprocess.STDOUT, check=True)
    with (directory / "build.log").open("w") as log:
        result = subprocess.run(["cmake", "--build", str(binary), "-j4", "--target", "llama"], stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        errors = [line for line in (directory / "build.log").read_text().splitlines() if "error:" in line]
        raise RuntimeError(f"engine {revision} build failed: {'; '.join(errors[:3])}; log: {directory / 'build.log'}")
    command = ["c++", "-std=c++17", "-O2", "-Wall", "-Wextra",
               "-I" + str(source / "include"), "-I" + str(source / "src"), "-I" + str(source / "ggml/include")]
    if "llama_ridgefill_set_alpha" in (source / "src/llama-ext.h").read_text():
        command.append("-DRF27_API")
    command += [str(ROOT / "tests/ridgefill/rf_dump.cpp"), "-L" + str(binary / "bin"),
                "-Wl,-rpath," + str(binary / "bin"),
                "-lllama", "-lggml", "-lggml-base", "-o", str(directory / "rf_dump")]
    with (directory / "harness-build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    (directory / "build.json").write_text(json.dumps(dict(revision=revision, source=str(source), binary=str(binary), command=command), indent=2) + "\n")
    return directory / "rf_dump"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="rf27-engine")
    parser.add_argument("--work", type=Path, default=ROOT / "out/gate1-engine")
    args = parser.parse_args()
    print(build(args.ref, args.work))
