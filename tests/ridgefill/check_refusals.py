"""Live RidgeFill unsupported-combination checks, with exact-slot save and active-fill controls."""

import argparse
import json
import subprocess
import time
import urllib.error
from pathlib import Path

import numpy as np

from eval_client import Client, EvalError, completion_body


def run(args) -> dict:
    args.work.mkdir(parents=True, exist_ok=False)
    model = args.fixture / "tiny-qwen35-f32.gguf"
    projector = args.fixture / "projector-s3-f32.gguf"
    ids = np.fromfile(args.fixture / "tokens.bin", dtype=np.int32).tolist()
    if len(ids) != 700:
        raise ValueError("refusal controls require the 700-token tiny fixture")
    records = []

    def check(name, condition, **details):
        row = dict(check=name, result="PASS" if condition else "FAIL", **details)
        records.append(row)
        print(json.dumps(row), flush=True)
        if not condition:
            raise AssertionError(name)

    common = ["-m", str(model), "-ngl", str(args.gpu_layers), "-t", "4", "-ctk", "f16", "-ctv", "f16", "-fa", args.flash_attn,
              "--ridgefill", str(projector), "--ridgefill-tail", "200", "--no-warmup"]
    server_args = ["-c", "1024", "-b", "700", "-ub", "128", "-np", "1", "--ridgefill-min-prompt", "1", "--offline",
                   "--host", "127.0.0.1", "--port", str(args.port)]
    missing = args.work / "must-not-open.gguf"
    cases = (("lora", ["--lora", str(missing)], "LoRA"),
             ("control-vector", ["--control-vector", str(missing)], "control vectors"),
             ("embeddings", ["--embeddings"], "embeddings"),
             ("cache-reuse", ["--cache-reuse", "16"], "--cache-reuse"),
             ("row-split", ["-sm", "row"], "--split-mode row"))
    for name, extra, message in cases:
        command = [str(args.bin_dir / "llama-server"), *common, *server_args, *extra]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        text = result.stdout + result.stderr
        (args.work / (name + ".log")).write_text(text)
        check("startup-refuses-" + name, result.returncode == 1 and "RidgeFill" in text and message in text, returncode=result.returncode)
    for name, extra, message in (("row-split", ["-sm", "row"], "--split-mode row"), ("embeddings", ["--embeddings", "1"], "embeddings")):
        command = [str(args.bin_dir / "llama-bench"), *common, "-p", "700", "-n", "0", "-r", "1", *extra]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        text = result.stdout + result.stderr
        (args.work / ("bench-" + name + ".log")).write_text(text)
        check("bench-refuses-" + name, result.returncode == 1 and "RidgeFill" in text and message in text, returncode=result.returncode)

    stock_args = common[:]
    index = stock_args.index("--ridgefill")
    del stock_args[index:index + 2]
    result = subprocess.run([str(args.bin_dir / "llama-server"), *stock_args, *server_args, "--lora", str(missing)],
                            capture_output=True, text=True, timeout=60)
    text = result.stdout + result.stderr
    (args.work / "stock-missing-lora.log").write_text(text)
    check("negative-stock-lora-uses-normal-loader", result.returncode == 1 and str(missing) in text and "RidgeFill does not support LoRA" not in text)

    slots = args.work / "slots"
    slots.mkdir()
    client = Client(f"http://127.0.0.1:{args.port}", timeout=60)
    with (args.work / "server.log").open("w") as log:
        server = subprocess.Popen([str(args.bin_dir / "llama-server"), *common, *server_args, "--cache-reuse", "0",
                                   "--slot-save-path", str(slots)], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while True:
                if server.poll() is not None:
                    raise RuntimeError("control server exited; see server.log")
                try:
                    client.request("/health")
                    break
                except (urllib.error.URLError, EvalError):
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.1)
            body = completion_body(ids, "speed", 200)
            body["n_cache_reuse"] = 16
            try:
                client.request("/completion", body)
            except EvalError as error:
                check("request-refuses-cache-reuse", "RidgeFill" in str(error) and "n_cache_reuse" in str(error))
            else:
                raise AssertionError("request cache shifting unexpectedly succeeded")
            filled = client.request("/completion", completion_body(ids, "speed", 200))
            check("positive-fill-is-active", filled["timings"].get("prompt_filled_n") == 500)
            try:
                client.request("/slots/0?action=save", dict(filename="approx.bin"))
            except EvalError as error:
                check("slot-save-refuses-filled-rows", "RidgeFill" in str(error) and "save slot" in str(error))
            else:
                raise AssertionError("filled slot save unexpectedly succeeded")
            check("refused-slot-save-created-no-file", not list(slots.iterdir()))
            exact = client.request("/completion", completion_body(ids, "off", 200))
            check("negative-off-is-exact", exact["timings"].get("prompt_filled_n", 0) == 0)
            saved = client.request("/slots/0?action=save", dict(filename="exact.bin"))
            check("negative-exact-slot-save-works", saved["n_saved"] > 0 and saved["n_written"] > 0)
            check("exact-slot-file-is-nonempty", (slots / "exact.bin").is_file() and (slots / "exact.bin").stat().st_size > 0)
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    result = dict(gate="G5-refusals", result="PASS" if records else "SKIPPED", checks=len(records), records=records)
    (args.work / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18110)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--flash-attn", choices=("off", "on", "auto"), default="off")
    args = parser.parse_args()
    for key in ("bin_dir", "fixture", "work"):
        setattr(args, key, getattr(args, key).resolve())
    try:
        print(json.dumps(run(args)), flush=True)
    except (OSError, ValueError, RuntimeError, AssertionError, subprocess.TimeoutExpired) as error:
        print(json.dumps(dict(gate="G5-refusals", result="FAIL", reason=str(error))), flush=True)
        raise SystemExit(1)
