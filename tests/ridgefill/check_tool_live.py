"""CPU checks for RidgeFill tool flags, server request modes, cache changes and named refusals."""

import argparse
import json
import math
import re
import subprocess
import time
import urllib.error
from pathlib import Path

import numpy as np

from eval_client import Client, EvalError, check_fill, completion_body, target_logprob


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=root / "out/gate1-qwen35")
    parser.add_argument("--work", type=Path, default=root / "out/tool-live-control")
    parser.add_argument("--port", type=int, default=18088)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    binary, fixture = args.bin_dir.resolve(), args.fixture.resolve()
    model, projector = fixture / "tiny-qwen35-f32.gguf", fixture / "projector-s3-f32.gguf"
    ids = np.fromfile(fixture / "tokens.bin", dtype=np.int32).tolist()
    if len(ids) != 700:
        raise ValueError("this control requires the 700-token tiny fixture")
    checked = 0

    def check(condition, message):
        nonlocal checked
        if not condition:
            raise AssertionError(message)
        checked += 1

    def run(command, label, *, refused=None):
        result = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=120)
        (args.work / (label + ".stdout")).write_text(result.stdout)
        (args.work / (label + ".stderr")).write_text(result.stderr)
        if refused:
            check(result.returncode == 1 and refused in result.stderr, label + " did not refuse by name")
        elif result.returncode:
            raise RuntimeError(f"{label}: exit {result.returncode}; see {args.work / (label + '.stderr')}")
        return result

    expected = {}
    target = ids[-1]
    for mode, tail in (("off", 200), ("speed", 200), ("speed", 201)):
        label = f"{mode}-t{tail}"
        prefix = args.work.resolve() / ("native-" + label)
        run([args.dump.resolve(), "--model", model, "--tokens", fixture / "tokens.bin", "--output", prefix,
             "--projector", projector, "--mode", mode, "--alpha", "1", "--tail", tail, "--keep", "1",
             "--batch", "700", "--ubatch", "128"], "dump-" + label)
        logits = np.fromfile(str(prefix) + ".logits", dtype=np.float32).astype(np.float64)
        check(logits.shape == (300,) and np.isfinite(logits).all(), "missing/nonfinite native logits")
        shift = float(np.max(logits))
        expected[label] = float(logits[target] - shift - np.log(np.exp(logits - shift).sum()))

    common = ["-m", model, "-ngl", "0", "-t", "4", "-ctk", "f32", "-ctv", "f32", "-fa", "off", "--no-warmup"]
    knobs = ["--ridgefill", projector, "--ridgefill-tail", "200", "--ridgefill-alpha", "1", "--ridgefill-share", "0.5",
             "--ridgefill-digits", "off", "--ridgefill-qattn-share", "0.2", "--ridgefill-qattn-window", "32", "--ridgefill-qattn-blind"]
    with (args.work / "server.log").open("w") as log:
        command = [binary / "llama-server", *common, *knobs, "--ridgefill-min-prompt", "1", "--ridgefill-mode", "speed",
                   "-c", "1024", "-b", "700", "-ub", "128", "-np", "1", "--host", "127.0.0.1", "--port", args.port, "--offline"]
        server = subprocess.Popen(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT)
        client = Client(f"http://127.0.0.1:{args.port}", timeout=30)
        try:
            deadline = time.monotonic() + 30
            while True:
                if server.poll() is not None:
                    raise RuntimeError("server exited; see server.log")
                try:
                    client.request("/health")
                    break
                except (urllib.error.URLError, EvalError):
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.1)
            props = client.request("/props")["ridgefill"]
            check(all(props[key] == value for key, value in dict(mode="speed", tail=200, min_prompt=1, share=0.5,
                      digits=0, alpha=1, qattn_window=32, qattn_blind=True).items()) and abs(props["qattn_share"] - 0.2) < 1e-6,
                  "CLI knobs were not propagated to server props")
            responses = []
            for mode, cache, tail, enabled, expected_fill in (("off", False, 200, True, 0), ("speed", False, 200, True, 500),
                    ("speed", True, 200, True, 0), ("speed", True, 201, True, 499), ("off", True, 200, True, 0),
                    ("speed", False, 700, True, 0), ("quality+qattn", False, 200, False, 0)):
                body = completion_body(ids, mode, tail, cache=cache)
                body.update(ridgefill=enabled, n_probs=1, logit_bias=[[target, 1000000.0]])
                result = client.request("/completion", body)
                effective = "speed" if mode == "speed" and tail < len(ids) else "off"
                filled = check_fill(result, expected=expected_fill > 0, mode=effective)
                observed = target_logprob(result, target)
                reference = f"speed-t{tail}" if effective == "speed" else "off-t200"
                check(abs(observed - expected[reference]) < 1e-5, f"{mode}/cache={cache}/tail={tail} differs from native {reference}")
                check(filled == expected_fill, "wrong filled-row count")
                responses.append(dict(mode=mode, cache=cache, tail=tail, enabled=enabled, filled=filled, logprob=observed))
            for mode, message in (("quality", "rho-scaled correction"), ("quality+qattn", "question-attention sweep driver"),
                                  ("invalid", "ridgefill_mode")):
                try:
                    client.request("/completion", completion_body(ids, mode, 200))
                except EvalError as error:
                    check(message in str(error), f"{mode} refusal did not name {message}")
                else:
                    raise AssertionError(f"{mode} unexpectedly succeeded")
            text = client.request("/detokenize", dict(tokens=ids))["content"]
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()

    document = args.work.resolve() / "perplexity.txt"
    document.write_text((text + "\n") * 8)
    for tool in ("llama-bench", "llama-perplexity"):
        for mode, message in (("quality", "rho-scaled correction"), ("quality+qattn", "question-attention sweep")):
            run([binary / tool, "-m", model, "-b", "512", "-ub", "128", "-t", "4", "-ngl", "0",
                 "--ridgefill", projector, "--ridgefill-tail", "200", "--ridgefill-alpha", "1",
                 "--ridgefill-mode", mode], tool + "-refuse-" + mode, refused=message)
    run([binary / "llama-perplexity", *common, *knobs, "--ridgefill-mode", "quality", "--ridgefill-alpha", "0",
         "-c", "512", "-b", "1024", "--no-kv-unified", "-f", document], "ppl-refuse-parallel", refused="--kv-unified")
    short_document = args.work.resolve() / "too-short.txt"
    short_document.write_text("short\n")
    run([binary / "llama-perplexity", *common, *knobs, "--ridgefill-mode", "quality", "--ridgefill-alpha", "0",
         "-c", "512", "-b", "512", "-f", short_document], "ppl-refuse-no-score", refused="no finite positive score")
    for mode in ("off", "speed", "quality"):
        extra = ["--ridgefill-alpha", "0", "--ridgefill-digits", "on"] if mode == "quality" else []
        bench_common = ["f16" if item == "f32" else item for item in common]
        result = run([binary / "llama-bench", *bench_common, *knobs, *extra, "--ridgefill-mode", mode,
                      "-p", "700", "-n", "0", "-b", "700", "-ub", "128", "-r", "1", "-o", "json"], "bench-" + mode)
        rows = json.loads(result.stdout)
        check(len(rows) == 1 and rows[0]["n_prompt"] == 700 and math.isfinite(rows[0]["avg_ts"]) and rows[0]["avg_ts"] > 0,
              "bench did not measure its prompt")
        if mode == "quality":
            selected = re.search(r"mode quality selected ([0-9]+) exact rows", result.stderr)
            check(selected is not None and int(selected[1]) > 0, "bench quality selected no exact rows")
        result = run([binary / "llama-perplexity", *common, *knobs, *extra, "--ridgefill-mode", mode, "--ridgefill-min-prompt", "1",
                      "-c", "512", "-b", "512", "-ub", "128", "--chunks", "1", "-f", document], "ppl-" + mode)
        estimate = re.search(r"Final estimate: PPL = ([0-9.]+)", result.stdout + result.stderr)
        check(estimate is not None and math.isfinite(float(estimate[1])) and float(estimate[1]) > 0, "perplexity scored no chunk")
        if mode != "off":
            check("fills positions [0, 256)" in result.stderr, "perplexity did not activate fill")
        if mode == "quality":
            selected = re.search(r"mode quality selected ([0-9]+) exact rows", result.stderr)
            check(selected is not None and int(selected[1]) > 0, "perplexity quality selected no exact rows")
    print(json.dumps(dict(result="PASS", checks=checked, scope="tiny CPU off/speed/quality tool integration; rho/qattn refuse", requests=responses)))


if __name__ == "__main__":
    main()
