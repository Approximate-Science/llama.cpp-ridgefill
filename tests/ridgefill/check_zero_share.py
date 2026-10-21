#!/usr/bin/env python3
"""Check zero class-share HTTP/tool plumbing and byte-identical empty selection on the tiny fixture."""

import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.error

import numpy as np

from eval_client import Client, EvalError, completion_body, target_logprob


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    model = args.chat_fixture / "chat.gguf"
    projector = args.chat_fixture / "projector-s4.gguf"
    zero_projector = args.chat_fixture / "projector-s4-zero.gguf"
    ids = np.fromfile(args.fixture / "tokens.bin", dtype="<i4").tolist()
    for pos in (20, 300, 600):
        header = [1, *[byte + 3 for byte in b"user\n"]]
        ids[pos:pos + len(header)] = header
    source, tokens = args.out / "tokens.jsonl", args.out / "tokens.bin"
    source.write_text(json.dumps(dict(name="chat", ids=[*ids, 7])) + "\n")
    np.asarray(ids, dtype="<i4").tofile(tokens)
    checks = []

    def check(condition, name):
        assert condition, name
        checks.append(name)

    def call(command, name):
        result = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=60)
        (args.out / (name + ".log")).write_text(result.stdout + result.stderr)
        check(result.returncode == 0, name)
        return result

    call([args.dump, "--model", model, "--projector", projector, "--tokens", tokens,
          "--output", args.out / "empty", "--tail", 200, "--select-share", 0,
          "--select-digits", 0], "select-empty")
    check((args.out / "empty.rows.i32").stat().st_size == 0, "selector-zero-returns-empty")
    for mode in ("speed", "quality"):
        call([args.dump, "--model", model, "--projector", projector, "--tokens", tokens,
              "--output", args.out / mode, "--mode", mode, "--tail", 200, "--keep", 200,
              "--batch", 128, "--ubatch", 128], "dump-" + mode)
    for suffix in (".logits", ".kv", ".prefix.kv"):
        a, b = (args.out / (mode + suffix) for mode in ("speed", "quality"))
        check(a.stat().st_size > 0 and a.read_bytes() == b.read_bytes(), "empty-selection-byte-identical" + suffix)

    command = [args.bin_dir / "llama-server", "-m", model, "--ridgefill", projector,
               "--ridgefill-mode", "quality", "--ridgefill-share", 0, "--ridgefill-digits", "off",
               "--ridgefill-alpha", 0, "--ridgefill-tail", 200, "--ridgefill-min-prompt", 1,
               "-c", 1024, "-np", 1, "-b", 128, "-ub", 128, "-t", 4, "-ngl", 0,
               "-ctk", "f32", "-ctv", "f32", "-fa", "off", "--offline", "--no-warmup",
               "--host", "127.0.0.1", "--port", args.port]
    client = Client(f"http://127.0.0.1:{args.port}", timeout=60)
    with (args.out / "server.log").open("w") as log:
        server = subprocess.Popen(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while True:
                assert server.poll() is None, "server exited; see server.log"
                try:
                    client.request("/health")
                    break
                except (urllib.error.URLError, EvalError):
                    assert time.monotonic() < deadline, "startup timeout"
                    time.sleep(0.05)
            check(client.request("/props")["ridgefill"]["share"] == 0, "startup-share-zero")
            for label, mode, special, share in (("empty", "quality", False, 0),
                                               ("qattn-only", "quality+qattn", False, 0),
                                               ("special-only", "quality", True, 0),
                                               ("qattn+special", "quality+qattn", True, 0),
                                               ("override-positive", "quality", False, 0.5)):
                output = args.out / (label + ".jsonl")
                call([args.bin_dir / "llama-ridgefill-score", "--model", model, "--projector", projector,
                      "--input", source, "--output", output, "--mode", mode, "--scored", 1, "--tail", 201,
                      "--min-prompt", 1, "--cache-type", "f32", "--h-type", "f16", "--flash-attn", "off",
                      "--batch", 128, "--ubatch", 128, "--share", share, "--digits", "off",
                      "--special-exact", "on" if special else "off", "--qattn-window-end", 700], "native-" + label)
                native = json.loads(output.read_text())
                body = completion_body(ids, mode, 200)
                body.update(n_probs=300, logit_bias=[[7, 1000000.0]], ignore_eos=True, ridgefill_share=share, ridgefill_special_exact=special)
                response = client.request("/completion", body)
                check(abs(target_logprob(response, 7) - native["logprobs"][0]) < 1e-4, "HTTP-logits-" + label)
                for field in ("n_exact_prefix", "n_exact_prefix_by_source", "n_filled"):
                    check(response["timings"]["ridgefill"][field] == native["ridgefill"][field], "HTTP-" + field + "-" + label)
            for bad in (-0.1, 1.1, "0", True):
                body = completion_body(ids, "quality", 200)
                body["ridgefill_share"] = bad
                try:
                    client.request("/completion", body)
                except EvalError as error:
                    check("ridgefill_share" in str(error), "HTTP-refuse-share-" + str(bad))
                else:
                    raise AssertionError(f"HTTP accepted invalid share {bad!r}")
        finally:
            server.terminate()
            server.wait(timeout=10)

    for tool in ("llama-bench", "llama-perplexity"):
        command = [args.bin_dir / tool, "-m", model, "--ridgefill", zero_projector,
                   "--ridgefill-mode", "quality", "--ridgefill-digits", "off", "--ridgefill-tail", 64,
                   "-t", 4, "-ngl", 0, "-b", 128, "-ub", 128]
        if tool == "llama-bench":
            command += ["-p", 256, "-n", 0, "-r", 1]
        else:
            text = args.out / "perplexity.txt"
            text.write_text("A tiny row-selection test. " * 128)
            command += ["-f", text, "-c", 256, "--chunks", 1, "--no-warmup", "--ridgefill-min-prompt", 1]
        for label in ("zero-default", "zero-option"):
            if label == "zero-option":
                command[command.index(zero_projector)] = projector
                command += ["--ridgefill-share", 0]
            result = call(command, tool + "-" + label)
            reports = [json.loads(line.split("RidgeFill: report ", 1)[1]) for line in result.stderr.splitlines()
                       if "RidgeFill: report " in line]
            check(bool(reports) and all(r["n_exact_prefix"] == 0 and r["n_filled"] > 0 for r in reports),
                  tool + "-uses-" + label)
    summary = dict(result="PASS", checks=len(checks), records=checks)
    (args.out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bin-dir", "fixture", "chat-fixture", "dump", "out"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--port", type=int, default=18134)
    run(parser.parse_args())
