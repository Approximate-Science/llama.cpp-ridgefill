#!/usr/bin/env python3
"""Check disabled S1/vtail defaults and named tool/request refusals on the tiny fixture."""

import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.error

import numpy as np

from eval_client import Client, EvalError, completion_body


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    model, projector = args.chat_fixture / "chat.gguf", args.chat_fixture / "projector-s4.gguf"
    ids = np.fromfile(args.fixture / "tokens.bin", dtype="<i4").tolist()
    source = args.out / "tokens.jsonl"
    source.write_text(json.dumps(dict(name="raw", ids=ids)) + "\n")
    defaults = ["--ridgefill-s1-window", 0, "--ridgefill-s1-factor", 1, "--ridgefill-vtail", "off", "--ridgefill-vtail-rows", 512]
    checks = []

    def check(condition, name):
        assert condition, name
        checks.append(name)

    def call(command, label, refused=None):
        result = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=60)
        (args.out / (label + ".log")).write_text(result.stdout + result.stderr)
        check(result.returncode == 0 if refused is None else result.returncode != 0 and refused in result.stderr, label)
        return result

    for mode in ("off", "speed", "quality", "quality+qattn"):
        results = []
        for label, extra, binary in (("baseline", [], args.baseline_binary), ("default", [], args.bin_dir / "llama-ridgefill-score"),
                                     ("explicit", defaults, args.bin_dir / "llama-ridgefill-score")):
            output = args.out / f"{mode}-{label}.jsonl"
            call([binary, "--model", model, "--projector", projector, "--input", source, "--output", output,
                  "--mode", mode, "--scored", 64, "--tail", 200, "--min-prompt", 1, "--cache-type", "f32",
                  "--h-type", "f32", "--flash-attn", "off", "--diagnostics", *extra], mode + "-" + label)
            results.append(json.loads(output.read_text()))
        for field in ("logprobs", "logsumexp", "exact_rows", "qattn_window", "question_mass"):
            check(results[0].get(field) == results[1].get(field) == results[2].get(field), mode + "-byte-default-" + field)
        check(results[1]["ridgefill"] == results[2]["ridgefill"] and "s1" not in results[1]["ridgefill"], mode + "-report-default")

    for tool in ("llama-server", "llama-perplexity", "llama-bench", "llama-cli"):
        common = [args.bin_dir / tool, "-m", "missing-model.gguf", "--ridgefill", "missing-projector.gguf"]
        for flag, value in (("--ridgefill-s1-window", 64), ("--ridgefill-vtail", "on")):
            call([*common, "--ridgefill-mode", "quality", flag, value], tool + "-refuse-" + flag, "two-sweep flow")
        for flag, value in (("--ridgefill-s1-window", -1), ("--ridgefill-s1-factor", "nan"), ("--ridgefill-vtail-rows", 0)):
            call([*common, flag, value], tool + "-bad-" + flag, flag)

    for flag, value in (("--ridgefill-s1-window", 64), ("--ridgefill-vtail", "on")):
        call([args.bin_dir / "llama-cli", "-m", "missing-model", "--ridgefill", "missing-projector",
              "--ridgefill-mode", "quality+qattn", flag, value], "llama-cli-unsupported-" + flag, "unavailable in llama-cli")

    unsupported = (("LoRA", ["--lora", "missing-adapter"]),
                   ("control vectors", ["--control-vector", "missing-control"]),
                   ("embeddings", ["--embeddings"]),
                   ("--cache-reuse", ["--cache-reuse", 16]),
                   ("--split-mode row", ["-sm", "row"]))
    for tool in ("llama-server", "llama-perplexity", "llama-bench"):
        for feature, value in (("--ridgefill-s1-window", 64), ("--ridgefill-vtail", "on")):
            for message, extra in unsupported:
                if tool == "llama-bench" and message == "embeddings":
                    extra = [*extra, 1]
                call([args.bin_dir / tool, "-m", "missing-model", "--ridgefill", "missing-projector",
                      "--ridgefill-mode", "quality+qattn", feature, value, *extra], tool + "-" + feature + "-" + message.replace(" ", "-"), message)

    command = [args.bin_dir / "llama-server", "-m", model, "--ridgefill", projector,
               "--ridgefill-mode", "quality+qattn", "--ridgefill-alpha", 0,
               "--ridgefill-tail", 200, "--ridgefill-min-prompt", 1,
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
            props = client.request("/props")["ridgefill"]
            check([props[key] for key in ("s1_window", "s1_factor", "vtail", "vtail_rows")] == [0, 1, False, 512], "HTTP-default-settings")
            body = completion_body(ids, "quality+qattn", 200)
            body.update(n_probs=300, ignore_eos=True)
            implicit = client.request("/completion", body)
            explicit = client.request("/completion", dict(body, ridgefill_s1_window=0, ridgefill_s1_factor=1,
                                                          ridgefill_vtail=False, ridgefill_vtail_rows=512))
            for field in ("tokens", "completion_probabilities"):
                check(implicit[field] == explicit[field], "HTTP-byte-default-" + field)
            check(implicit["timings"]["ridgefill"] == explicit["timings"]["ridgefill"], "HTTP-byte-default-report")
            for name, value in (("ridgefill_s1_window", -1), ("ridgefill_s1_window", 1.5), ("ridgefill_s1_window", 2147483648),
                                ("ridgefill_s1_factor", -1), ("ridgefill_s1_factor", "1"), ("ridgefill_s1_factor", 1e300),
                                ("ridgefill_vtail", "on"), ("ridgefill_vtail_rows", 0), ("ridgefill_vtail_rows", 1.5)):
                try:
                    client.request("/completion", dict(body, **{name: value}))
                except EvalError as error:
                    check(name in str(error), "HTTP-bad-" + name + "-" + str(value))
                else:
                    raise AssertionError("HTTP accepted invalid " + name)
            for name, value in (("ridgefill_s1_window", 64), ("ridgefill_vtail", True)):
                try:
                    client.request("/completion", dict(body, ridgefill_mode="quality", **{name: value}))
                except EvalError as error:
                    check("two-sweep flow" in str(error), "HTTP-refuse-" + name)
                else:
                    raise AssertionError("HTTP accepted a feature without the two-sweep mode")
        finally:
            server.terminate()
            server.wait(timeout=10)
    summary = dict(result="PASS", scope="disabled defaults and named refusals; no enabled S1/vtail inference claim",
                   checks=len(checks), records=checks)
    (args.out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bin-dir", "baseline-binary", "fixture", "chat-fixture", "out"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--port", type=int, default=18136)
    run(parser.parse_args())
