#!/usr/bin/env python3
"""Check real row-set accounting and analytic MACs against kva.compute; compare native/HTTP/bench reporting."""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np
from safetensors import safe_open

from eval_client import Client, EvalError, completion_body, validate_ridgefill_report

ROOT = Path(__file__).resolve().parents[2]


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("ridgefill_compute_reference", args.kva / "kva/compute.py")
    assert spec is not None and spec.loader is not None
    compute = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compute)
    dims = compute.Dims(args.fixture / "tiny-qwen35-hf")
    # Derive the optional gate dimension from weights, independently of the report and an omitted config flag.
    with safe_open(args.fixture / "tiny-qwen35-hf/model.safetensors", framework="numpy") as weights:
        shape = weights.get_slice("model.layers.3.self_attn.q_proj.weight").get_shape()
    assert shape[0] in (dims.hq * dims.d, 2 * dims.hq * dims.d)
    dims.gated = shape[0] == 2 * dims.hq * dims.d
    model, projector = args.chat_fixture / "chat.gguf", args.chat_fixture / "projector-s4.gguf"
    ids = np.fromfile(args.fixture / "tokens.bin", dtype="<i4").tolist()
    header = [1, *[byte + 3 for byte in b"user\n"]]
    for pos in (20, 300):
        ids[pos:pos + len(header)] = header
    for pos, token in ((0, 1), (10, 2), (40, 0), (499, 2), (600, 1)):
        ids[pos] = token
    source, tokens = args.out / "tokens.jsonl", args.out / "tokens.bin"
    source.write_text(json.dumps(dict(name="chat", ids=ids)) + "\n")
    np.asarray(ids, dtype="<i4").tofile(tokens)
    records, native = [], {}

    def check(condition, name, **details):
        if not condition:
            raise AssertionError(name)
        records.append(dict(check=name, **details))

    def call(command, name):
        result = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=60)
        (args.out / (name + ".log")).write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stderr
        return result

    def selected(digits, special, headers):
        prefix = args.out / f"select-{digits}-{special}-{headers}"
        call([args.dump, "--model", model, "--projector", projector, "--tokens", tokens, "--output", prefix,
              "--tail", 200, "--keep", 1, "--select-share", 0.5, "--select-digits", int(digits),
              "--select-special", int(special), "--select-header", headers], prefix.name)
        return set(np.fromfile(str(prefix) + ".rows.i32", dtype="<i4").tolist())

    cases = (("off", "off", False, 0, "last", 256), ("speed", "speed", False, 0, "last", 256),
             ("quality", "quality", False, 0, "last", 256), ("chat", "quality", True, 3, "last", 256),
             ("qattn", "quality+qattn", True, 3, "last", 256),
             ("turn32", "quality+qattn", True, 3, "turn", 32),
             ("turn256", "quality+qattn", True, 3, "turn", 256),
             ("all-exact", "quality+qattn", True, 500, "last", 256))
    classes = selected(False, False, 0)
    for label, mode, special, headers, window, cap in cases:
        output = args.out / (label + ".jsonl")
        call([args.bin_dir / "llama-ridgefill-score", "--model", model, "--projector", projector,
              "--input", source, "--output", output, "--mode", mode, "--scored", 64, "--tail", 200,
              "--min-prompt", 1, "--cache-type", "f32", "--h-type", "f16", "--flash-attn", "off",
              "--batch", 128, "--ubatch", 128, "--qattn-window-end", "document", "--special-exact",
              "on" if special else "off", "--header-rows", headers, "--qattn-window-mode", window,
              "--qattn-turn-cap", cap], "native-" + label)
        row = json.loads(output.read_text())
        report = validate_ridgefill_report(row["ridgefill"], 700, mode)
        exact = set(row["exact_rows"])
        if mode in ("quality", "quality+qattn"):
            stages = [classes, selected(True, False, 0), selected(True, special, 0), selected(True, special, headers), exact]
            counts, previous = [], set()
            for stage in stages:
                assert previous <= stage
                counts.append(len(stage - previous))
                previous = stage
            want = dict(zip(("class", "digit", "special", "header", "qattn"), counts))
            check(report["n_exact_prefix_by_source"] == want, label + "-selector-ext-breakdown", counts=want)
        check(report["n_exact_prefix"] == len(exact) and report["n_filled"] == row["prefix"] - len(exact),
              label + "-filled-excludes-exact-rows")
        baseline = compute.exact(dims, 700)
        pos = (sum(value + 1 for value in exact) / len(exact)) / (row["prefix"] + 1) if exact else 0.5
        cost = compute.bypass(dims, 700, 4, 700 - row["prefix"], len(exact), pos=pos)
        if row["qattn_window"] is not None:
            begin, end = row["qattn_window"]
            cost += compute.qattn_select(dims, end, 4, window=end-begin)
        error = max(abs(report["macs_exact"] / baseline - 1), abs(report["macs_ridgefill"] / cost - 1))
        check(error < 1e-12 and abs(report["mac_share_skipped"] - (1-cost/baseline)) < 1e-12,
              label + "-kva-compute", relative_error=error)
        native[label] = row
    check(native["chat"]["ridgefill"]["n_exact_prefix"] > native["quality"]["ridgefill"]["n_exact_prefix"],
          "positive-structural-rows-reduce-skipped-tokens")
    check(native["off"]["ridgefill"]["tokens_skipped"] == native["off"]["ridgefill"]["mac_share_skipped"] == 0,
          "negative-off-skips-nothing")
    check(native["all-exact"]["ridgefill"]["n_filled"] == 0 and native["all-exact"]["ridgefill"]["n_exact_prefix"] == 500 and
          native["all-exact"]["ridgefill"]["mac_share_skipped"] < 0, "all-exact-still-charges-question-selector")

    client = Client(f"http://127.0.0.1:{args.port}", timeout=60)
    command = [args.bin_dir / "llama-server", "-m", model, "-a", "tiny", "--ridgefill", projector,
               "--ridgefill-mode", "quality+qattn", "--ridgefill-alpha", 0, "--ridgefill-tail", 200,
               "--ridgefill-min-prompt", 1, "-c", 1024, "-np", 1, "-b", 128, "-ub", 128, "-t", 4, "-ngl", 0,
               "-ctk", "f32", "-ctv", "f32", "-fa", "off", "--no-warmup", "--offline",
               "--chat-template", "chatml", "--host", "127.0.0.1", "--port", args.port]
    with (args.out / "server.log").open("w") as log:
        server = subprocess.Popen(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while True:
                assert server.poll() is None, "server exited during startup"
                try:
                    client.request("/health")
                    break
                except (urllib.error.URLError, EvalError):
                    assert time.monotonic() < deadline
                    time.sleep(0.05)
            for label, mode, special, headers, window, cap in cases:
                body: dict = completion_body(ids, mode, 200)
                body.update(ridgefill_special_exact=special, ridgefill_header_rows=headers,
                            ridgefill_qattn_window_mode=window, ridgefill_qattn_turn_cap=cap)
                response = client.request("/completion", body)
                report = validate_ridgefill_report(response["timings"]["ridgefill"], 700, mode)
                check(report == native[label]["ridgefill"], label + "-HTTP-native-report")
                body["cache_prompt"] = True
                cached = client.request("/completion", body)
                check(cached["timings"]["ridgefill"] == report, label + "-cache-coverage-report")
            client.request("/completion", completion_body(ids, "off", 200))
            cached = client.request("/completion", completion_body(ids, "speed", 200, cache=True))
            report = validate_ridgefill_report(cached["timings"]["ridgefill"], 700, "speed")
            check(cached["timings"]["cache_n"] >= 500 and report["n_filled"] == 0 and
                  report["n_cached_exact_prefix"] == 500 and report["mac_share_skipped"] == 0,
                  "exact-cache-is-not-reported-as-filled")
            client.request("/completion", completion_body(ids[:300], "off", 200))
            cached = client.request("/completion", completion_body(ids, "speed", 200, cache=True))
            report = validate_ridgefill_report(cached["timings"]["ridgefill"], 700, "speed")
            reused = cached["timings"]["cache_n"]
            check(0 < reused < 500 and report["n_cached_exact_prefix"] == reused and report["n_filled"] == 500-reused,
                  "partial-exact-cache-keeps-full-cost-rows")
            cost = compute.bypass(dims, 700, 4, 200, reused, pos=(reused + 1) / (2 * 501))
            check(abs(report["macs_ridgefill"] / cost - 1) < 1e-12, "partial-exact-cache-MACs-vs-kva")
            client.request("/completion", completion_body(ids, "off", 200))
            body = dict(prompt=ids, max_tokens=1, model="tiny", temperature=0, ridgefill_mode="quality", ridgefill_tail=200, cache_prompt=False)
            response = client.request("/v1/completions", body)
            check(response["usage"]["ridgefill"] == native["quality"]["ridgefill"], "OpenAI-usage-report")
            body.update(stream=True, stream_options=dict(include_usage=True))
            request = urllib.request.Request(client.url + "/v1/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
            usage = None
            with urllib.request.urlopen(request, timeout=60) as stream:
                for line in stream:
                    if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                        event = json.loads(line[5:])
                        if event.get("usage"):
                            usage = event["usage"]
            check(usage is not None and usage["ridgefill"] == native["quality"]["ridgefill"], "OpenAI-stream-usage-report")
            body = dict(model="tiny", messages=[dict(role="user", content="hello " * 100)], max_tokens=1,
                        temperature=0, ridgefill_mode="quality", ridgefill_tail=200, cache_prompt=False)
            response = client.request("/v1/chat/completions", body)
            chat_report = validate_ridgefill_report(response["usage"]["ridgefill"], response["usage"]["prompt_tokens"], "quality")
            check(chat_report["n_filled"] > 0, "OpenAI-chat-usage-report")
            body.update(stream=True, stream_options=dict(include_usage=True))
            request = urllib.request.Request(client.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
            usage = None
            with urllib.request.urlopen(request, timeout=60) as stream:
                for line in stream:
                    if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                        event = json.loads(line[5:])
                        if event.get("usage"):
                            usage = event["usage"]
            check(usage is not None and usage["ridgefill"] == chat_report, "OpenAI-chat-stream-usage-report")
            text = client.request("/detokenize", dict(tokens=ids))["content"]

        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    result = call([args.bin_dir / "llama-bench", "-m", model, "--ridgefill", projector,
                   "--ridgefill-mode", "quality", "--ridgefill-special-exact", "on", "--ridgefill-header-rows", 3,
                   "--ridgefill-tail", 200, "-p", 700, "-n", 0, "-r", 2, "-t", 4, "-ngl", 0,
                   "-b", 128, "-ub", 128, "-fa", "off", "--no-warmup", "-o", "json"], "bench")
    reports = [json.loads(line.removeprefix("RidgeFill: report ")) for line in result.stderr.splitlines()
               if line.startswith("RidgeFill: report ")]
    selected_counts = [int(value) for value in re.findall(r"mode quality selected (\d+) exact rows", result.stderr)]
    check(len(reports) == len(selected_counts) == 2, "bench-per-repetition-report")
    for index, (report, count) in enumerate(zip(reports, selected_counts)):
        validate_ridgefill_report(report, 700, "quality")
        check(report["n_exact_prefix"] == count and report["n_filled"] == 500-count, f"bench-row-count-{index}")
    document = args.out / "perplexity.txt"
    document.write_text((text + "\n") * 8)
    result = call([args.bin_dir / "llama-perplexity", "-m", model, "--ridgefill", projector,
                   "--ridgefill-mode", "quality", "--ridgefill-special-exact", "on", "--ridgefill-header-rows", 3,
                   "--ridgefill-tail", 200, "--ridgefill-min-prompt", 1, "--chunks", 1, "-f", document,
                   "-c", 512, "-b", 512, "-ub", 128, "-t", 4, "-ngl", 0, "-fa", "off"], "perplexity")
    reports = [json.loads(value) for value in re.findall(r"RidgeFill: report (\{[^\n]+\})", result.stderr)]
    counts = [int(value) for value in re.findall(r"mode quality selected (\d+) exact rows", result.stderr)]
    check(len(reports) == len(counts) == 1 and "Final estimate" in result.stderr, "perplexity-scored-chunk-report")
    report = validate_ridgefill_report(reports[0], 512, "quality")
    check(report["n_exact_prefix"] == counts[0] and report["n_filled"] == 256-counts[0], "perplexity-row-count")
    summary = dict(result="PASS", checks=len(records), mac_configs=len(cases), records=records)
    (args.out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bin-dir", "fixture", "chat-fixture", "dump", "kva", "out"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--port", type=int, default=18131)
    run(parser.parse_args())
