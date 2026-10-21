#!/usr/bin/env python3
"""Check live S1/vtail HTTP settings, costs and cache transitions against native scoring and kva.compute."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import urllib.error

import numpy as np
from safetensors import safe_open

from eval_client import Client, EvalError, completion_body, target_logprob, validate_ridgefill_report


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("s1_http_compute", args.kva / "kva/compute.py")
    assert spec is not None and spec.loader is not None
    compute = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compute)
    dims = compute.Dims(args.fixture / "tiny-qwen35-hf")
    with safe_open(args.fixture / "tiny-qwen35-hf/model.safetensors", framework="numpy") as weights:
        dims.gated = weights.get_slice("model.layers.3.self_attn.q_proj.weight").get_shape()[0] == 2 * dims.hq * dims.d
    model, projector = args.chat_fixture / "chat.gguf", args.chat_fixture / "projector-s4.gguf"
    ids = np.fromfile(args.fixture / "tokens.bin", dtype="<i4").tolist()
    for pos in (20, 300, 600):
        header = [1, *[byte + 3 for byte in b"user\n"]]
        ids[pos:pos + len(header)] = header
    source = args.out / "tokens.jsonl"
    source.write_text(json.dumps(dict(name="chat", ids=[*ids, 7])) + "\n")
    records = []

    def check(condition, label, **details):
        assert condition, label
        records.append(dict(check=label, **details))

    for qshare in (0.05, 0.0):
        command = [args.bin_dir / "llama-server", "-m", model, "--ridgefill", projector,
                   "--ridgefill-mode", "quality+qattn", "--ridgefill-alpha", 0,
                   "--ridgefill-tail", 200, "--ridgefill-min-prompt", 1, "--ridgefill-share", 0,
                   "--ridgefill-digits", "off", "--ridgefill-qattn-share", qshare,
                   "-c", 1024, "-np", 1, "-b", 512, "-ub", 512, "-t", 4, "-ngl", 0,
                   "-ctk", "f32", "-ctv", "f32", "-fa", "off", "--offline", "--no-warmup",
                   "--host", "127.0.0.1", "--port", args.port]
        client = Client(f"http://127.0.0.1:{args.port}", timeout=60)
        with (args.out / f"server-{qshare}.log").open("w") as log:
            server = subprocess.Popen(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 60
                while True:
                    assert server.poll() is None, "server exited; see server log"
                    try:
                        client.request("/health")
                        break
                    except (urllib.error.URLError, EvalError):
                        assert time.monotonic() < deadline, "startup timeout"
                        time.sleep(0.05)
                cases = [("base", 0, 1, False, False, 512, 0), ("s1", 64, 1, False, False, 512, 0),
                         ("factor-tie", 64, 0.1, False, False, 512, 0), ("special", 64, 0.5, True, False, 512, 0.5),
                         ("s1-vtail", 64, 1, True, True, 128, 0), ("vtail", 0, 1, False, True, 32, 0),
                         ("saturated", 64, 100, False, False, 512, 0)] if qshare else [
                         ("vtail-only", 0, 1, False, True, 32, 0), ("rescale-off", 0, 1, False, False, 512, 0),
                         ("vtail-change", 0, 1, False, True, 64, 0), ("rescale-off-again", 0, 1, False, False, 512, 0)]
                for name, width, factor, special, vtail, count, share in cases:
                    label = f"{qshare}-{name}"
                    output = args.out / (label + ".jsonl")
                    native_command = [args.bin_dir / "llama-ridgefill-score", "--model", model, "--projector", projector,
                                      "--input", source, "--output", output, "--mode", "quality+qattn", "--scored", 1,
                                      "--tail", 201, "--min-prompt", 1, "--cache-type", "f32", "--h-type", "f16",
                                      "--flash-attn", "off", "--batch", 512, "--ubatch", 512, "--qattn-window-end", 700,
                                      "--share", share, "--digits", "off", "--qattn-share", qshare,
                                      "--special-exact", "on" if special else "off", "--ridgefill-s1-window", width,
                                      "--ridgefill-s1-factor", factor, "--ridgefill-vtail", "on" if vtail else "off",
                                      "--ridgefill-vtail-rows", count]
                    native = subprocess.run(list(map(str, native_command)), capture_output=True, text=True, timeout=60)
                    output.with_suffix(".log").write_text(native.stdout + native.stderr)
                    assert native.returncode == 0, native.stderr
                    expected = json.loads(output.read_text())
                    body = completion_body(ids, "quality+qattn", 200, cache=True)
                    body.update(n_probs=300, logit_bias=[[7, 1000000.0]], ignore_eos=True, ridgefill_share=share,
                                ridgefill_s1_window=width, ridgefill_s1_factor=factor, ridgefill_special_exact=special,
                                ridgefill_vtail=vtail, ridgefill_vtail_rows=count)
                    response = client.request("/completion", body)
                    report = validate_ridgefill_report(response["timings"]["ridgefill"], 700, "quality+qattn")
                    error = abs(target_logprob(response, 7) - expected["logprobs"][0])
                    check(error < 1e-4, label + "-native-logits", error=error)
                    for field in ("n_exact_prefix", "n_exact_prefix_by_source", "n_filled"):
                        check(report[field] == expected["ridgefill"][field], label + "-" + field)
                    if name == "factor-tie":
                        check(expected["s1"]["n_base"] == 25 and report["n_exact_prefix"] == 2, "HTTP-double-half-even-factor")
                    if name == "saturated":
                        check(report["n_filled"] == 0 and report["tokens_skipped"] == 0, "HTTP-saturated-no-token-savings")
                    positions = sum(row + 1 for row in expected["exact_rows"]) / len(expected["exact_rows"]) / 501 if expected["exact_rows"] else 0.5
                    want = compute.bypass(dims, 700, 4, 200, len(expected["exact_rows"]), pos=positions)
                    if qshare:
                        want += compute.qattn_select(dims, 700, 4, window=64)
                    if width:
                        extra = report["s1"]
                        probe = sum(dims.layer_exact(i, 0, width) for i in range(4, dims.L)) - dims.attn_ctx(0, width)
                        probe += 2 * dims.hq * dims.d * extra["probe_kv_pairs"]
                        fill = 500 * (dims.H**2 + dims.attn_fill())
                        check(extra["window"] == [700-width, 700] and extra["macs_fill"] == fill and extra["macs_probe"] == probe,
                              label + "-S1-spent-cost")
                        want += fill + probe
                    if vtail:
                        extra = count * (dims.H**2 + 2 * dims.H * dims.hkv * dims.d)
                        check(report["vtail"]["rows"] == count and report["vtail"]["macs_measure"] == extra, label + "-vtail-spent-cost")
                        want += extra
                    check(abs(report["macs_ridgefill"]-want)/want < 1e-12, label + "-independent-total-MACs")
                    if not qshare and not vtail:
                        check(response["timings"]["cache_n"] == 0, label + "-rescaled-cache-invalidated")
                    (args.out / (label + "-http.json")).write_text(json.dumps(response) + "\n")
                invalid = [(dict(ridgefill_s1_window=201), "S1 window" if qshare else "positive --ridgefill-qattn-share"),
                           (dict(ridgefill_vtail=True, ridgefill_vtail_rows=200), "deferred final token")]
                for override, message in invalid:
                    body = completion_body(ids, "quality+qattn", 200)
                    body.update(override)
                    try:
                        client.request("/completion", body)
                    except EvalError as error:
                        check(message in str(error), "HTTP-refuse-" + str(qshare) + "-" + message)
                    else:
                        raise AssertionError("accepted invalid S1/vtail request")
            finally:
                server.terminate()
                server.wait(timeout=10)
    summary = dict(result="PASS", checks=len(records), records=records)
    (args.out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bin-dir", "fixture", "chat-fixture", "kva", "out"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--port", type=int, default=18138)
    run(parser.parse_args())
