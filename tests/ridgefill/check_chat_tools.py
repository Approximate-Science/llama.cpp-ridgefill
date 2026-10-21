"""Tiny CPU chat-row plumbing gate; the synthetic conversation is not a frozen chat data set."""

import argparse
import json
import math
import re
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

import numpy as np

from check_g3_live import class_digit_rows
from eval_client import Client, EvalError, completion_body, target_logprob


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--kva", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=root / "out/gate1-qwen35")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18138)
    args = parser.parse_args()
    args.bin_dir, args.fixture, args.out = args.bin_dir.resolve(), args.fixture.resolve(), args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    model = args.fixture / "tiny-qwen35-f32.gguf"
    projector = args.fixture / "projector-s3-f32.gguf"
    ids = np.fromfile(args.fixture / "tokens.bin", dtype=np.int32).tolist()
    if len(ids) != 700:
        raise ValueError("chat tool gate requires the 700-token, 300-vocabulary tiny fixture")
    # The tiny vocabulary uses <s>/</s> as synthetic turn markers and byte tokens for roles.
    for begin, role, end in ((0, "user", 90), (100, "assistant", 190), (200, "user", 310),
                             (320, "assistant", 390), (400, "user", 510), (520, "assistant", None)):
        ids[begin] = 1
        ids[begin + 1:begin + len(role) + 2] = [3 + byte for byte in (role + "\n").encode()]
        if end is not None:
            ids[end] = 2
    ids[499] = 0
    docs: list[dict] = [dict(name="conversation-0", ids=ids), dict(name="conversation-1", ids=ids[:])]
    docs[1]["ids"][630:636] = [270, 272, 274, 276, 278, 280]
    source = args.out / "conversations.jsonl"
    source.write_text("".join(json.dumps(doc) + "\n" for doc in docs))
    checks = 0

    def check(condition, message):
        nonlocal checks
        if not condition:
            raise AssertionError(message)
        checks += 1

    def command(argv, label, refused=None):
        result = subprocess.run(list(map(str, argv)), capture_output=True, text=True, timeout=180)
        (args.out / (label + ".stdout")).write_text(result.stdout)
        (args.out / (label + ".stderr")).write_text(result.stderr)
        if refused:
            check(result.returncode != 0 and refused in result.stdout + result.stderr, label + " missed named refusal")
        elif result.returncode:
            raise RuntimeError(f"{label}: see {args.out / (label + '.stderr')}")
        return result

    expected = {}
    base = set(class_digit_rows(root, args.fixture, ids, 500))
    specials = [pos for pos in range(500) if ids[pos] in (0, 1, 2)]
    native_common = ["--model", model, "--projector", projector, "--input", source, "--scored", "1", "--tail", "200",
                     "--min-prompt", "1", "--batch", "700", "--ubatch", "128", "--cache-type", "f32", "--flash-attn", "off"]
    for label, special, header in (("base", False, 0), ("special", True, 0), ("headers", False, 2), ("both", True, 2)):
        output = args.out / ("score-" + label + ".jsonl")
        command([args.bin_dir / "llama-ridgefill-score", *native_common, "--mode", "quality", "--output", output,
                 "--ridgefill-special-exact", "on" if special else "off", "--ridgefill-header-rows", header], "score-" + label)
        row = json.loads(output.read_text().splitlines()[0])
        selected = base | (set(specials) if special else set())
        selected.update(pos for begin in specials for pos in range(begin + 1, min(500, begin + header + 1)))
        check(row["exact_rows"] == sorted(selected), label + " differs from independent class/digit/special/header rows")
        check(row["parameters"]["special_exact"] == special and row["parameters"]["header_rows"] == header,
              label + " scorer did not record its parameters")
        expected[label] = row["logprobs"][0]
    check(len(base | set(specials)) > len(base), "special-row positive control did not change selection")
    check(abs(expected["both"] - expected["base"]) > 1e-7, "chat-row positive control did not change logits")
    qattn_common = native_common[:]
    qattn_common[qattn_common.index("--projector") + 1] = args.fixture / "projector-s4-f32.gguf"
    qattn_common[qattn_common.index("--scored") + 1] = "64"
    qattn = {}
    for label, special, header in (("base", False, 0), ("both", True, 2)):
        output = args.out / ("qattn-" + label + ".jsonl")
        command([args.bin_dir / "llama-ridgefill-score", *qattn_common, "--mode", "quality+qattn", "--output", output,
                 "--qattn-window-end", "known", "--special-exact", "on" if special else "off", "--header-rows", header,
                 "--diagnostics"], "qattn-" + label)
        qattn[label] = json.loads(output.read_text().splitlines()[0])
    chat_rows = set(specials) | {pos for begin in specials for pos in range(begin + 1, min(500, begin + 3))}
    check(qattn["both"]["exact_rows"] == sorted(set(qattn["base"]["exact_rows"]) | chat_rows),
          "question sweep did not union chat rows with unchanged question rows")
    check(qattn["both"]["question_mass"] == qattn["base"]["question_mass"], "chat rows changed lower question mass")
    for mode in ("speed", "off"):
        output = args.out / ("score-" + mode + ".jsonl")
        command([args.bin_dir / "llama-ridgefill-score", *native_common, "--mode", mode, "--output", output], "score-" + mode)
        expected[mode] = json.loads(output.read_text().splitlines()[0])["logprobs"][0]

    for tool in ("llama-bench", "llama-perplexity"):
        common = [args.bin_dir / tool, "-m", model, "--ridgefill", projector, "--ridgefill-tail", "200",
                  "--ridgefill-alpha", "0", "-ngl", "0", "-t", "4", "-b", "512", "-ub", "128"]
        for flag, value in (("--ridgefill-special-exact", "on"), ("--ridgefill-header-rows", "2")):
            command([*common, "--ridgefill-mode", "speed", flag, value], tool + "-refuse-" + flag, "require quality")
        command([*common, "--ridgefill-mode", "quality", "--ridgefill-qattn-window-mode", "turn"],
                tool + "-refuse-turn", "quality+qattn")
        command([*common, "--ridgefill-mode", "quality", "--qattn-turn-cap", "0"],
                tool + "-refuse-turn-cap", "--ridgefill-qattn-turn-cap")

    server_args = [args.bin_dir / "llama-server", "-m", model, "--ridgefill", projector, "--ridgefill-mode", "quality",
                   "--ridgefill-special-exact", "on", "--ridgefill-header-rows", "2", "--ridgefill-alpha", "0",
                   "--ridgefill-tail", "200", "--ridgefill-min-prompt", "1", "-c", "1024", "-np", "1", "-b", "700",
                   "-ub", "128", "-t", "4", "-ngl", "0", "-ctk", "f32", "-ctv", "f32", "-fa", "off",
                   "--no-warmup", "--offline", "--host", "127.0.0.1", "--port", str(args.port)]
    client = Client(f"http://127.0.0.1:{args.port}", timeout=60)
    with (args.out / "server.log").open("w") as log:
        server = subprocess.Popen(list(map(str, server_args)), stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while True:
                if server.poll() is not None:
                    raise RuntimeError("chat test server exited; see server.log")
                try:
                    client.request("/health")
                    break
                except (urllib.error.URLError, EvalError):
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.1)
            props = client.request("/props")["ridgefill"]
            check(props["special_exact"] and props["header_rows"] == 2 and props["qattn_window_mode"] == "last",
                  "startup chat options missing from props")
            for label, special, header, cache in (("both", None, None, False), ("base", False, 0, True),
                    ("special", True, 0, True), ("headers", False, 2, True), ("both", True, 2, True),
                    ("both", True, 2, True)):
                body = completion_body(ids[:-1], "quality", 199, cache=cache)
                body.update(n_probs=1, logit_bias=[[ids[-1], 1000000.0]])
                if special is not None:
                    body.update(ridgefill_special_exact=special, ridgefill_header_rows=header)
                response = client.request("/completion", body)
                check(abs(target_logprob(response, ids[-1]) - expected[label]) < 1e-5,
                      label + " per-request selection/cache differs from native scorer")
            for mode in ("speed", "off"):
                body = completion_body(ids[:-1], mode, 199, cache=True)
                body.update(ridgefill_special_exact=False if mode == "speed" else True,
                            ridgefill_header_rows=0 if mode == "speed" else 2,
                            n_probs=1, logit_bias=[[ids[-1], 1000000.0]])
                response = client.request("/completion", body)
                check(abs(target_logprob(response, ids[-1]) - expected[mode]) < 1e-5,
                      mode + " did not override startup chat options with the expected control")
            for fields, message in ((dict(ridgefill_mode="speed", ridgefill_special_exact=True), "require quality"),
                    (dict(ridgefill_header_rows=-1), "ridgefill_header_rows"),
                    (dict(ridgefill_special_exact="on"), "ridgefill_special_exact"),
                    (dict(ridgefill_qattn_window_mode="bad"), "ridgefill_qattn_window_mode"),
                    (dict(ridgefill_mode="quality+qattn", ridgefill_qattn_window_mode="turn"), "no user header")):
                body = completion_body(ids[:-1], "quality", 199)
                body.update(fields)
                try:
                    client.request("/completion", body)
                except EvalError as error:
                    check(message in str(error), "request refusal did not name " + message)
                else:
                    raise AssertionError("unsupported request unexpectedly succeeded")
            text = client.request("/detokenize", dict(tokens=ids))["content"]
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()

    document = args.out / "perplexity.txt"
    document.write_text((text + "\n") * 8)
    for tool in ("llama-bench", "llama-perplexity"):
        common = [args.bin_dir / tool, "-m", model, "--ridgefill", projector, "--ridgefill-mode", "quality",
                  "--ridgefill-special-exact", "on", "--ridgefill-header-rows", "2", "--ridgefill-alpha", "0",
                  "--ridgefill-tail", "200", "-ngl", "0", "-t", "4", "-b", "512", "-ub", "128", "-fa", "off"]
        extra = ["-p", "700", "-n", "0", "-r", "1", "-o", "json"] if tool == "llama-bench" else [
            "-c", "512", "--ridgefill-min-prompt", "1", "--chunks", "1", "-f", document]
        result = command([*common, *extra], tool + "-quality-chat")
        check("mode quality selected" in result.stderr, tool + " did not select rows")
        if tool == "llama-bench":
            check(math.isfinite(json.loads(result.stdout)[0]["avg_ts"]), "bench did not complete a finite measurement")
        else:
            check(re.search(r"Final estimate: PPL = [0-9.]+", result.stdout + result.stderr) is not None,
                  "perplexity scored no chunk")

    command([sys.executable, root / "tools/ridgefill/run_nll.py", "--binary", args.bin_dir / "llama-ridgefill-score",
             "--model", model, "--projector", projector, "--docs", source, "--kva", args.kva, "--out", args.out / "paired",
             "--mode", "quality", "--special-exact", "on", "--header-rows", "2", "--length", "700", "--scored", "64",
             "--tail", "200", "--min-prompt", "1", "--cache-type", "f32", "--batch", "700", "--ubatch", "128",
             "--flash-attn", "off"], "paired-wrapper")
    summary = json.loads((args.out / "paired/summary.json").read_text())
    check(summary["result"] == "COMPLETE" and summary["mode"] == "quality", "paired wrapper did not complete quality")
    native = [json.loads(line) for line in (args.out / "paired/quality.jsonl").read_text().splitlines()]
    check(all(row["parameters"]["special_exact"] and row["parameters"]["header_rows"] == 2 for row in native),
          "paired wrapper lost chat selector flags")
    result = dict(result="PASS", checks=checks, scope="tiny CPU special/header tool plumbing; missing turn marker refuses by name")
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
