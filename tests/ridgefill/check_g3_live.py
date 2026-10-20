"""Run the one-command G3 pipeline on the tiny model and verify its measurements with direct logits."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from eval_client import needle_correct


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--kva", type=Path, required=True)
    parser.add_argument("--chat-template-file", type=Path, required=True, help="the Qwen3.8 template extracted from its GGUF")
    parser.add_argument("--fixture", type=Path, default=root / "out/gate1-qwen35")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18091)
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    fixture = args.fixture.resolve()
    model, projector = fixture / "tiny-qwen35-f32.gguf", fixture / "projector-s3-f32.gguf"
    original = np.fromfile(fixture / "tokens.bin", dtype=np.int32)
    docs: list[dict] = [dict(name=f"doc-{i}", domain="synthetic", ids=np.roll(original, i * 17).tolist()) for i in range(2)]
    needles = []
    for i, doc in enumerate(docs):
        value = str(1234567 + i)
        text = f" The code for test-{i} is {value}. What is the code for test-{i}? Reply with seven digits only."
        source = doc["ids"][:600] + [3 + byte for byte in text.encode()]
        answer = [3 + byte for byte in (value + "\n").encode()]
        needles.append(dict(name=f"needle-{i}", domain="synthetic", ids=source + answer, answer_len=len(answer), value=value))
    for name, rows in (("docs", docs), ("needles", needles)):
        (work / (name + ".jsonl")).write_text("".join(json.dumps(row) + "\n" for row in rows))
    checked = 0

    def check(condition, message):
        nonlocal checked
        if not condition:
            raise AssertionError(message)
        checked += 1

    expected = {}
    for i, doc in enumerate(docs):
        tokens = work / f"doc-{i}.bin"
        np.asarray(doc["ids"], dtype=np.int32).tofile(tokens)
        for mode in ("off", "speed"):
            prefix = work / f"native-{i}-{mode}"
            command = [str(args.dump.resolve()), "--model", str(model), "--projector", str(projector), "--tokens", str(tokens),
                       "--output", str(prefix), "--mode", mode, "--tail", "600", "--keep", "513", "--alpha", "0",
                       "--batch", "700", "--ubatch", "128"]
            with (work / f"native-{i}-{mode}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=30)
            logits = np.fromfile(str(prefix) + ".logits", dtype=np.float32).reshape(513, 300)[:-1].astype(np.float64)
            shift = np.max(logits, axis=1)
            normalizer = shift + np.log(np.exp(logits - shift[:, None]).sum(axis=1))
            expected[i, mode] = logits[np.arange(512), doc["ids"][-512:]] - normalizer

    summaries = {}
    max_error = 0.0
    for offset, mode in enumerate(("speed", "off")):
        output = work / mode
        command = [sys.executable, str(root / "tests/ridgefill/run_g3.py"), "--bin-dir", str(args.bin_dir.resolve()),
                   "--model", str(model), "--projector", str(projector), "--docs", str(work / "docs.jsonl"),
                   "--needles", str(work / "needles.jsonl"), "--out", str(output), "--kva", str(args.kva.resolve()),
                   "--chat-template-file", str(args.chat_template_file.resolve()), "--mode", mode, "--build-label", "tiny-cpu-control",
                   "--length", "700", "--lengths", "700", "--ctx-size", "4096", "--tail", "600", "--tail-min", "1",
                   "--min-prompt", "1", "--batch", "700", "--ubatch", "128", "--cache-type", "f32",
                   "--flash-attn", "off", "--no-warmup", "--port", str(args.port + offset)]
        (work / (mode + "-command.json")).write_text(json.dumps(command, indent=2) + "\n")
        with (work / (mode + ".log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        summary = json.loads((output / "summary.json").read_text())
        check(summary["result"] == "COMPLETE" and len(summary["summaries"]) == 5, "the command did not measure every stage")
        for row in summary["summaries"]:
            check(row["status"] == "MEASURED" and row["resamples"] == 5000, "missing measurement or BCa integration")
        summaries[mode] = summary["summaries"]
        rows = [row for row in map(json.loads, (output / "nll.jsonl").read_text().splitlines()) if row["kind"] == "nll"]
        check(len(rows) == 2, "wrong document count")
        for i, row in enumerate(rows):
            for arm, arm_mode in (("reference", "off"), ("candidate", mode)):
                observed = np.asarray(row[arm]["logprobs"])
                error = float(np.max(np.abs(observed - expected[i, arm_mode])))
                max_error = max(max_error, error)
                check(row[arm]["scored"] == 512 and error < 1e-5, "last-512 NLL differs from direct logits")
            if mode == "off":
                check(row["reference"]["logprobs"] == row["candidate"]["logprobs"], "off/off control differs")
            else:
                check(row["candidate"]["filled"] == 100, "NLL did not preserve and reuse the fixed fill boundary")
        rows = [row for row in map(json.loads, (output / "needles.jsonl").read_text().splitlines()) if row["kind"] == "needle"]
        check(len(rows) == 2, "wrong needle count")
        for row in rows:
            check(row["reference"]["prompt_sha256"] == row["candidate"]["prompt_sha256"], "needle arms used different prompts")
            check(all(not row[arm]["thinking"] and row[arm]["greedy"] and row[arm]["removed_answer_tokens"] == 8 for arm in ("reference", "candidate")),
                  "needle did not remove answer suffix or disable thinking")
            if mode == "speed":
                check(row["candidate"]["filled"] > 0, "needle candidate did not fill")
        if mode == "speed":
            nll = next(row for row in summary["summaries"] if row["metric"] == "nll")
            check(abs(nll["delta"]) > 1e-5, "speed positive control did not move NLL")
    for value in ("1234567", "0000001"):
        check(needle_correct("The answer is " + value + ".", value), "seven-digit positive control failed")
        check(not needle_correct("9" + value, value) and not needle_correct(value + "8", value), "longer-number negative control failed")
    print(json.dumps(dict(result="PASS", checks=checked, max_logprob_abs_error=max_error, documents=2, scored_per_document=512,
                          needles=2, modes=["speed", "off"], metrics=["nll", "needles", "ttft", "bench_prefill", "bench_decode"],
                          scope="tiny G3 instrumentation controls; no real-model quality claim", summaries=summaries)))


if __name__ == "__main__":
    main()
