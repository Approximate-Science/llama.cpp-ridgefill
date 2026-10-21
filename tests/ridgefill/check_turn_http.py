#!/usr/bin/env python3
"""Compare HTTP chat selector/window overrides with fresh native scoring on the tiny ChatML fixture."""

import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.error

import numpy as np

from eval_client import Client, EvalError, completion_body, target_logprob


def run(args):
    args.work.mkdir(parents=True, exist_ok=False)
    model, projector = args.chat_fixture / "chat.gguf", args.chat_fixture / "projector-s4.gguf"
    raw = np.fromfile(args.fixture / "tokens.bin", dtype="<i4").tolist()
    assert len(raw) == 700
    header = [1, *[byte + 3 for byte in b"user\n"]]
    ids = raw.copy()
    for pos in (20, 300, 600):
        ids[pos:pos + len(header)] = header
    target = 7
    source = args.work / "next-token.jsonl"
    source.write_text(json.dumps(dict(name="chat", ids=[*ids, target])) + "\n")
    checks = []

    def check(condition, name, **details):
        if not condition:
            raise AssertionError(name)
        checks.append(dict(check=name, **details))

    def expected(mode, cap, special, headers):
        output = args.work / f"native-{mode}-{cap}-{special}-{headers}.jsonl"
        command = [args.bin_dir / "llama-ridgefill-score", "--model", model, "--projector", projector,
                   "--input", source, "--output", output, "--mode", "quality+qattn", "--scored", 1, "--tail", 201,
                   "--min-prompt", 1, "--cache-type", "f32", "--h-type", "f16", "--flash-attn", "off",
                   "--batch", 128, "--ubatch", 128, "--qattn-window-end", 700,
                   "--special-exact", "on" if special else "off", "--header-rows", headers,
                   "--qattn-window-mode", mode, "--qattn-turn-cap", cap]
        result = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=60)
        output.with_suffix(".log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stderr
        record = json.loads(output.read_text())
        begin = max(600 + len(header), 700 - cap) if mode == "turn" else 636
        check(record["qattn_window"] == [begin, 700], "native-window-" + output.stem)
        return record["logprobs"][0]

    command = [args.bin_dir / "llama-server", "-m", model, "--ridgefill", projector,
               "--ridgefill-mode", "quality+qattn", "--ridgefill-alpha", 0, "--ridgefill-tail", 200,
               "--ridgefill-min-prompt", 1, "--ridgefill-special-exact", "on", "--ridgefill-header-rows", 2,
               "--qattn-window-mode", "turn", "--qattn-turn-cap", 32,
               "-c", 1024, "-np", 1, "-b", 128, "-ub", 128, "-t", 4, "-ngl", 0,
               "-ctk", "f32", "-ctv", "f32", "-fa", "off", "--offline", "--no-warmup",
               "--host", "127.0.0.1", "--port", args.port]
    client = Client(f"http://127.0.0.1:{args.port}", timeout=60)
    with (args.work / "server.log").open("w") as log:
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
            check(props["qattn_window_mode"] == "turn" and props["qattn_turn_cap"] == 32 and
                  props["special_exact"] and props["header_rows"] == 2, "startup-inheritance")
            responses = []
            cases = (("turn", 32, True, 2, {}),
                     ("turn", 256, True, 2, dict(ridgefill_qattn_turn_cap=256)),
                     ("turn", 32, False, 0, dict(ridgefill_special_exact=False, ridgefill_header_rows=0)),
                     ("turn", 32, False, 3, dict(ridgefill_special_exact=False, ridgefill_header_rows=3)),
                     ("last", 7, True, 2, dict(ridgefill_qattn_window_mode="last", ridgefill_qattn_turn_cap=7)),
                     ("last", 256, True, 2, dict(ridgefill_qattn_window_mode="last", ridgefill_qattn_turn_cap=256)))
            for index, (mode, cap, special, headers, overrides) in enumerate(cases):
                body = completion_body(ids, "quality+qattn", 200, cache=True)
                body.update(overrides, n_probs=1, logit_bias=[[target, 1000000.0]], ignore_eos=True)
                response = client.request("/completion", body)
                error = abs(target_logprob(response, target) - expected(mode, cap, special, headers))
                check(error < 1e-5 and response["timings"]["prompt_filled_n"] == 500,
                      f"HTTP-native-{index}", absolute_logprob_error=error)
                responses.append(response)
            check(responses[-2]["completion_probabilities"] == responses[-1]["completion_probabilities"],
                  "last-mode-ignores-turn-cap")
            invalid = ((dict(prompt=raw), "no user header"),
                       (dict(prompt=[*raw[:-len(header)], *header]), "empty user turn"),
                       (dict(ridgefill_qattn_turn_cap=0), "ridgefill_qattn_turn_cap"),
                       (dict(ridgefill_qattn_turn_cap=-1), "ridgefill_qattn_turn_cap"),
                       (dict(ridgefill_qattn_turn_cap=1.5), "ridgefill_qattn_turn_cap"),
                       (dict(ridgefill_qattn_turn_cap=2**32 + 32), "ridgefill_qattn_turn_cap"),
                       (dict(ridgefill_header_rows=0.5), "ridgefill_header_rows"),
                       (dict(ridgefill_header_rows=2**32 + 2), "ridgefill_header_rows"),
                       (dict(ridgefill_qattn_window_mode="bad"), "ridgefill_qattn_window_mode"),
                       (dict(ridgefill_mode="quality"), "turn requires quality+qattn"))
            for overrides, message in invalid:
                body = completion_body(ids, "quality+qattn", 200)
                body.update(overrides)
                try:
                    client.request("/completion", body)
                except EvalError as error:
                    check(message in str(error), "named-refusal-" + message)
                else:
                    raise AssertionError("unsupported request succeeded")
            body = completion_body(ids, "quality+qattn", 200)
            body.update(n_probs=1, logit_bias=[[target, 1000000.0]], ignore_eos=True)
            recovered = client.request("/completion", body)
            check(recovered["completion_probabilities"] == responses[0]["completion_probabilities"],
                  "request-after-refusals-recovers-defaults")
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    result = dict(result="PASS", checks=len(checks), records=checks)
    (args.work / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bin-dir", "fixture", "chat-fixture", "work"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--port", type=int, default=18126)
    print(json.dumps(run(parser.parse_args())), flush=True)
