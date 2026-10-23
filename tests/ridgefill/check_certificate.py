#!/usr/bin/env python3
"""Short CPU certificate controls, native/HTTP agreement and accepted-parent byte checks."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

from eval_client import validate_ridgefill_report


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    ids = json.loads(args.input.read_text())["ids"]
    checks = []
    native_input = args.out / "native-input.json"
    native_input.write_text(json.dumps(dict(ids=[*ids, ids[-1]])) + "\n")

    def check(ok, label):
        assert ok, label
        checks.append(label)

    common = ["--model", str(args.model), "--projector", str(args.projector), "--input", str(native_input),
              "--scored", "1", "--tail", "401", "--min-prompt", "1", "--mode", "quality+qattn",
              "--share", "0", "--digits", "off", "--qattn-share", "0.2", "--qattn-window", "64",
              "--qattn-window-end", "known", "--alpha", "0", "--ridgefill-s1-window", "64",
              "--batch", "128", "--ubatch", "128", "--threads", "4", "--cache-type", "f32",
              "--h-type", "f16", "--flash-attn", "off", "--diagnostics"]

    def native(label, extra, parent=False, error=None):
        bundle = args.parent_bin if parent else args.bin_dir
        output = args.out / (label + ".jsonl")
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = str(bundle.resolve())
        stream = ["--ridgefill-stream-projector", "on"] if args.stream_projector and not parent else []
        command = [str(bundle / "llama-ridgefill-score"), *common, *extra, *stream, "--output", str(output)]
        process = subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)
        (args.out / (label + ".log")).write_text(process.stdout + process.stderr)
        if error:
            check(process.returncode != 0 and error in process.stderr, label)
            return None
        check(process.returncode == 0, label + "-completed")
        return json.loads(output.read_text())

    baseline = native("baseline", [])
    for mode in ("off", "quality+qattn"):
        a = native("parent-" + mode, ["--mode", mode], parent=True)
        b = native("new-" + mode, ["--mode", mode])
        for field in ("logprobs", "logsumexp", "exact_rows", "ridgefill"):
            value = b[field]
            if field == "ridgefill" and args.stream_projector:
                value = dict(value)
                check(value.pop("projector_streamed", False) == (mode != "off"), mode + "-streaming-report")
            check(a[field] == value, mode + "-byte-identity-" + field)
    negative = native("negative", ["--ridgefill-certificate", "on", "--ridgefill-certificate-tau", "1"])
    check(negative["exact_rows"] == baseline["exact_rows"], "tau1-retains-rows")
    check(negative["logprobs"] == baseline["logprobs"], "tau1-logits-identical")
    cert = negative["ridgefill"]["certificate"]
    check(cert["flagged"] == 0 and cert["n_added"] == 0 and not cert["capped"], "tau1-no-flags")
    check(cert["macs"] == 64*64*(64+32) + 4*16*64*negative["n"], "independent-certificate-MACs")
    positive = native("positive", ["--ridgefill-certificate", "on", "--ridgefill-certificate-tau", "0"])
    cert = positive["ridgefill"]["certificate"]
    check(cert["flagged"] == positive["prefix"] and cert["capped"] and not cert["complete"], "tau0-flags-prefix-and-caps")
    check(cert["n_added"] > 0 and len(positive["exact_rows"]) == cert["row_cap"], "cap-adds-rows")
    check(set(baseline["exact_rows"]).issubset(positive["exact_rows"]), "base-rows-retained")
    check(cert["dropped"] == positive["prefix"] - len(positive["exact_rows"]), "dropped-count")
    check(positive["ridgefill"]["n_exact_prefix_by_source"]["certificate"] == cert["n_added"], "source-attribution")
    check(positive["logprobs"] != baseline["logprobs"], "positive-changes-output")
    check(cert["probe_rows"] == 64 and cert["layers"] == [7] and len(cert["beta"][0]) == 2 and min(cert["beta"][0]) > 0, "per-head-calibration")
    check(positive["ridgefill"]["s1"]["window"][1] == len(ids) and positive["selector_excludes_targets"], "target-excluded")
    multi = native("multi-ubatch", ["--ridgefill-certificate", "on", "--ridgefill-s1-window", "256"])
    check(multi["ridgefill"]["certificate"]["probe_rows"] == 256, "multi-ubatch-calibration")
    native("base-over-cap", ["--ridgefill-certificate", "on", "--ridgefill-certificate-cap", "0.1"], error="existing exact rows exceed")
    native("missing-probe", ["--ridgefill-certificate", "on", "--ridgefill-s1-window", "0"], error="positive S1 window")
    native("bad-tau", ["--ridgefill-certificate-tau", "nan"], error="finite and in")
    native("bad-cap", ["--ridgefill-certificate-cap", "1.1"], error="finite and in")

    # Match HTTP's full prompt endpoint; this parity control does not measure NLL quality.
    parity = native("positive-http-endpoint", ["--input", str(args.input), "--tail", "400",
                    "--qattn-window-end", "document", "--ridgefill-certificate", "on", "--ridgefill-certificate-tau", "0"])
    url = "http://127.0.0.1:" + str(args.port)

    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)

    server_base = ["-m", str(args.model), "--ridgefill", str(args.projector), "--ridgefill-mode", "quality+qattn",
                   "--ridgefill-share", "0", "--ridgefill-digits", "off", "--ridgefill-alpha", "0",
                   "--ridgefill-s1-window", "64", "--ridgefill-tail", "400", "--ridgefill-min-prompt", "1",
                   "--ridgefill-qattn-share", "0.2", "-c", "2304", "-np", "1", "-b", "128", "-ub", "128",
                   "-t", "4", "-ngl", "0", "-ctk", "f32", "-ctv", "f32", "-fa", "off", "--no-warmup",
                   "--cache-ram", "0", "--host", "127.0.0.1", "--port", str(args.port)]
    responses = {}
    for arm, bundle in (("parent", args.parent_bin), ("new", args.bin_dir)):
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = str(bundle.resolve())
        with (args.out / ("http-" + arm + ".log")).open("w") as log:
            stream = ["--ridgefill-stream-projector", "on"] if args.stream_projector and arm == "new" else []
            proc = subprocess.Popen([str(bundle / "llama-server"), *server_base, *stream], stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                for _ in range(1200):
                    assert proc.poll() is None, "server exited; see " + arm + " log"
                    try:
                        request("/health")
                        break
                    except urllib.error.URLError:
                        time.sleep(0.05)
                else:
                    raise RuntimeError("server startup timeout")
                cases = [("off", dict(ridgefill_mode="off")), ("s1", {})]
                if arm == "new":
                    cases += [("negative", dict(ridgefill_certificate=True, ridgefill_certificate_tau=1)),
                              ("positive", dict(ridgefill_certificate=True, ridgefill_certificate_tau=0)),
                              ("off-again", {})]
                responses[arm] = {}
                for label, extra in cases:
                    body = dict(prompt=ids, n_predict=4, n_probs=300, temperature=0, seed=123,
                                ignore_eos=True, cache_prompt=False, **extra)
                    response = request("/completion", body)
                    validate_ridgefill_report(response["timings"]["ridgefill"], len(ids), extra.get("ridgefill_mode", "quality+qattn"))
                    (args.out / ("http-" + arm + "-" + label + ".json")).write_text(json.dumps(response) + "\n")
                    responses[arm][label] = {key: response[key] for key in ("content", "tokens", "completion_probabilities")}
                    if label in ("positive", "negative"):
                        observed = response["timings"]["ridgefill"]["certificate"]
                        expected = (parity if label == "positive" else negative)["ridgefill"]["certificate"]
                        for key in ("flagged", "n_added", "added", "dropped", "capped", "row_cap", "beta"):
                            check(observed[key] == expected[key], "HTTP-native-" + label + "-" + key)
                if arm == "new":
                    for extra, message in ((dict(ridgefill_certificate=True, ridgefill_s1_window=0), "positive S1 window"),
                                           (dict(ridgefill_certificate_cap=1.1), "[0, 1]"),
                                           (dict(ridgefill_certificate="yes"), "boolean")):
                        try:
                            request("/completion", dict(prompt=ids, n_predict=1, **extra))
                        except urllib.error.HTTPError as error:
                            detail = error.read().decode()
                            check(message in detail, "HTTP-refusal-" + message)
                        else:
                            raise AssertionError("accepted invalid HTTP certificate options")
            finally:
                proc.terminate()
                proc.wait(timeout=15)
    for mode in ("off", "s1"):
        check(responses["parent"][mode] == responses["new"][mode], "HTTP-full-vocabulary-byte-identity-" + mode)
    check(responses["new"]["s1"] == responses["new"]["negative"], "HTTP-tau1-no-op")
    check(responses["new"]["s1"] == responses["new"]["off-again"], "HTTP-next-request-no-leak")
    summary = dict(result="PASS", checks=len(checks), records=checks)
    (args.out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bin-dir", "parent-bin", "model", "projector", "input", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--port", type=int, default=18304)
    parser.add_argument("--stream-projector", action="store_true", help="Check streamed new tools against the resident parent")
    run(parser.parse_args())
