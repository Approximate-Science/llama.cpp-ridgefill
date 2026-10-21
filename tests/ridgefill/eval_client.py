#!/usr/bin/env python3
"""Paired RidgeFill evaluation against one server/build. No torch or model weights are loaded here.

Traps: needle files append the answer IDs (remove them); /completion does not score prompt rows;
post-sampling probabilities are invalid NLL; changing the fill end while teacher forcing changes the arm;
prompt cache invalidation between arms is mandatory. Missing probabilities or inactive fill are errors.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


class EvalError(ValueError):
    pass


class Client:
    def __init__(self, url: str, timeout: float = 1800):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def request(self, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body, allow_nan=False).encode("utf-8")
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            raise EvalError(f"{path}: HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}") from error
        if "error" in result:
            raise EvalError(f"{path}: {result['error']}")
        return result

    def ttft(self, body: dict) -> tuple[dict, float, float]:
        req = urllib.request.Request(self.url + "/completion", data=json.dumps(dict(body, stream=True)).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        first, final = None, None
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                raw = line[5:].strip()
                if raw == b"[DONE]":
                    break
                item = json.loads(raw)
                if "error" in item:
                    raise EvalError(f"stream: {item['error']}")
                if first is None and (item.get("tokens") or item.get("content")):
                    first = time.perf_counter() - start
                if item.get("stop"):
                    final = item
                    if first is None and item.get("timings", {}).get("predicted_n", 0) > 0:
                        first = time.perf_counter() - start
        if final is None or first is None:
            raise EvalError("TTFT stream did not produce a token and a final timing record")
        return final, first, time.perf_counter() - start


def read_items(path: Path, limit: int | None = None) -> list[dict]:
    items = []
    names = set()
    with path.open() as file:
        for line in file:
            row = json.loads(line)
            name, ids = row["name"], row["ids"]
            if name in names or not ids or any(type(token) is not int or token < 0 for token in ids):
                raise EvalError(f"{path}: duplicate name or invalid tokens for {name}")
            names.add(name)
            items.append(row)
            if limit is not None and len(items) == limit:
                break
    if not items:
        raise EvalError(f"{path}: no evaluation items")
    return items


def completion_body(ids: list[int], mode: str, tail: int, *, cache: bool = False, predict: int = 1) -> dict:
    return dict(prompt=ids, n_predict=predict, temperature=0, top_k=1, seed=0, cache_prompt=cache,
                return_tokens=True, ridgefill=mode != "off", ridgefill_mode=mode, ridgefill_tail=tail,
                post_sampling_probs=False)


def check_fill(response: dict, *, expected: bool, mode: str) -> int:
    filled = response.get("timings", {}).get("prompt_filled_n", 0)
    if type(filled) is not int or filled < 0:
        raise EvalError("invalid prompt_filled_n")
    if mode == "off" and filled != 0:
        raise EvalError("mode off reported filled rows")
    if expected and filled == 0:
        raise EvalError(f"mode {mode} did not fill an eligible prompt; check projector, mode and min-prompt")
    return filled


def validate_ridgefill_report(report: dict | None, n_prompt: int, mode: str) -> dict:
    if not isinstance(report, dict):
        raise EvalError("missing RidgeFill reporting; use a matching server/scorer build")
    counts = ("n_prompt", "n_prefix", "n_filled", "n_exact_prefix", "n_cached_exact_prefix", "n_exact_tail")
    if any(type(report.get(key)) is not int or report[key] < 0 for key in counts):
        raise EvalError("RidgeFill report has missing or invalid row counts")
    if (report["n_prompt"] != n_prompt or report["n_filled"] + report["n_exact_prefix"] != report["n_prefix"] or
            report["n_prefix"] + report["n_exact_tail"] != n_prompt):
        raise EvalError("RidgeFill report row counts do not cover the prompt")
    sources = report.get("n_exact_prefix_by_source", {})
    if (not isinstance(sources, dict) or set(sources) not in ({"class", "digit", "special", "header", "qattn"}, {"class", "digit", "special", "header", "qattn", "s1"}) or
            any(type(value) is not int or value < 0 for value in sources.values()) or
            sum(sources.values()) + report["n_cached_exact_prefix"] != report["n_exact_prefix"]):
        raise EvalError("RidgeFill report exact-row breakdown does not match its total")
    if ("s1" in sources) != ("s1" in report):
        raise EvalError("S1 reporting needs both source counts and completed probe costs")
    if "s1" in report:
        s1 = report["s1"]
        window = s1.get("window") if isinstance(s1, dict) else None
        if (not isinstance(window, list) or len(window) != 2 or any(type(row) is not int for row in window) or
                not report["n_prefix"] <= window[0] < window[1] <= n_prompt or s1.get("fill_recomputed") is not True or
                any(sources[key] != 0 for key in ("class", "digit", "header", "qattn")) or
                any(type(s1.get(key)) not in (int, float) or not math.isfinite(s1[key]) or s1[key] < 0 for key in ("macs_fill", "macs_probe"))):
            raise EvalError("invalid S1 window/source/cost reporting")
    if "vtail" in report:
        vtail = report["vtail"]
        if (not isinstance(vtail, dict) or type(vtail.get("rows")) is not int or not 0 < vtail["rows"] <= report["n_exact_tail"] or
                type(vtail.get("macs_measure")) not in (int, float) or not math.isfinite(vtail["macs_measure"]) or vtail["macs_measure"] < 0):
            raise EvalError("invalid vtail measurement/cost reporting")
    for name in ("tokens_skipped", "mac_share_skipped"):
        if type(report.get(name)) not in (int, float) or not math.isfinite(report[name]):
            raise EvalError("RidgeFill report has missing or nonfinite fractions/MACs")
    expected = report["n_filled"] / n_prompt if n_prompt else 0.0
    if abs(report["tokens_skipped"] - expected) > 1e-12:
        raise EvalError("RidgeFill report token fraction does not match counts")
    if (mode == "off" and report["n_prefix"] == 0 and report.get("macs_exact") is None and
            report.get("macs_ridgefill") is None and report["mac_share_skipped"] == 0):
        return report
    for name in ("macs_exact", "macs_ridgefill"):
        if type(report.get(name)) not in (int, float) or not math.isfinite(report[name]):
            raise EvalError("RidgeFill report has missing or nonfinite MAC counts")
    macs = report["macs_exact"]
    if macs < 0 or report["macs_ridgefill"] < 0 or (n_prompt > 0 and macs == 0):
        raise EvalError("RidgeFill report has invalid analytic MAC counts")
    mac_share = 1 - report["macs_ridgefill"] / macs if macs else 0.0
    if abs(report["tokens_skipped"] - expected) > 1e-12 or abs(report["mac_share_skipped"] - mac_share) > 1e-12:
        raise EvalError("RidgeFill report fractions do not match counts/MACs")
    if mode == "off" and (report["n_prefix"] or abs(report["mac_share_skipped"]) > 1e-12):
        raise EvalError("RidgeFill off control reports skipped work")
    return report


def response_ridgefill_report(response: dict, n_prompt: int, mode: str) -> dict:
    report = response.get("timings", {}).get("ridgefill")
    if report is None and mode == "off":
        report = dict(n_prompt=n_prompt, n_prefix=0, n_filled=0, n_exact_prefix=0, n_cached_exact_prefix=0, n_exact_tail=n_prompt,
                      n_exact_prefix_by_source=dict.fromkeys(("class", "digit", "special", "header", "qattn"), 0),
                      tokens_skipped=0.0, mac_share_skipped=0.0, macs_exact=None, macs_ridgefill=None,
                      scope="client stock exact control; server has no RidgeFill report; MAC totals unavailable")
    return validate_ridgefill_report(report, n_prompt, mode)


def target_logprob(response: dict, target: int) -> float:
    rows = response.get("completion_probabilities", [])
    if len(rows) != 1 or rows[0].get("id") != target:
        raise EvalError(f"teacher forcing did not return target token {target}")
    value = rows[0].get("logprob")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value > 0:
        raise EvalError("missing/invalid pre-sampling logprob (post-sampling probabilities cannot score NLL)")
    return float(value)


def score_nll(client: Client, ids: list[int], mode: str, tail: int, min_prompt: int, tail_min: int = 512) -> dict:
    scored = 512
    n = len(ids)
    prefix = max(0, n - tail)
    if n <= scored:
        raise EvalError("NLL requires at least 513 tokens")
    eligible = mode != "off" and prefix > 0 and n >= min_prompt
    if eligible and (tail - scored < tail_min or n - scored < min_prompt):
        raise EvalError("NLL teacher forcing requires tail-512 >= tail_min and N-512 >= min-prompt")
    values, filled_total, reports = [], 0, []
    start = time.perf_counter()
    for position in range(n - scored, n):
        # Keep P fixed while the known exact tail grows one teacher-forced token at a time.
        request_tail = max(1, position - prefix) if eligible else tail
        body = completion_body(ids[:position], mode, request_tail, cache=bool(values))
        # Sampling copies logits to its own candidates. The server reports the chosen token's RAW
        # probability over the full vocabulary when post_sampling_probs=false, even outside top n_probs.
        body.update(n_probs=1, logit_bias=[[ids[position], 1000000.0]])
        response = client.request("/completion", body)
        filled_total += check_fill(response, expected=eligible and not values, mode=mode)
        values.append(target_logprob(response, ids[position]))
        reports.append(response_ridgefill_report(response, position, mode))
    if len(values) != scored:
        raise EvalError("NLL did not score exactly 512 targets")
    return dict(nll=-float(np.mean(values)), scored=scored, logprobs=values, prefix=prefix if eligible else 0,
                filled=filled_total, ridgefill_requests=reports, wall_s=time.perf_counter() - start)


def needle_prompt(client: Client, item: dict) -> str:
    answer_len = item["answer_len"]
    if type(answer_len) is not int or not 0 < answer_len < len(item["ids"]):
        raise EvalError("needle answer_len is invalid; refusing to expose appended answer IDs")
    raw = client.request("/detokenize", dict(tokens=item["ids"][:-answer_len]))["content"]
    prompt = client.request("/apply-template", dict(messages=[dict(role="user", content=raw)],
                                                    chat_template_kwargs=dict(enable_thinking=False)))["prompt"]
    if not isinstance(prompt, str) or not re.search(r"<\|im_start\|>assistant\n<think>\s*</think>\s*$", prompt):
        raise EvalError("Qwen thinking-off chat template must end with the assistant's closed empty reasoning block")
    return prompt


def needle_correct(answer: str, value: str) -> bool:
    if not isinstance(value, str) or re.fullmatch(r"[0-9]{7}", value) is None:
        raise EvalError("needle value must contain exactly seven ASCII digits")
    return re.search(r"(?<!\d)" + re.escape(value) + r"(?!\d)", answer) is not None


def load_stats(kva: Path):
    path = kva / "finals/stats.py"
    spec = importlib.util.spec_from_file_location("ridgefill_eval_stats", path)
    if spec is None or spec.loader is None:
        raise EvalError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def reporting_summary(records: list[dict], arm: str) -> dict | None:
    reports = [report for row in records for report in
               (row[arm].get("ridgefill_requests") or [row[arm].get("ridgefill")])]
    if all(report is None for report in reports):
        return None
    if any(not isinstance(report, dict) for report in reports):
        raise EvalError("paired measurements contain missing RidgeFill reports")
    totals = {key: sum(report[key] for report in reports)
              for key in ("n_prompt", "n_prefix", "n_filled", "n_exact_prefix", "n_cached_exact_prefix", "n_exact_tail")}
    source_keys = ("class", "digit", "special", "header", "qattn")
    if any("s1" in report["n_exact_prefix_by_source"] for report in reports):
        source_keys += ("s1",)
    totals["n_exact_prefix_by_source"] = {key: sum(report["n_exact_prefix_by_source"].get(key, 0) for report in reports)
                                          for key in source_keys}
    totals["tokens_skipped"] = totals["n_filled"] / totals["n_prompt"] if totals["n_prompt"] else 0.0
    available = all(report.get("macs_exact") is not None and report.get("macs_ridgefill") is not None for report in reports)
    totals["macs_exact"] = sum(report["macs_exact"] for report in reports) if available else None
    totals["macs_ridgefill"] = sum(report["macs_ridgefill"] for report in reports) if available else None
    if available:
        totals["mac_share_skipped"] = 1 - totals["macs_ridgefill"] / totals["macs_exact"] if totals["macs_exact"] else 0.0
    else:
        if any(report["mac_share_skipped"] != 0 for report in reports):
            raise EvalError("cannot aggregate unknown MAC totals for approximate requests")
        totals["mac_share_skipped"] = 0.0
    totals.update(units=len(records), requests=len(reports), scope="aggregate prompt coverage including cached rows; analytic cold prefill")
    return totals


def paired_summary(records: list[dict], kind: str, kva: Path) -> dict:
    if not records:
        raise EvalError("SKIPPED: no paired measurements")
    stats = load_stats(kva)
    metric = {"nll": "nll", "needle": "accuracy", "ttft": "ttft_s", "bench": "tokens_per_second"}[kind]
    reference = {row["name"]: row["reference"][metric] for row in records}
    candidate = {row["name"]: row["candidate"][metric] for row in records}
    if len(reference) != len(records):
        raise EvalError("duplicate pairing units")
    _, exact, approximate = stats.paired(reference, candidate, what=kind)
    interval = stats.bca(approximate - exact)
    result = dict(kind=kind, units=len(exact), reference_mean=float(exact.mean()), candidate_mean=float(approximate.mean()),
                delta=interval.estimate, ci95=[interval.lo, interval.hi], degenerate=interval.degenerate,
                resamples=interval.n_resamples, seed=interval.seed, estimand="document mean" if kind == "nll" else "paired unit mean")
    reporting = {arm: reporting_summary(records, arm) for arm in ("reference", "candidate")}
    if any(value is not None for value in reporting.values()):
        if any(value is None for value in reporting.values()):
            raise EvalError("paired measurements lack one arm's RidgeFill reporting")
        result["ridgefill"] = reporting
    return result


def write_record(file, record: dict) -> None:
    encoded = json.dumps(record, allow_nan=False)
    file.write(encoded + "\n")
    file.flush()
    print(encoded, flush=True)


def evaluate(args, client: Client, file) -> list[dict]:
    items = read_items(args.data, args.limit)
    records = []
    for index, item in enumerate(items):
        results = {}
        arms = [("reference", "off"), ("candidate", args.mode)]
        for arm, mode in arms[::1 if index % 2 == 0 else -1]:
            if args.command == "nll":
                ids = item["ids"][:args.length]
                if len(ids) != args.length:
                    raise EvalError(f"{item['name']}: fewer than --length {args.length} tokens")
                results[arm] = score_nll(client, ids, mode, args.tail, args.min_prompt, args.tail_min)
            else:
                needle_correct("", item["value"])
                prompt = needle_prompt(client, item)
                ids = client.request("/tokenize", dict(content=prompt, add_special=False, parse_special=True))["tokens"]
                body = completion_body(ids, mode, args.tail, predict=32)
                start = time.perf_counter()
                response = client.request("/completion", body)
                filled = check_fill(response, expected=mode != "off" and len(ids) >= args.min_prompt and len(ids) > args.tail, mode=mode)
                results[arm] = dict(accuracy=100.0 * needle_correct(response["content"], str(item["value"])), answer=response["content"],
                                    prompt_tokens=len(ids), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                                    thinking=False, greedy=True, removed_answer_tokens=item["answer_len"],
                                    filled=filled, ridgefill=response_ridgefill_report(response, len(ids), mode),
                                    timings=response["timings"], wall_s=time.perf_counter() - start)
        record = dict(kind=args.command, name=item["name"], domain=item.get("domain"), candidate_mode=args.mode, **results)
        records.append(record)
        write_record(file, record)
    return records


def measure_ttft(args, client: Client, file) -> list[dict]:
    pool = [token for item in read_items(args.data) for token in item["ids"]]
    records = []
    for length in args.lengths:
        ids = (pool * ((length + len(pool) - 1) // len(pool)))[:length]
        for mode in ("off", args.mode):
            client.request("/completion", completion_body(ids, mode, args.tail))
        for repeat in range(args.repeats):
            results = {}
            arms = [("reference", "off"), ("candidate", args.mode)]
            for arm, mode in arms[::1 if repeat % 2 == 0 else -1]:
                response, first, wall = client.ttft(completion_body(ids, mode, args.tail, predict=args.predict))
                filled = check_fill(response, expected=mode != "off" and length >= args.min_prompt and length > args.tail, mode=mode)
                results[arm] = dict(ttft_s=first, wall_s=wall, timings=response["timings"], filled=filled,
                                    ridgefill=response_ridgefill_report(response, length, mode))
            record = dict(kind="ttft", name=f"n{length}-r{repeat}", length=length, candidate_mode=args.mode, **results)
            records.append(record)
            write_record(file, record)
    return records


def measure_bench(args, file) -> list[dict]:
    records = []
    for length in args.lengths:
        for repeat in range(args.repeats):
            results = {}
            arms = [("reference", "off"), ("candidate", args.mode)]
            for arm, mode in arms[::1 if repeat % 2 == 0 else -1]:
                command = [str(args.binary.resolve()), *args.bench_arg, "-m", str(args.model), "-p", str(length), "-n", str(args.predict),
                           "-r", "1", "-o", "json", "--ridgefill", str(args.projector), "--ridgefill-mode", mode,
                           "--ridgefill-tail", str(args.tail)]
                completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=args.timeout)
                rows = json.loads(completed.stdout)
                if len(rows) != 1 or rows[0]["n_prompt"] != length or rows[0]["n_gen"] != args.predict:
                    raise EvalError("llama-bench returned an unexpected test set")
                if not math.isfinite(rows[0]["avg_ts"]) or rows[0]["avg_ts"] <= 0:
                    raise EvalError("llama-bench returned no positive finite throughput")
                reports = [json.loads(line.removeprefix("RidgeFill: report ")) for line in completed.stderr.splitlines()
                           if line.startswith("RidgeFill: report ")]
                if len(reports) != 1:
                    raise EvalError("llama-bench did not report one measured prefill's skipped work")
                report = validate_ridgefill_report(reports[0], length, mode)
                results[arm] = dict(tokens_per_second=rows[0]["avg_ts"], native=rows[0], command=command, ridgefill=report)
            record = dict(kind="bench", name=f"n{length}-g{args.predict}-r{repeat}", length=length, candidate_mode=args.mode, **results)
            records.append(record)
            write_record(file, record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("nll", "needle", "ttft", "bench"))
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kva", type=Path, required=True, help="oracle repo containing finals/stats.py")
    parser.add_argument("--build-label", required=True)
    parser.add_argument("--mode", choices=("off", "speed", "quality", "quality+qattn"), default="speed")
    parser.add_argument("--tail", type=int, default=4096)
    parser.add_argument("--tail-min", type=int, default=512)
    parser.add_argument("--min-prompt", type=int, default=8192)
    parser.add_argument("--length", type=int, default=16384)
    parser.add_argument("--lengths", type=lambda value: [int(x) for x in value.split(',')], default=[8192, 16384, 32768])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--predict", type=int, help="default: 1 for TTFT, 0 for bench; use bench --lengths 0 --predict 128 for decode")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--projector", type=Path)
    parser.add_argument("--bench-arg", action="append", default=[], help="one native backend argument; use --bench-arg=-ngl --bench-arg=99")
    args = parser.parse_args()
    if args.tail < 1 or args.tail_min < 1 or args.min_prompt < 0 or (args.limit is not None and args.limit < 1):
        parser.error("tail/tail-min/limit must be positive and min-prompt nonnegative")
    if any(length < 0 for length in args.lengths) or not args.lengths or args.timeout <= 0 or not math.isfinite(args.timeout):
        parser.error("lengths must be nonnegative and timeout must be finite and positive")
    if args.command in ("ttft", "bench") and args.repeats < 3:
        parser.error("paired speed measurements require at least 3 repeats")
    if args.predict is None:
        args.predict = 0 if args.command == "bench" else 1
    if args.predict < 0 or (args.command == "ttft" and args.predict == 0):
        parser.error("predict must be nonnegative, and positive for TTFT")
    if args.command != "bench" and args.data is None:
        parser.error("--data is required")
    if args.command == "bench" and any(value is None for value in (args.binary, args.model, args.projector)):
        parser.error("bench requires --binary, --model and --projector")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = Client(args.url, args.timeout)
    metadata = dict(kind="metadata", build=args.build_label, machine=platform.node(), args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    if args.command != "bench":
        metadata["server"] = client.request("/props")
    if args.data:
        metadata["data_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    with args.output.open("x") as file:
        write_record(file, metadata)
        if args.command == "bench":
            records = measure_bench(args, file)
        elif args.command == "ttft":
            records = measure_ttft(args, client, file)
        else:
            records = evaluate(args, client, file)
        # Different prompt lengths are different speed estimands; never pool them into one CI.
        groups = {length: [row for row in records if row["length"] == length] for length in args.lengths} if args.command in ("ttft", "bench") else {None: records}
        for length, rows in groups.items():
            write_record(file, dict(kind="summary", length=length, result=paired_summary(rows, args.command, args.kva)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (EvalError, OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(json.dumps(dict(result="FAILED", reason=str(error))), file=sys.stderr)
        sys.exit(1)
