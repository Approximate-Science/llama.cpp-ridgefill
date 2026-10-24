#!/usr/bin/env python3
"""Compare streamed scorer output alone and after other documents in one context."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess


def float_bytes(values):
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("missing or non-finite values")
    return struct.pack("<" + "d" * len(values), *values)


def compare(reference, actual):
    checks = {}
    for field in ("logprobs", "logsumexp", "s1_mass"):
        left, right = reference[field], actual[field]
        checks[field] = len(left) == len(right) and float_bytes(left) == float_bytes(right)
    for field in ("exact_rows", "target_positions", "n", "scored", "prefills"):
        checks[field] = reference[field] == actual[field]
    return checks


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    documents = read(args.input)
    by_name = {doc["name"]: doc for doc in documents}
    if len(by_name) != len(documents) or args.target not in by_name:
        raise ValueError("document names must be unique and include the target")
    others = [doc["name"] for doc in documents if doc["name"] != args.target]
    if not others:
        raise ValueError("provide at least one predecessor document")
    orders = [[args.target]] + [[name, args.target] for name in others]
    command = json.loads(args.command_json.read_text())
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ValueError("command-json must contain a native scorer argv array")
    for flag in ("--input", "--output", "--ridgefill-stream-projector"):
        if flag not in command or command.index(flag) + 1 >= len(command):
            raise ValueError("missing scorer flag: " + flag)
    binary = args.binary.resolve()
    command[0] = str(binary)
    records = []

    def call(label, names, streamed, trim):
        source, target = args.out / (label + "-input.jsonl"), args.out / (label + ".jsonl")
        source.write_text("".join(json.dumps(by_name[name]) + "\n" for name in names))
        cmd = list(command)
        for flag, value in (("--input", str(source)), ("--output", str(target)),
                            ("--ridgefill-stream-projector", "on" if streamed else "off")):
            cmd[cmd.index(flag) + 1] = value
        env = dict(os.environ, LD_LIBRARY_PATH=str(binary.parent))
        env["LLAMA_RIDGEFILL_TRIM_KV"] = "0"
        env.pop("LLAMA_RIDGEFILL_PROFILE", None)
        if trim:
            env["LLAMA_RIDGEFILL_TRIM_KV"] = "1"
        (args.out / (label + "-command.json")).write_text(json.dumps(cmd, indent=2) + "\n")
        with (args.out / (label + ".log")).open("w") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
        rows = read(target)
        if [row["name"] for row in rows] != names:
            raise ValueError("scorer output changed document order/count")
        return rows

    reference = read(args.reference) if args.reference else call("resident", list(by_name), False, False)
    reference = {row["name"]: row for row in reference}
    if not set(by_name).issubset(reference):
        raise ValueError("resident reference does not cover the input documents")
    for trim in (False, True):
        for index, order in enumerate(orders):
            label = f"order{index}-trim{int(trim)}"
            for row in call(label, order, True, trim):
                checks = compare(reference[row["name"]], row)
                records.append(dict(order=order, trim=trim, document=row["name"], checks=checks,
                                    byte_identical=all(checks.values())))
                print(json.dumps(records[-1]), flush=True)
            (args.out / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    proof = dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                 input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                 command_sha256=hashlib.sha256(args.command_json.read_bytes()).hexdigest())
    (args.out / "provenance.json").write_text(json.dumps(proof, indent=2) + "\n")
    if not records or not all(record["byte_identical"] for record in records):
        raise AssertionError("streamed output differs from resident output")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="JSONL with unique name/ids documents")
    parser.add_argument("--command-json", type=Path, required=True, help="complete scorer argv including input/output/stream flags")
    parser.add_argument("--target", required=True)
    parser.add_argument("--reference", type=Path, help="optional cached resident scorer JSONL")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
