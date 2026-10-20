#!/usr/bin/env python3
"""CPU G1: oracle agreement and controls over three chunk sizes. Nothing skipped is a pass.

Use split 3 for speed/quality and split 4 for the future qattn sweep gate. Quality alpha 1 must
refuse the unavailable rho-scaled correction; it is not an inference-agreement case. Empty prefixes
must not receive exact rows. An engine build failure or unavailable sweep produces nonzero SKIPPED.
Logs and reference arrays live under --work; cached engine build logs identify the exact revision.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import build_gate1
import compare_qwen35

ROOT = Path(__file__).resolve().parents[2]


def run(args) -> int:
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if args.mode == "quality+qattn":
        raise ValueError("quality+qattn: sweep dump/mass API not available in this harness yet; use split 4 when integrated")
    if not (work / "fixture.json").exists():
        subprocess.run([sys.executable, str(ROOT / "tests/ridgefill/make_tiny_qwen35.py"), "--out", str(work)], check=True)
    dump = args.dump.resolve() if args.dump else build_gate1.build(args.engine_ref, args.build_work)
    out = work / f"g1-s{args.split}-{args.mode}"
    out.mkdir(exist_ok=True)
    rows = [0, 7, 31, 127, 255, 499]
    other_rows = [1, 6, 30, 126, 254, 498]
    n_prompt = (work / "tokens.bin").stat().st_size // 4
    expected = {}
    records = []

    def record(name, passed, **details):
        entry = dict(check=name, result="PASS" if passed else "FAIL", **details)
        records.append(entry)
        print(json.dumps(entry, sort_keys=True), flush=True)
        if not passed:
            raise AssertionError(f"G1 check failed: {name}")

    def invoke(tag, batch, ubatch, mode="off", alpha=0, projector=None, tail=200, filled_logit=False, exact_rows=None, expect_refusal=False):
        prefix = out / tag
        command = [str(dump), "--model", str(work / "tiny-qwen35-f32.gguf"), "--tokens", str(work / "tokens.bin"),
                   "--output", str(prefix), "--mode", mode, "--tail", str(tail), "--keep", "200",
                   "--batch", str(batch), "--ubatch", str(ubatch), "--alpha", str(alpha)]
        if projector:
            command += ["--projector", str(projector)]
        if mode == "quality" and tail < n_prompt:
            command += ["--rows", ",".join(str(row) for row in (rows if exact_rows is None else exact_rows))]
        if filled_logit:
            command.append("--filled-logit")
        with prefix.with_suffix(".log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode and not (filled_logit or expect_refusal):
            raise RuntimeError(f"rf_dump failed ({result.returncode}); see {prefix.with_suffix('.log')}")
        return prefix, result.returncode

    def compare(prefix, mode, alpha):
        key = mode, alpha
        if key not in expected:
            expected[key] = compare_qwen35.reference(args.kva, work, args.split, mode, alpha, rows)
            import numpy as np
            np.savez(out / f"oracle-{mode}-a{alpha}.npz", **expected[key])
        return compare_qwen35.compare(work, prefix, expected[key], args.split, mode == "exact")

    projector = work / f"projector-s{args.split}-f32.gguf"
    wrong_map = work / f"projector-s{args.split}-wrong-f32.gguf"
    wrong_st = work / f"projector-s{args.split}-wrong-st-f32.gguf"
    for batch, ubatch in ((700, 512), (128, 128), (50, 50)):
        tag = f"b{batch}-ub{ubatch}"
        stock, _ = invoke("stock-" + tag, batch, ubatch)
        comparison = compare(stock, "exact", 0)
        record("oracle-stock-" + tag, comparison["result"] == "PASS", comparison=comparison)
        off, _ = invoke("off-" + tag, batch, ubatch, projector=projector)
        record("negative-off-" + tag, all(Path(str(stock) + suffix).read_bytes() == Path(str(off) + suffix).read_bytes()
                                         for suffix in (".kv", ".prefix.kv", ".logits")))
        stock_short, _ = invoke("stock-no-prefix-" + tag, batch, ubatch, tail=700)
        no_prefix, _ = invoke("no-prefix-" + tag, batch, ubatch, mode=args.mode, projector=projector, tail=700)
        record("negative-tail-ge-n-" + tag, all(Path(str(stock_short) + suffix).read_bytes() == Path(str(no_prefix) + suffix).read_bytes()
                                               for suffix in (".kv", ".logits")))
        for alpha in ((0, 1) if args.mode == "speed" else (0,)):
            suffix = tag + f"-a{alpha}"
            filled, _ = invoke("fill-" + suffix, batch, ubatch, mode=args.mode, alpha=alpha, projector=projector)
            comparison = compare(filled, args.mode, alpha)
            record("oracle-fill-" + suffix, comparison["result"] == "PASS", comparison=comparison)
            wrong, _ = invoke("wrong-map-" + suffix, batch, ubatch, mode=args.mode, alpha=alpha, projector=wrong_map)
            comparison = compare(wrong, args.mode, alpha)
            record("positive-wrong-map-" + suffix, comparison["result"] == "FAIL" and comparison["max_relative_error"] > 1e-2,
                   comparison=comparison)
            if alpha == 1:
                wrong, _ = invoke("wrong-st-" + suffix, batch, ubatch, mode=args.mode, alpha=alpha, projector=wrong_st)
                comparison = compare(wrong, args.mode, alpha)
                prefix_errors = [error for name, error in comparison["relative_errors"].items() if name.startswith("prefix.") and name.endswith(".rec")]
                record("positive-wrong-st-" + suffix, comparison["result"] == "FAIL" and bool(prefix_errors) and max(prefix_errors) > 1e-2,
                       comparison=comparison)
        if args.mode == "quality":
            wrong, _ = invoke("wrong-rows-" + tag, batch, ubatch, mode="quality", projector=projector, exact_rows=other_rows)
            comparison = compare(wrong, "quality", 0)
            record("positive-wrong-rows-" + tag, comparison["result"] == "FAIL" and comparison["max_relative_error"] > 1e-2,
                   comparison=comparison)
            speed, _ = invoke("speed-instead-of-quality-" + tag, batch, ubatch, mode="speed", projector=projector)
            comparison = compare(speed, "quality", 0)
            record("positive-speed-instead-of-quality-" + tag, comparison["result"] == "FAIL" and comparison["max_relative_error"] > 1e-2,
                   comparison=comparison)
            refused, code = invoke("quality-alpha1-" + tag, batch, ubatch, mode="quality", alpha=1, projector=projector, expect_refusal=True)
            message = refused.with_suffix(".log").read_text().lower()
            record("quality-alpha1-refused-" + tag, code == 2 and "rho-scaled correction" in message, returncode=code)
        refused, code = invoke("filled-logit-" + tag, batch, ubatch, mode=args.mode, projector=projector, filled_logit=True)
        message = refused.with_suffix(".log").read_text().lower()
        record("filled-logit-refused-" + tag, code == 2 and ("filled" in message or "ridgefill" in message))
    result = dict(gate="G1", result="PASS" if records else "SKIPPED", checks=len(records), split=args.split, mode=args.mode)
    (out / "results.json").write_text(json.dumps(dict(summary=result, checks=records), indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if records else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kva", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=ROOT / "out/gate1-qwen35")
    parser.add_argument("--build-work", type=Path, default=ROOT / "out/gate1-engine")
    parser.add_argument("--engine-ref", default="rf27-engine")
    parser.add_argument("--dump", type=Path, help="existing rf_dump binary (otherwise build the pinned engine ref)")
    parser.add_argument("--split", type=int, choices=(3, 4), default=3)
    parser.add_argument("--mode", choices=("speed", "quality", "quality+qattn"), default="speed")
    args = parser.parse_args()
    try:
        sys.exit(run(args))
    except (subprocess.CalledProcessError, ValueError, RuntimeError, AssertionError, OSError) as error:
        print(json.dumps(dict(gate="G1", result="FAIL" if isinstance(error, AssertionError) else "SKIPPED",
                              reason=str(error), split=args.split, mode=args.mode)), flush=True)
        sys.exit(1)
