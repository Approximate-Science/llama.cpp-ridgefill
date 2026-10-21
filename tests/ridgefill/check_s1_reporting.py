#!/usr/bin/env python3
"""Compare S1/vtail reporting math with independent kva.compute dimensions; no S1 inference claim."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

from safetensors import safe_open

from eval_client import validate_ridgefill_report


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("ridgefill_compute", args.kva / "kva/compute.py")
    assert spec is not None and spec.loader is not None
    compute = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compute)
    dims = compute.Dims(args.fixture / "tiny-qwen35-hf")
    with safe_open(args.fixture / "tiny-qwen35-hf/model.safetensors", framework="numpy") as weights:
        shape = weights.get_slice("model.layers.3.self_attn.q_proj.weight").get_shape()
    assert shape[0] in (dims.hq * dims.d, 2 * dims.hq * dims.d)
    dims.gated = shape[0] == 2 * dims.hq * dims.d
    result = subprocess.run([str(args.binary), str(args.fixture)], capture_output=True, text=True, timeout=60)
    (args.out / "native.log").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    native = json.loads(result.stdout)
    (args.out / "native.json").write_text(json.dumps(native, indent=2) + "\n")
    assert native["result"] == "PASS" and len(native["records"]) == 16
    errors = []
    for row in native["records"]:
        split, width, count = row["split"], row["window"], row["vtail_rows"]
        late = range(split, dims.L)
        attention = [i for i in late if dims.types[i] == "attn"]
        # Probe keys are prefix + causal window rows; the uncomputed intervening tail is masked.
        fill = sum(500 * (dims.H**2 + dims.attn_fill()) for _ in attention) if width else 0
        probe = sum(dims.layer_exact(i, 0, width) for i in late)
        probe += len(attention) * 2 * dims.hq * dims.d * width * 500
        vtail = len(attention) * count * (dims.H**2 + dims.H * dims.hkv * dims.d)
        positions = sum(value + 1 for value in row["rows"]) / len(row["rows"]) / 501
        base = compute.bypass(dims, 700, split, 200, len(row["rows"]), pos=positions)
        base += compute.qattn_select(dims, 700, split, window=64)
        report = validate_ridgefill_report(row["report"], 700, "quality+qattn")
        assert report["n_exact_prefix"] == 6 and report["n_filled"] == 494
        assert report["tokens_skipped"] == 494 / 700
        if width:
            assert report["s1"]["macs_fill"] == fill and report["s1"]["macs_probe"] == probe
            assert report["s1"]["fill_recomputed"] is True
            assert report["n_exact_prefix_by_source"] == {
                "class": 0, "digit": 0, "special": 2, "header": 0, "qattn": 0, "s1": 4}
        if count:
            assert report["vtail"]["rows"] == count and report["vtail"]["macs_measure"] == vtail
        want = base + fill + probe + vtail
        error = abs(report["macs_ridgefill"] - want) / want
        assert error < 1e-12, (row, want)
        errors.append(error)
    summary = dict(result="PASS", scope="reporting math only; S1 engine not exercised", native_checks=native["checks"],
                   configs=len(errors), worst_relative_error=max(errors))
    (args.out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "fixture", "kva", "out"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    run(parser.parse_args())
