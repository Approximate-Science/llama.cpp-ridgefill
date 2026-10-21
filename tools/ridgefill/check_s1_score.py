#!/usr/bin/env python3
"""Tiny CPU S1/vtail scorer gate against lane T's frozen reference dumps. Exit 77 means SKIPPED, not PASS."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "baseline-binary", "fixture", "reference", "kva", "work"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.reference / "MANIFEST.json").read_text())
    for line in (args.reference / "SHA256SUMS").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        path = args.reference / filename.lstrip("*")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, filename
    for filename in ("tiny-qwen35-f32.gguf", "tiny-qwen35-hf/model.safetensors", "projector-s4/rowsel.safetensors"):
        assert hashlib.sha256((args.fixture / filename).read_bytes()).hexdigest() == manifest["inputs"][filename]
    ids = np.fromfile(args.fixture / "tokens.bin", dtype="<i4")
    crafted = ids.copy()
    for pos, token in ((0, 1), (10, 2), (11, 1), (40, 0), (41, 2), (300, 1), (497, 2), (499, 1), (600, 2)):
        crafted[pos] = token
    prompts = {"fixture": ids, "selchat": crafted}
    for name, values in prompts.items():
        assert hashlib.sha256(values.tobytes()).hexdigest() == manifest["inputs"]["tokens:" + name]
        (args.work / (name + ".json")).write_text(json.dumps(dict(name=name, ids=values.tolist())))
    serial = 0

    def run(name, extra=(), mode="quality+qattn", chunk=128, precision="f32", binary=args.binary, refused=None):
        nonlocal serial
        serial += 1
        output = args.work / f"result-{serial}.jsonl"
        command = [binary, "--model", args.fixture / "tiny-qwen35-f32.gguf", "--projector", args.fixture / "projector-s4-f32.gguf",
                   "--input", args.work / (name + ".json"), "--output", output, "--mode", mode,
                   "--scored", 64, "--tail", 200, "--min-prompt", 1, "--qattn-window-end", "document",
                   "--batch", chunk, "--ubatch", chunk, "--cache-type", precision, "--h-type", precision,
                   "--flash-attn", "off", "--diagnostics", *extra]
        result = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=60)
        (args.work / f"result-{serial}.log").write_text(result.stdout + result.stderr)
        if refused:
            assert result.returncode != 0 and refused in result.stderr, result.stderr
            return None
        assert result.returncode == 0, result.stderr
        return json.loads(output.read_text())

    for name in prompts:
        for precision in ("f32", "f16"):
            for mode in ("off", "speed", "quality", "quality+qattn"):
                before = run(name, mode=mode, precision=precision, binary=args.baseline_binary)
                after = run(name, mode=mode, precision=precision)
                for key in ("logprobs", "logsumexp", "exact_rows", "qattn_window", "question_mass", "ridgefill"):
                    assert before.get(key) == after.get(key), (name, mode, precision, key)
    print("PASS defaults: byte-identical numerical outputs, row selection and cost report; two prompts, four modes, F32/F16", flush=True)
    version = int(subprocess.check_output([args.binary, "--ridgefill-api-version"], text=True))
    if version < 2:
        for option in ("--s1", "--vtail"):
            run("fixture", [option, "on"], refused="requires libllama C API v2")
        result = dict(result="SKIPPED", reason="libllama C API v2 is not available", api_version=version,
                      defaults="PASS", named_api_refusals="PASS", s1_mass_rows_logits="SKIPPED", s1_cost="SKIPPED", vtail="SKIPPED")
        (args.work / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return 77

    sys.path.insert(0, str(args.kva))
    from kva.compute import Dims, bypass, qattn_select
    dm = Dims(args.fixture / "tiny-qwen35-hf")
    dm.gated = True
    p, n, w, split = manifest["P"], manifest["N"], manifest["window"], manifest["split"]
    late_attn = manifest["late_attention_layers"]
    fill_cost = sum(p * (dm.H ** 2 + dm.attn_fill()) for _ in late_attn)
    probe_cost = sum(dm.layer_exact(il, p, p + w) for il in range(split, dm.L))
    vtail_cost = len(late_attn) * 128 * (dm.H ** 2 + dm.H * dm.hkv * dm.d)
    checks, worst, mass_error = 0, 0.0, 0.0
    for name, values in prompts.items():
        for s1 in (False, True):
            for vtail in (False, True):
                arm = ("S1gw" if s1 else "T3s") + ("-vtail-k72" if vtail else "")
                ref = np.load(args.reference / f"{name}-{arm}.npz")
                logits = ref["logits"][-65:-1].astype(np.float64)
                lse = np.logaddexp.reduce(logits, axis=-1)
                lp = logits[np.arange(64), values[-64:]] - lse
                for chunk in (50, 128, 512):
                    extra = ["--special-exact", "on", "--s1", "on" if s1 else "off", "--s1-window", w,
                             "--vtail", "on" if vtail else "off", "--vtail-rows", 128]
                    got = run(name, extra, chunk=chunk)
                    error = max(np.max(np.abs(got["logprobs"] - lp)), np.max(np.abs(got["logsumexp"] - lse)))
                    assert error < 1e-4, (name, arm, chunk, error)
                    assert got["exact_rows"] == ref["rows"].tolist()
                    if s1:
                        delta = np.max(np.abs(got["s1_mass"] - ref["mass_s1"])) / np.max(np.abs(ref["mass_s1"]))
                        assert delta < 1e-4
                        assert got["s1"]["n_base"] == int(ref["n_rows_ref"])
                        assert got["s1"]["window"] == [n - w, n]
                        assert got["s1"]["fill_rows"] == p and got["s1"]["probe_rows"] == w
                        assert not got["s1"]["prefix_kv_reused"]
                        assert got["s1"]["fill_macs"] == fill_cost and got["s1"]["probe_macs"] == probe_cost
                        assert got["s1"]["macs"] == fill_cost + probe_cost > probe_cost
                        mass_error = max(mass_error, delta)
                    if vtail:
                        assert np.max(np.abs(got["vtail"]["scales"] - ref["vtail_scales"].reshape(-1))) < 1e-4
                        assert got["vtail"]["rows"] == 128 and got["vtail"]["macs"] == vtail_cost
                    rows = np.asarray(got["exact_rows"])
                    position = np.mean(rows + 1) / (p + 1)
                    base_cost = bypass(dm, n, split, 200, len(rows), pos=position) + qattn_select(dm, n, split, 64)
                    expected = base_cost + (fill_cost + probe_cost if s1 else 0) + (vtail_cost if vtail else 0)
                    assert abs(got["ridgefill"]["macs_ridgefill"] - expected) < 1e-6
                    checks += 1; worst = max(worst, error)
                print(f"PASS {name} {arm}: mass/rows/logits/scales/cost across 3 chunks", flush=True)
    run("fixture", ["--s1", "on", "--s1-window", 201], refused="window entirely in the known exact tail")
    run("fixture", ["--vtail", "on", "--vtail-rows", 200], refused="would read scored NLL targets")
    result = dict(result="PASS", checks=checks, worst_logprob_lse=float(worst), worst_mass_relative=float(mass_error),
                  cost="includes S1 prefix projector/KV recomputation and all W probe rows", vtail_k0="SKIPPED: scorer refuses target-reading calibration")
    (args.work / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
