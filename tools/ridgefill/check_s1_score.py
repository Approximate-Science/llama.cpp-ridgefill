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
    for name in ("binary", "baseline-binary", "native-dump", "fixture", "reference", "kva", "work"):
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
        values.tofile(args.work / (name + ".bin"))
    serial = 0

    def run(name, extra=(), mode="quality+qattn", chunk=128, precision="f32", binary=args.binary, refused=None, endpoint="document"):
        nonlocal serial
        serial += 1
        output = args.work / f"result-{serial}.jsonl"
        command = [binary, "--model", args.fixture / "tiny-qwen35-f32.gguf", "--projector", args.fixture / "projector-s4-f32.gguf",
                   "--input", args.work / (name + ".json"), "--output", output, "--mode", mode,
                   "--scored", 64, "--tail", 200, "--min-prompt", 1,
                   "--batch", chunk, "--ubatch", chunk, "--cache-type", precision, "--h-type", precision,
                   "--flash-attn", "off", "--diagnostics", *extra]
        if endpoint is not None:
            command += ["--qattn-window-end", endpoint]
        result = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=60)
        (args.work / f"result-{serial}.log").write_text(result.stdout + result.stderr)
        if refused:
            assert result.returncode != 0 and refused in result.stderr, result.stderr
            return None
        assert result.returncode == 0, result.stderr
        records = [json.loads(line) for line in output.read_text().splitlines()]
        return records[0] if len(records) == 1 else records

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
            run("fixture", [option, "on"], refused="RidgeFill: S1/vtail requires C API v2; unavailable in this build")
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
    cache_rows = (n + 255) // 256 * 256
    probe_cost = sum(w * (dm.attn_linear() + dm.mlp()) + 2 * dm.hq * dm.d * w * cache_rows if il in late_attn
                     else dm.layer_exact(il, 0, w) for il in range(split, dm.L))
    vtail_cost = len(late_attn) * 128 * (dm.H ** 2 + 2 * dm.H * dm.hkv * dm.d)
    checks, worst, mass_error = 0, 0.0, 0.0
    outputs = {}
    for name, values in prompts.items():
        for s1 in (False, True):
            for vtail in (False, True):
                arm = ("S1gw" if s1 else "T3s") + ("-vtail-k72" if vtail else "")
                ref = np.load(args.reference / f"{name}-{arm}.npz")
                logits = ref["logits"][-65:-1].astype(np.float64)
                lse = np.max(logits, axis=-1) + np.log(np.exp(logits - np.max(logits, axis=-1, keepdims=True)).sum(axis=-1))
                lp = logits[np.arange(64), values[-64:]] - lse
                for chunk in (50, 128, 512):
                    extra = ["--special-exact", "on", "--s1", "on" if s1 else "off", "--s1-window", w if s1 else 0,
                             "--vtail", "on" if vtail else "off", "--vtail-rows", 128]
                    got = run(name, extra, chunk=chunk)
                    error = max(np.max(np.abs(got["logprobs"] - lp)), np.max(np.abs(got["logsumexp"] - lse)))
                    assert error < 1e-4, (name, arm, chunk, error)
                    assert got["exact_rows"] == ref["rows"].tolist()
                    assert np.max(np.abs(np.asarray(got["question_mass"])[:p] - ref["mass_qattn"])) < 1e-4
                    if s1:
                        delta = np.max(np.abs(got["s1_mass"] - ref["mass_s1"])) / np.max(np.abs(ref["mass_s1"]))
                        assert delta < 1e-4
                        assert got["s1"]["n_base"] == int(ref["n_rows_ref"])
                        assert got["s1"]["window"] == [n - w, n]
                        assert got["s1"]["fill_rows"] == p and got["s1"]["probe_rows"] == w
                        assert got["s1"]["probe_kv_pairs"] == w * cache_rows
                        assert not got["s1"]["prefix_kv_reused"]
                        assert got["s1"]["fill_macs"] == fill_cost and got["s1"]["probe_macs"] == probe_cost
                        assert got["s1"]["macs"] == fill_cost + probe_cost > probe_cost
                        mass_error = max(mass_error, delta)
                    if vtail:
                        assert np.max(np.abs(got["vtail"]["scales"] - ref["vtail_scales"].reshape(-1))) < 1e-4
                        assert got["vtail"]["rows"] == 128 and got["vtail"]["macs"] == vtail_cost
                        assert got["vtail"]["cache_rows"] == cache_rows
                        elements = len(late_attn) * cache_rows * dm.hkv * dm.d
                        assert got["vtail"]["cache_elements_scaled"] == elements
                        assert got["vtail"]["cache_read_bytes"] == got["vtail"]["cache_write_bytes"] == 4 * elements
                    rows = np.asarray(got["exact_rows"])
                    position = np.mean(rows + 1) / (p + 1)
                    base_cost = bypass(dm, n, split, 200, len(rows), pos=position) + qattn_select(dm, n, split, 64)
                    expected = base_cost + (fill_cost + probe_cost if s1 else 0) + (vtail_cost if vtail else 0)
                    assert abs(got["ridgefill"]["macs_ridgefill"] - expected) < 1e-6
                    checks += 1; worst = max(worst, error)
                    outputs[name, arm] = got
                print(f"PASS {name} {arm}: mass/rows/logits/scales/cost across 3 chunks", flush=True)
        s1 = outputs[name, "S1gw"]
        t3 = outputs[name, "T3s"]
        assert s1["s1"]["n_base"] == len(t3["exact_rows"])
        assert s1["exact_rows"] != t3["exact_rows"]
        vt = outputs[name, "T3s-vtail-k72"]
        assert np.max(np.abs(np.asarray(vt["logprobs"]) - t3["logprobs"])) > 1e-4
        print(f"PASS {name}: computed base count {s1['s1']['n_base']}, final S1 rows {len(s1['exact_rows'])}; S1 and vtail positive controls", flush=True)

    native_checks = 0
    for name, values in prompts.items():
        for arm in ("T3s", "T3s-vtail-k0", "T3s-vtail-k72", "S1gw", "S1gw-vtail-k0", "S1gw-vtail-k72"):
            ref = np.load(args.reference / f"{name}-{arm}.npz")
            out = args.work / f"native-{name}-{arm}"
            extra = (["--s1-window", w, "--s1-k", int(ref["n_rows_ref"]), "--s1-special", 1] if arm.startswith("S1gw")
                     else ["--rows", ",".join(map(str, ref["rows"]))])
            if "vtail" in arm:
                extra += ["--vtail", str(200 - int(arm.rsplit("-k", 1)[1]))]
            command = [args.native_dump, "--model", args.fixture / "tiny-qwen35-f32.gguf", "--projector", args.fixture / "projector-s4-f32.gguf",
                       "--tokens", args.work / (name + ".bin"), "--output", out, "--mode", "quality", "--tail", 200,
                       "--keep", 200, "--ubatch", 512, "--sweep", "--sweep-boundary", *extra]
            result = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=60)
            out.with_suffix(".log").write_text(result.stdout + result.stderr)
            assert result.returncode == 0, result.stderr
            logits = np.fromfile(str(out) + ".logits", dtype=np.float32).reshape(ref["logits"].shape)
            assert np.max(np.abs(logits - ref["logits"])) < 1e-4
            if arm in ("S1gw", "S1gw-vtail-k72"):
                got = outputs[name, arm]
                mass = np.fromfile(str(out) + ".mass_s1.f64", dtype=np.float64)
                assert np.max(np.abs(mass - got["s1_mass"])) < 1e-4
                assert got["exact_rows"] == np.fromfile(str(out) + ".rows.i32", dtype=np.int32).tolist()
            if "vtail" in arm:
                scale = np.fromfile(str(out) + ".vtail.f32", dtype=np.float32)
                assert np.max(np.abs(scale - ref["vtail_scales"].reshape(-1))) < 1e-4
            if arm in ("S1gw-vtail-k72", "T3s-vtail-k72"):
                assert np.max(np.abs(scale - outputs[name, arm]["vtail"]["scales"])) < 1e-4
            native_checks += 1
    print(f"PASS native harness: {native_checks}/12 arms including c=200; scorer/native mass, rows and vtail scales agree", flush=True)

    pair_names = ("fixture", "selchat", "fixture")
    pair = [dict(name=f"{name}-{i}", ids=prompts[name].tolist()) for i, name in enumerate(pair_names)]
    (args.work / "paired.json").write_text(json.dumps(pair))
    reused = run("paired", ["--s1", "on", "--s1-window", w, "--special-exact", "on", "--vtail", "on", "--vtail-rows", 128], chunk=512)
    assert len(reused) == len(pair)
    for name, got in zip(pair_names, reused):
        single = outputs[name, "S1gw-vtail-k72"]
        for key in ("logprobs", "logsumexp", "exact_rows", "s1_mass", "s1", "vtail", "ridgefill"):
            assert got[key] == single[key], (name, key)
    print("PASS repeated requests: reused context == fresh contexts, vtail enabled again after each seq_set", flush=True)

    sys.path[:0] = [str(args.kva / "q27"), str(args.fixture.parents[1])]
    import engine_ref_s1_vtail as oracle
    import convert_ridgefill_to_gguf as converter
    import torch
    from safetensors.numpy import load_file
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    model = oracle.Qwen35(args.fixture / "tiny-qwen35-hf", device="cpu", dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(str(args.fixture / "tiny-qwen35-hf"), local_files_only=True)
    package = args.fixture / "projector-s4"
    package_manifest, _ = converter.read_package(package)
    maps = {il: converter.package_map(package, package_manifest, il) for il in range(split, dm.L)}
    projector = oracle.Projector(split, {il: torch.from_numpy(pair[0]) for il, pair in maps.items()},
                                 {il: torch.from_numpy(pair[1]) for il, pair in maps.items()})
    scores = load_file(str(package / "rowsel.safetensors"))["score"]
    assert tokenizer is not None, "reference tokenizer failed to load"
    special_ids = set(tokenizer.added_tokens_decoder)
    digit_ids = oracle.SEL.digit_ids(tokenizer)
    for name, values in prompts.items():
        end = n - 64
        window = {}

        def capture(il, hidden):
            if il == 3:
                window[il] = hidden[end - 64:end].clone()

        with torch.no_grad():
            tokens = torch.tensor(values.astype(np.int64))
            hs, lower = oracle.run_lower(model, tokens, {split}, hook=capture)
            mass_q = oracle.SEL.window_mass(model, window, lower, [3], end=end).numpy()
            mass_s1 = oracle.s1_mass_gdnwin(model, projector, hs[split], lower, end, p, w).numpy()
            special = {r for r in range(p) if values[r] in special_ids}
            base = set(oracle.class_rows(values.tolist(), p, scores, 0.5)) | set(oracle.SEL.digits(values.tolist(), p, digit_ids)) | special
            base |= set(sorted(range(p), key=lambda r: (-mass_q[r], r))[:round(0.2 * p)])
            rows = sorted(set(sorted(range(p), key=lambda r: (-mass_s1[r], r))[:len(base)]) | special)
            for vtail in (False, True):
                ref, _ = oracle.run_arm(model, projector, tokens, hs[split], lower, rows, 72 if vtail else None)
                extra = ["--s1", "on", "--s1-window", w, "--special-exact", "on", "--vtail", "on" if vtail else "off", "--vtail-rows", 128]
                got = run(name, extra, endpoint=None)
                assert got["s1"]["n_base"] == len(base) and got["exact_rows"] == rows
                assert got["s1"]["window"] == [end - w, end]
                assert np.max(np.abs(got["s1_mass"] - mass_s1)) < 1e-4
                logits = ref["logits"][-65:-1].astype(np.float64)
                lse = np.max(logits, axis=-1) + np.log(np.exp(logits - np.max(logits, axis=-1, keepdims=True)).sum(axis=-1))
                lp = logits[np.arange(64), values[-64:]] - lse
                assert max(np.max(np.abs(got["logprobs"] - lp)), np.max(np.abs(got["logsumexp"] - lse))) < 1e-4
                future = values.copy(); future[-64:] = (future[-64:] + 17) % dm.V
                (args.work / "future.json").write_text(json.dumps(future.tolist()))
                changed = run("future", extra, endpoint="known")
                for key in ("s1_mass", "exact_rows", "qattn_window", "vtail"):
                    assert changed[key] == got[key], (name, vtail, key)
                assert changed["s1"]["n_base"] == got["s1"]["n_base"]
                assert changed["logsumexp"][0] == got["logsumexp"][0]
        print(f"PASS {name}: N-K default matches reference mass/base/rows/logits; future targets do not change selection or calibration", flush=True)

    for factor in (0, 0.5, 1.5, 3):
        got = run("fixture", ["--s1", "on", "--s1-window", w, "--special-exact", "on", "--s1-factor", factor])
        ref = np.load(args.reference / "fixture-S1gw.npz")
        count = min(p, max(1, round(factor * int(ref["n_rows_ref"]))))
        special = {r for r in range(p) if ids[r] in special_ids}
        want = sorted(set(sorted(range(p), key=lambda r: (-ref["mass_s1"][r], r))[:count]) | special)
        assert got["exact_rows"] == want
    print("PASS S1 factor: zero minimum, ties-to-even budgets 108.5/325.5, saturation at P", flush=True)
    run("fixture", ["--s1", "on", "--s1-window", 201], refused="RidgeFill: S1 window must be nonempty, within LOWER rows and not overlap the prefix")
    run("fixture", ["--vtail", "on", "--vtail-rows", 200], refused="would read scored NLL targets")
    result = dict(result="PASS", checks=checks, native_checks=native_checks, worst_logprob_lse=float(worst), worst_mass_relative=float(mass_error),
                  cost="includes recomputed prefix projector/KV, dense all-cell probe attention, two vtail V projections and cache read/write bytes",
                  vtail_k0="PASS native harness; scorer target-reading calibration refused by name")
    (args.work / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
