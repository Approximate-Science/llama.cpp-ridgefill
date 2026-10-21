#!/usr/bin/env python3
"""CPU split-4 scorer gate against kva finish(), including window and stock-logit controls."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "fixture", "kva", "work", "stock-dump"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.kva), str(ROOT), str(ROOT / "gguf-py")]
    from kva.model import Qwen35
    from kva.kva import Arm, Projector, run_lower, finish
    from kva.select import window_mass, digits, top_rows
    from gguf import GGUFReader, TokenType
    import convert_ridgefill_to_gguf as converter
    torch.set_num_threads(4)
    ids = np.fromfile(args.fixture / "tokens.bin", dtype='<i4').astype(np.int64)
    assert len(ids) == 700
    tokens = torch.from_numpy(ids)
    input_path = args.work / "tokens.json"
    input_path.write_text(json.dumps(dict(name="tiny", ids=ids.tolist())))
    model = Qwen35(args.fixture / "tiny-qwen35-hf", device="cpu", dtype=torch.float32)
    folder = args.fixture / "projector-s4"
    manifest, _ = converter.read_package(folder)
    weights, biases = {}, {}
    for il in range(4, 8):
        w, b = converter.package_map(folder, manifest, il)
        weights[il], biases[il] = torch.from_numpy(w), torch.from_numpy(b)
    projector = Projector(4, weights, biases)
    score = load_file(folder / "rowsel.safetensors")["score"]
    reader = GGUFReader(args.fixture / "tiny-qwen35-f32.gguf")
    pieces = reader.get_field("tokenizer.ggml.tokens").contents()
    types = reader.get_field("tokenizer.ggml.token_type").contents()
    excluded = {TokenType.UNKNOWN, TokenType.CONTROL, TokenType.USER_DEFINED, TokenType.UNUSED}
    digit_ids = {i for i, p in enumerate(pieces) if types[i] not in excluded and any('0' <= c <= '9' for c in p)}

    def run(mode, k, tail, chunk, suffix="", extra=(), source=input_path, window_end="known"):
        label = f"{mode}-k{k}-u{chunk}{suffix}"
        output = args.work / (label + ".jsonl")
        cmd = [args.binary, "--model", args.fixture / "tiny-qwen35-f32.gguf", "--projector", args.fixture / "projector-s4-f32.gguf",
               "--input", source, "--output", output, "--mode", mode, "--scored", k, "--tail", tail, "--min-prompt", 1,
               "--cache-type", "f32", "--h-type", "f32", "--flash-attn", "off", "--batch", 700, "--ubatch", chunk, "--diagnostics"]
        if window_end is not None:
            cmd += ["--qattn-window-end", len(ids) - k if window_end == "known" else window_end]
        cmd += extra
        with (args.work / (label + ".log")).open("w") as log:
            subprocess.run(list(map(str, cmd)), stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
        return json.loads(output.read_text())

    checks = 0
    with torch.no_grad():
        for k, tail in ((64, 200), (512, 600)):
            n, prefix, end = len(ids), len(ids) - tail, len(ids) - k
            win, layer_inputs = {}, {}
            def hook(il, h):
                if il == 3:
                    layer_inputs[il] = h.clone()
                    win[il] = h[end - 64:end].clone()
            hs, lower = run_lower(model, tokens, {4}, hook=hook)
            mass = window_mass(model, win, lower, [3], end=end).numpy()
            scores = score[tokens[:prefix]].clone()
            eligible = torch.isfinite(scores)
            scores[~eligible] = -torch.inf
            base = sorted(set(top_rows(scores, prefix, round(0.5 * int(eligible.sum())))) | set(digits(ids, prefix, digit_ids)))
            rows = sorted(set(base) | set(top_rows(torch.from_numpy(mass), prefix, round(0.2 * prefix))))
            expected = {}
            for mode in ("off", "speed", "quality", "quality+qattn"):
                arm = Arm("exact", split=4) if mode == "off" else Arm(mode, split=4, tail=tail, fill="proj", projector=projector)
                exact_rows = rows if mode == "quality+qattn" else (base if mode == "quality" else None)
                logits, _, _ = finish(model, tokens, hs[4], lower, arm, keep=k + 1, exact_rows=exact_rows, keep_cache=False)
                logits = logits[:-1].double()
                lse = torch.logsumexp(logits, -1).numpy()
                lp = logits[torch.arange(k), tokens[-k:]].numpy() - lse
                expected[mode] = (lp, lse)
            by_chunk = {}
            for chunk in (50, 128, 512):
                outputs = {}
                for mode in expected:
                    result = run(mode, k, tail, chunk)
                    lp, lse = expected[mode]
                    assert result["target_positions"] == list(range(end, n))
                    assert (result["logits_begin"], result["logits_end"]) == (end - 1, n - 1)
                    errors = [float(np.max(np.abs(np.asarray(result[key]) - target))) for key, target in (("logprobs", lp), ("logsumexp", lse))]
                    assert max(errors) <= 1e-4, (mode, k, chunk, errors)
                    assert abs(result["nll"] + lp.mean()) <= 1e-4
                    if mode == "quality+qattn":
                        got = np.asarray(result["question_mass"])
                        mass_error = float(np.abs(got - mass).max() / np.abs(mass).max())
                        assert mass_error < 1e-4, mass_error
                        assert result["exact_rows"] == rows
                        assert result["qattn_window"] == [end - 64, end] and result["selector_excludes_targets"]
                        assert not got[end:].any()
                    outputs[mode] = result
                    checks += 1
                    print(f"PASS {mode} K={k} ubatch={chunk}: max logprob/LSE error {max(errors):.3g}; NLL error {abs(result['nll'] + lp.mean()):.3g}", flush=True)
                zero = run("quality+qattn", k, tail, chunk, "-zero", ["--qattn-share", 0])
                assert zero["logprobs"] == outputs["quality"]["logprobs"] and zero["logsumexp"] == outputs["quality"]["logsumexp"]
                by_chunk[chunk] = outputs
            stock = args.work / f"stock-k{k}"
            with (args.work / f"stock-k{k}.log").open("w") as log:
                subprocess.run(list(map(str, [args.stock_dump, "--model", args.fixture / "tiny-qwen35-f32.gguf",
                    "--tokens", args.fixture / "tokens.bin", "--output", stock, "--mode", "off", "--keep", k + 1,
                    "--tail", tail, "--batch", 700, "--ubatch", 512])), stdout=log, stderr=subprocess.STDOUT, check=True)
            stock_logits = np.fromfile(str(stock) + ".logits", dtype='<f4').reshape(k + 1, 300)[:-1].astype(np.float64)
            stock_lse = np.logaddexp.reduce(stock_logits, axis=-1)
            stock_lp = stock_logits[np.arange(k), ids[-k:]] - stock_lse
            assert np.max(np.abs(stock_lp - by_chunk[512]["off"]["logprobs"])) < 1e-5
            print(f"PASS controls K={k}: zero share == quality byte-identical; off == stock perplexity-style next-token scoring", flush=True)
            changed = ids.copy(); changed[-k:] = (changed[-k:] + 1) % 300
            changed_path = args.work / f"changed-k{k}.json"
            changed_path.write_text(json.dumps(changed.tolist()))
            altered = run("quality+qattn", k, tail, 128, "-future", source=changed_path)
            original = by_chunk[128]["quality+qattn"]
            assert altered["question_mass"] == original["question_mass"] and altered["exact_rows"] == original["exact_rows"]
            assert altered["logsumexp"][0] == original["logsumexp"][0]
            print(f"PASS no target leakage K={k}: replacing all scored tokens preserves selector mass/rows and first predictive LSE", flush=True)
        boundaries = {}
        for boundary, endpoint in ((None, n), (n, n), (n - 32, n - 32)):
            result = run("quality+qattn", k, tail, 128, f"-end-{boundary}", window_end=boundary)
            mass = window_mass(model, {3: layer_inputs[3][endpoint - 64:endpoint]}, lower, [3], end=endpoint)
            rows = sorted(set(base) | set(top_rows(mass, prefix, round(0.2 * prefix))))
            logits, _, _ = finish(model, tokens, hs[4], lower, arm, keep=k + 1, exact_rows=rows, keep_cache=False)
            logits = logits[:-1].double()
            lse = torch.logsumexp(logits, -1).numpy()
            lp = logits[torch.arange(k), tokens[-k:]].numpy() - lse
            assert np.max(np.abs(result["question_mass"] - mass.numpy())) / mass.abs().max().item() < 1e-4
            assert result["exact_rows"] == rows and result["qattn_window"] == [endpoint - 64, endpoint]
            assert result["selector_excludes_targets"] == (endpoint <= n - k)
            assert np.max(np.abs(result["logprobs"] - lp)) < 1e-4
            assert np.max(np.abs(result["logsumexp"] - lse)) < 1e-4
            boundaries[boundary] = result
        for key in ("question_mass", "exact_rows", "logprobs", "logsumexp"):
            assert boundaries[None][key] == boundaries[n][key]
        assert boundaries[n - 32]["question_mass"] != boundaries[n]["question_mass"]
        for flag in ("--qattn-window-end", "--qattn-end"):
            alias = run("quality+qattn", k, tail, 128, f"-alias{flag}", [flag, "known"], window_end=None)
            for key in ("question_mass", "exact_rows", "logprobs", "logsumexp"):
                assert alias[key] == by_chunk[128]["quality+qattn"][key]
        print("PASS window endpoints: default == N; N and N-32 match oracle mass/rows/logits; known and legacy alias == N-K", flush=True)
    changed = ids[::-1].copy()
    changed[-16:] = changed[:16]
    docs_path = args.work / "paired-docs.jsonl"
    docs_path.write_text("".join(json.dumps(dict(name=name, ids=values.tolist())) + "\n"
                                 for name, values in (("a", ids), ("b", changed))))
    paired_dir = args.work / "paired"
    with (args.work / "paired.log").open("w") as log:
        subprocess.run(list(map(str, [sys.executable, ROOT / "tools/ridgefill/run_nll.py", "--binary", args.binary,
            "--model", args.fixture / "tiny-qwen35-f32.gguf", "--projector", args.fixture / "projector-s4-f32.gguf",
            "--docs", docs_path, "--out", paired_dir, "--kva", args.kva, "--length", 700, "--scored", 512,
            "--tail", 600, "--min-prompt", 1, "--cache-type", "f32", "--h-type", "f32", "--flash-attn", "off",
            "--batch", 700, "--ubatch", 128])), stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    summary = json.loads((paired_dir / "summary.json").read_text())
    assert summary["summaries"][0]["units"] == 2 and summary["summaries"][0]["resamples"] == 5000
    assert summary["summaries"][0]["seed"] == 20261004
    provenance = json.loads((paired_dir / "provenance.json").read_text())
    for command in provenance["commands"]:
        assert command[command.index("--qattn-window-end") + 1] == str(len(ids) - 512)
    paired = [json.loads(line) for line in (paired_dir / "nll.jsonl").read_text().splitlines()]
    paired = [row for row in paired if row["kind"] == "nll"]
    for row in paired:
        for arm in ("reference", "candidate"):
            result = row[arm]
            lp, mask = np.array(result["logprobs"]), np.array(result["novel"])
            assert abs(result["nll"] + lp[mask].mean()) < 1e-12
    assert paired[1]["candidate"]["novel_scored"] == 496
    changed_path = args.work / "second-document.json"
    changed_path.write_text(json.dumps(changed.tolist()))
    for arm, mode in (("reference", "off"), ("candidate", "quality+qattn")):
        fresh = run(mode, 512, 600, 128, "-fresh", source=changed_path)
        for key in ("logprobs", "logsumexp", "exact_rows"):
            assert fresh[key] == paired[1][arm][key]
    print("PASS paired wrapper: real BCa, novel mask 496/512, multi-document reuse equals fresh contexts", flush=True)
    base_command = [args.binary, "--model", args.fixture / "tiny-qwen35-f32.gguf", "--projector",
                    args.fixture / "projector-s4-f32.gguf", "--input", input_path, "--output", args.work / "refused.jsonl"]
    for extra, message in ((["--tail", "512"], "tail must exceed scored"),
                           (["--alpha", "1"], "rho-scaled correction"),
                           (["--mode", "bad"], "invalid mode"),
                           (["--qattn-share", "nan"], "fraction must be")):
        result = subprocess.run(list(map(str, base_command + extra)), capture_output=True, text=True, timeout=15)
        assert result.returncode != 0 and message in result.stderr, (extra, result.stderr)
    print("PASS named refusals: prediction outside exact tail, unsupported rho correction, invalid mode, nonfinite share", flush=True)
    for endpoint, message in (("0", "positive row index"), ("-1", "positive row index"), ("bad", "positive row index"),
                              ("12x", "positive row index"), (str(1 << 40), "positive row index"),
                              ("701", "exceeds document length"), ("63", "shorter than --qattn-window")):
        extra = ["--tail", "600", "--min-prompt", "1", "--qattn-window-end", endpoint]
        result = subprocess.run(list(map(str, base_command + extra)), capture_output=True, text=True, timeout=15)
        assert result.returncode != 0 and message in result.stderr, (endpoint, result.stderr)
    print("PASS window refusals: zero, negative, malformed, overflowing, beyond document and insufficient history", flush=True)
    for bad_id in (-1, True, 1 << 40, 300):
        invalid = ids.tolist(); invalid[0] = bad_id
        path = args.work / "invalid-tokens.json"
        path.write_text(json.dumps(invalid))
        command = list(base_command)
        command[command.index("--input") + 1] = path
        result = subprocess.run(list(map(str, command + ["--mode", "off"])), capture_output=True, text=True, timeout=15)
        assert result.returncode != 0 and "token ID" in result.stderr, result.stderr
    print("PASS input refusals: negative, boolean, overflowing and out-of-vocabulary token IDs", flush=True)
    print(f"PASS native scorer: {checks}/{checks} four-mode oracle comparisons, two K values, three chunk sizes", flush=True)


if __name__ == "__main__":
    main()
