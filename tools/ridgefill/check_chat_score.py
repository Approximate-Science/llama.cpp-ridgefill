#!/usr/bin/env python3
"""CPU chat-selector gate: synthetic ChatML, kva window_mass/finish, and pre-chat default equality."""
import argparse
import hashlib
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
    for name in ("binary", "baseline-binary", "fixture", "kva", "work"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.kva), str(ROOT), str(ROOT / "gguf-py")]
    from kva.model import Qwen35
    from kva.kva import Arm, Projector, run_lower, finish
    from kva.select import window_mass, digits, top_rows
    from gguf import GGUFReader, GGUFWriter, TokenType
    import convert_ridgefill_to_gguf as converter
    torch.set_num_threads(4)

    def copy_gguf(source, dest, replacements, prefix=False):
        reader = GGUFReader(source)
        writer = GGUFWriter(dest, "qwen35")
        for key, field in reader.fields.items():
            if key.startswith("GGUF.") or key == "general.architecture":
                continue
            writer.add_key_value(key, replacements.get(key, field.contents()), field.types[0],
                                 field.types[1] if len(field.types) > 1 else None)
        if prefix:
            writer.add_add_space_prefix(False)
        for tensor in reader.tensors:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()

    original = GGUFReader(args.fixture / "tiny-qwen35-f32.gguf")
    token_field = original.get_field("tokenizer.ggml.tokens")
    type_field = original.get_field("tokenizer.ggml.token_type")
    assert token_field is not None and type_field is not None, "tiny fixture lacks tokenizer metadata"
    pieces = token_field.contents()
    types = type_field.contents()
    # Reuse all weights and row scores. Only the two control-token spellings and whitespace-prefix policy change.
    pieces[1], pieces[2] = "<|im_start|>", "<|im_end|>"
    model_path, projector_path = args.work / "chat.gguf", args.work / "projector-s4.gguf"
    copy_gguf(args.fixture / "tiny-qwen35-f32.gguf", model_path, {"tokenizer.ggml.tokens": pieces}, prefix=True)
    digest = hashlib.sha256(b"".join(piece.encode() + b"\0" for piece in pieces)).hexdigest()
    copy_gguf(args.fixture / "projector-s4-f32.gguf", projector_path, {"adapter.ridgefill.vocab_sha256": digest})
    header = [1] + [byte + 3 for byte in b"user\n"]
    assistant = [1] + [byte + 3 for byte in b"assistant\n"]
    ids = np.fromfile(args.fixture / "tokens.bin", dtype="<i4").astype(np.int64)
    n, k, tail, end, prefix = len(ids), 64, 200, len(ids) - 64, len(ids) - 200
    crafted = ids.copy()
    for at, marker in ((20, header), (120, assistant), (300, header), (610, assistant), (660, header)):
        crafted[at:at + len(marker)] = marker
    for at, token in ((10, 0), (11, 2), (119, 2), (299, 2), (497, 1), (499, 2), (609, 2)):
        crafted[at] = token
    short = crafted.copy(); short[600:600 + len(header)] = header
    sources = {}

    def source(values):
        key = tuple(values.tolist())
        if key not in sources:
            path = args.work / f"tokens-{len(sources)}.json"
            path.write_text(json.dumps(dict(name="chat", ids=values.tolist())))
            sources[key] = path
        return sources[key]

    serial = 0
    def run(values, mode="quality+qattn", extra=(), binary=args.binary, chunk=128, refused=None, precision="f32"):
        nonlocal serial
        serial += 1
        output = args.work / f"score-{serial}.jsonl"
        command = [binary, "--model", model_path, "--projector", projector_path, "--input", source(values), "--output", output,
                   "--mode", mode, "--scored", k, "--tail", tail, "--min-prompt", 1, "--qattn-window-end", end,
                   "--cache-type", precision, "--h-type", precision, "--flash-attn", "off", "--batch", n, "--ubatch", chunk,
                   "--diagnostics", *extra]
        result = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=60)
        (args.work / f"score-{serial}.log").write_text(result.stdout + result.stderr)
        if refused:
            assert result.returncode != 0 and refused in result.stderr, result.stderr
            return None
        assert result.returncode == 0, result.stderr
        return json.loads(output.read_text())

    model = Qwen35(args.fixture / "tiny-qwen35-hf", device="cpu", dtype=torch.float32)
    folder = args.fixture / "projector-s4"
    manifest, _ = converter.read_package(folder)
    weights, biases = {}, {}
    for il in range(4, 8):
        w, b = converter.package_map(folder, manifest, il)
        weights[il], biases[il] = torch.from_numpy(w), torch.from_numpy(b)
    projector = Projector(4, weights, biases)
    scores = load_file(folder / "rowsel.safetensors")["score"]
    excluded = {TokenType.UNKNOWN, TokenType.CONTROL, TokenType.USER_DEFINED, TokenType.UNUSED}
    digit_ids = {i for i, piece in enumerate(pieces) if types[i] not in excluded and any('0' <= c <= '9' for c in piece)}
    special_ids = {0, 1, 2}
    lower_cache = {}
    checks, worst, worst_mass = 0, 0.0, 0.0

    def reference(values, special, headers, window_mode, cap, endpoint, share=0.2):
        key = tuple(values.tolist())
        tokens = torch.from_numpy(values)
        if key not in lower_cache:
            inputs = {}
            def hook(il, h):
                if il == 3:
                    inputs[il] = h.clone()
            hs, lower = run_lower(model, tokens, {4}, hook=hook)
            lower_cache[key] = hs, lower, inputs
        hs, lower, inputs = lower_cache[key]
        eligible = scores[tokens[:prefix]].clone()
        count = int(torch.isfinite(eligible).sum())
        eligible[~torch.isfinite(eligible)] = -torch.inf
        rows = set(top_rows(eligible, prefix, round(0.5 * count))) | set(digits(values, prefix, digit_ids))
        for i, token in enumerate(values[:prefix]):
            if token in special_ids:
                if special:
                    rows.add(i)
                rows.update(range(i + 1, min(prefix, i + headers + 1)))
        mass, begin = None, None
        if share:
            begin = endpoint - 64
            if window_mode == "turn":
                matches = [i for i in range(endpoint - len(header) + 1) if values[i:i + len(header)].tolist() == header]
                assert matches
                begin = max(matches[-1] + len(header), endpoint - cap)
                assert begin < endpoint
            mass = window_mass(model, {3: inputs[3][begin:endpoint]}, lower, [3], end=endpoint)
            rows.update(top_rows(mass, prefix, round(share * prefix)))
        rows = sorted(rows)
        logits, _, _ = finish(model, tokens, hs[4], lower, Arm("quality", split=4, tail=tail, fill="proj", projector=projector),
                              keep=k + 1, exact_rows=rows, keep_cache=False)
        logits = logits[:-1].double()
        lse = torch.logsumexp(logits, -1).numpy()
        lp = logits[torch.arange(k), tokens[-k:]].numpy() - lse
        return rows, mass, begin, lp, lse

    with torch.no_grad():
        for precision in ("f32", "f16"):
            for values in (ids, crafted):
                for mode in ("off", "speed", "quality", "quality+qattn"):
                    before = run(values, mode, binary=args.baseline_binary, precision=precision)
                    after = run(values, mode, precision=precision)
                    explicit = run(values, mode, ["--special-exact", "off", "--header-rows", 0, "--qattn-window-mode", "last", "--qattn-turn-cap", 256], precision=precision)
                    for field in ("logprobs", "logsumexp", "exact_rows", "qattn_window", "question_mass"):
                        assert before.get(field) == after.get(field) == explicit.get(field), (mode, precision, field)
        print("PASS defaults: old binary == new default == explicit defaults, raw/templated tokens, all four modes, F32/F16 KV and h_S", flush=True)
        cases = [(False, 0, "turn", 256, end, crafted), (True, 0, "last", 256, end, crafted),
                 (False, 3, "last", 256, end, crafted), (True, 3, "turn", 256, end, crafted),
                 (True, 3, "turn", 32, end, crafted), (True, 3, "turn", 1024, end, crafted),
                 (True, 2, "turn", 256, end, short), (True, 2, "turn", 256, n, crafted)]
        for special, headers, mode, cap, endpoint, values in cases:
            want_rows, mass, begin, lp, lse = reference(values, special, headers, mode, cap, endpoint)
            extra = ["--special-exact", "on" if special else "off", "--header-rows", headers,
                     "--qattn-window-mode", mode, "--qattn-turn-cap", cap, "--qattn-window-end", endpoint]
            for chunk in (50, 128, 512):
                got = run(values, extra=extra, chunk=chunk)
                error = max(np.max(np.abs(got["logprobs"] - lp)), np.max(np.abs(got["logsumexp"] - lse)))
                mass_error = np.max(np.abs(got["question_mass"] - mass.numpy())) / mass.abs().max().item()
                assert error < 1e-4 and mass_error < 1e-4, (error, mass_error)
                assert got["exact_rows"] == want_rows and got["qattn_window"] == [begin, endpoint]
                assert got["selector_excludes_targets"] == (endpoint <= end)
                worst, worst_mass = max(worst, error), max(worst_mass, mass_error)
                checks += 1
            print(f"PASS chat special={special} headers={headers} mode={mode} cap={cap} window=[{begin},{endpoint}) across 3 chunks", flush=True)
        chat_options = ["--special-exact", "on", "--header-rows", 3, "--qattn-window-mode", "turn"]
        original_result = run(crafted, extra=chat_options)
        changed = crafted.copy(); changed[-k:] = (changed[-k:] + 1) % 300
        changed[680:680 + len(header)] = header
        future = run(changed, extra=chat_options)
        for field in ("question_mass", "exact_rows", "qattn_window"):
            assert future[field] == original_result[field]
        assert future["logsumexp"][0] == original_result["logsumexp"][0]
        narrow = run(crafted, extra=chat_options + ["--qattn-window", 8])
        for field in ("question_mass", "exact_rows", "logprobs", "logsumexp"):
            assert narrow[field] == original_result[field]
        zero = run(crafted, extra=chat_options + ["--qattn-share", 0])
        quality = run(crafted, "quality", ["--special-exact", "on", "--header-rows", 3])
        for field in ("exact_rows", "logprobs", "logsumexp"):
            assert zero[field] == quality[field]
        want_rows, _, _, lp, lse = reference(crafted, True, 3, "last", 256, end, share=0)
        assert quality["exact_rows"] == want_rows
        assert np.max(np.abs(quality["logprobs"] - lp)) < 1e-4 and np.max(np.abs(quality["logsumexp"] - lse)) < 1e-4
        assert run(crafted)["exact_rows"] != run(crafted, extra=["--special-exact", "on", "--header-rows", 3])["exact_rows"]
        print("PASS controls: future user headers/targets excluded; turn ignores last-window size; zero share == chat quality; chat rows change selection", flush=True)

    docs = args.work / "paired-docs.jsonl"
    docs.write_text("".join(json.dumps(dict(name=name, ids=values.tolist())) + "\n" for name, values in (("long", crafted), ("short", short))))
    paired = args.work / "paired"
    command = [sys.executable, ROOT / "tools/ridgefill/run_nll.py", "--binary", args.binary, "--model", model_path,
               "--projector", projector_path, "--docs", docs, "--out", paired, "--kva", args.kva,
               "--length", n, "--scored", k, "--tail", tail, "--min-prompt", 1,
               "--cache-type", "f32", "--h-type", "f32", "--flash-attn", "off", "--batch", n, "--ubatch", 128,
               "--special-exact", "on", "--header-rows", 3, "--qattn-window-mode", "turn"]
    with (args.work / "paired.log").open("w") as log:
        subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    paired_results = [json.loads(line) for line in (paired / "quality+qattn.jsonl").read_text().splitlines()]
    for values, result in zip((crafted, short), paired_results):
        fresh = run(values, extra=chat_options)
        for field in ("exact_rows", "logprobs", "logsumexp", "qattn_window"):
            assert fresh[field] == result[field]
    summary = json.loads((paired / "summary.json").read_text())
    assert summary["result"] == "COMPLETE" and summary["summaries"][0]["units"] == 2
    assert summary["summaries"][0]["resamples"] == 5000
    print("PASS chat paired wrapper: both selector knobs, per-document turn bounds, fresh-context equality and real BCa", flush=True)

    absent = ids.copy(); absent[660:660 + len(header)] = header
    run(absent, extra=chat_options, refused="no user header before the window endpoint")
    empty = crafted.copy(); empty[end - len(header):end] = header
    run(empty, extra=chat_options, refused="empty user turn before the window endpoint")
    partial = ids.copy(); partial[end - 2:end - 2 + len(header)] = header
    run(partial, extra=chat_options, refused="no user header before the window endpoint")
    for extra, mode, message in ((["--qattn-turn-cap", 0], "quality+qattn", "--qattn-turn-cap must be positive"),
                                 (["--qattn-window-mode", "bad"], "quality+qattn", "--ridgefill-qattn-window-mode must be last or turn"),
                                 (["--header-rows", -1], "quality", "--ridgefill-header-rows must be nonnegative"),
                                 (["--special-exact", "on"], "speed", "require quality"),
                                 (["--header-rows", 3], "speed", "require quality"),
                                 (["--qattn-window-mode", "turn"], "quality", "turn requires quality+qattn")):
        run(crafted, mode, extra, refused=message)
    print("PASS named refusals: missing/future-only/partial/empty headers, invalid knobs and incompatible modes", flush=True)
    result = dict(result="PASS", oracle_checks=checks, worst_logprob_lse=float(worst), worst_mass_relative=float(worst_mass),
                  defaults="byte-identical to pre-chat binary", scope="tiny CPU correctness; no chat quality verdict")
    (args.work / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
