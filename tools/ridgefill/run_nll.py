#!/usr/bin/env python3
"""Paired native G3 NLL: one prefill per document/arm, reference qattn window, novel-token primary."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/ridgefill"))
from eval_client import paired_summary, read_items


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        items = read_items(args.docs, args.limit)
        if len(items) < 2:
            raise ValueError("BCa requires at least two documents; no measurement was made")
        spec = importlib.util.spec_from_file_location("ridgefill_nll_data", args.kva / "kva/data.py")
        if spec is None or spec.loader is None:
            raise ValueError("cannot import the oracle's kva/data.py")
        data = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data)
        prepared = []
        masks = {}
        for item in items:
            ids = item["ids"][:args.length]
            if len(ids) != args.length:
                raise ValueError(f"{item['name']}: fewer than --length tokens")
            prepared.append(dict(name=item["name"], ids=ids))
            masks[item["name"]] = data.novel_mask(ids, args.scored)
            if not any(masks[item["name"]]):
                raise ValueError(f"{item['name']}: no novel scored tokens; reference NLL is undefined")
        input_path = args.out / "tokens.jsonl"
        input_path.write_text("".join(json.dumps(item) + "\n" for item in prepared))
        native = {}
        commands = []
        modes = ("off",) if args.mode == "off" else ("off", args.mode)
        for mode in modes:
            output = args.out / (mode + ".jsonl")
            command = [str(args.binary), "--model", str(args.model), "--projector", str(args.projector),
                       "--input", str(input_path), "--output", str(output), "--mode", mode,
                       "--qattn-window-end", str(args.length - args.scored)]
            for option in ("scored", "tail", "min_prompt", "share", "digits", "alpha", "qattn_share", "qattn_window",
                           "special_exact", "header_rows", "qattn_window_mode", "qattn_turn_cap",
                           "gpu_layers", "threads", "batch", "ubatch", "cache_type", "h_type", "flash_attn", "ctx_size"):
                command += ["--" + option.replace("_", "-"), str(getattr(args, option))]
            commands.append(command)
            print(json.dumps(dict(stage="nll", mode=mode, documents=len(items), status="RUNNING")), flush=True)
            with (args.out / (mode + ".log")).open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            if len(rows) != len(items) or {row["name"] for row in rows} != set(masks):
                raise ValueError("native scorer returned missing/duplicate/unpaired documents")
            native[mode] = {row["name"]: row for row in rows}
            for row in rows:
                if (row["scored"] != args.scored or row["n"] != args.length or row["prefills"] != 1 or
                        row["target_positions"] != list(range(args.length - args.scored, args.length)) or
                        len(row["logprobs"]) != args.scored or len(row["logsumexp"]) != args.scored or
                        not all(math.isfinite(x) for x in row["logprobs"] + row["logsumexp"])):
                    raise ValueError("native scorer returned invalid scoring rows")
                if mode != "off":
                    expected_window = [args.length - args.scored - args.qattn_window, args.length - args.scored]
                    if row["prefix"] != args.length - args.tail or not row["selector_excludes_targets"]:
                        raise ValueError("candidate skipped fill or read scored targets")
                    if mode == "quality+qattn" and args.qattn_share > 0:
                        window = row["qattn_window"]
                        if args.qattn_window_mode == "last":
                            valid_window = window == expected_window
                        else:
                            end = args.length - args.scored
                            valid_window = isinstance(window, list) and len(window) == 2 and max(0, end - args.qattn_turn_cap) <= window[0] < window[1] == end
                        if not valid_window:
                            raise ValueError("candidate window differs from the reference protocol")
                    expected_options = dict(special_exact=args.special_exact == "on", header_rows=args.header_rows,
                                            qattn_window_mode=args.qattn_window_mode, qattn_turn_cap=args.qattn_turn_cap)
                    if any(row["parameters"].get(key) != value for key, value in expected_options.items()):
                        raise ValueError("candidate ignored chat selector options")
        metadata = dict(kind="metadata", metric="nll", build_label=args.build_label, commands=commands,
                        protocol="rf27_quality.py: frozen selector before K scored targets; novel-token primary",
                        docs_sha256=sha256(args.docs), projector_sha256=sha256(args.projector),
                        stats_sha256=sha256(args.kva / "finals/stats.py"), data_sha256=sha256(args.kva / "kva/data.py"),
                        args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
        (args.out / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
        records, all_records = [], []
        for item in items:
            name = item["name"]
            arms, all_arms = {}, {}
            for label, mode in (("reference", "off"), ("candidate", args.mode)):
                result = native[mode][name]
                mask = masks[name]
                values = [-lp for lp, novel in zip(result["logprobs"], mask) if novel]
                arms[label] = dict(result, nll=sum(values) / len(values), nll_all=result["nll"],
                                   novel=mask, novel_scored=len(values), token_subset="novel")
                all_arms[label] = dict(result, token_subset="all")
            common = dict(kind="nll", name=name, domain=item.get("domain"), candidate_mode=args.mode)
            records.append(dict(common, **arms))
            all_records.append(dict(common, **all_arms))
        summaries = []
        for metric, subset, rows in (("nll", "novel", records), ("nll_all", "all", all_records)):
            summaries.append(dict(metric=metric, length=None, status="MEASURED", token_subset=subset,
                                  **paired_summary(rows, "nll", args.kva)))
        summaries += [dict(metric=metric, status="SKIPPED", reason="native NLL scorer only")
                      for metric in ("needles", "ttft", "bench_prefill", "bench_decode")]
        with (args.out / "nll.jsonl").open("w") as file:
            for record in [metadata, *records, *[dict(kind="summary", result=s) for s in summaries[:2]]]:
                file.write(json.dumps(record, allow_nan=False) + "\n")
        result = dict(result="COMPLETE", scope="paired measurements; no quality or noninferiority verdict",
                      mode=args.mode, build=next(iter(native["off"].values()))["build"], summaries=summaries)
        (args.out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result
    except Exception as error:
        (args.out / "FAILED.json").write_text(json.dumps(dict(result="FAILED", reason=str(error))) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "model", "projector", "docs", "out", "kva"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--build-label", default="native-single-prefill")
    parser.add_argument("--mode", choices=("off", "speed", "quality", "quality+qattn"), default="quality+qattn")
    parser.add_argument("--special-exact", "--ridgefill-special-exact", choices=("on", "off"), default="off")
    parser.add_argument("--header-rows", "--ridgefill-header-rows", type=int, default=0)
    parser.add_argument("--qattn-window-mode", "--ridgefill-qattn-window-mode", choices=("last", "turn"), default="last")
    parser.add_argument("--length", type=int, default=16384)
    parser.add_argument("--scored", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tail", type=int, default=4096)
    parser.add_argument("--min-prompt", type=int, default=8192)
    parser.add_argument("--share", type=float, default=0.5)
    parser.add_argument("--digits", choices=("on", "off"), default="on")
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--qattn-share", type=float, default=0.2)
    parser.add_argument("--qattn-window", type=int, default=64)
    parser.add_argument("--qattn-turn-cap", "--ridgefill-qattn-turn-cap", type=int, default=256)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--ctx-size", type=int, default=0)
    parser.add_argument("--cache-type", choices=("f16", "f32", "bf16"), default="f16")
    parser.add_argument("--h-type", choices=("f16", "f32", "bf16"), default="f16")
    parser.add_argument("--flash-attn", choices=("off", "on", "auto"), default="auto")
    parser.add_argument("--timeout", type=float, default=7200)
    args = parser.parse_args()
    if args.header_rows < 0:
        parser.error("--header-rows must be nonnegative")
    if args.mode == "speed" and (args.special_exact == "on" or args.header_rows > 0):
        parser.error("--special-exact and --header-rows require quality modes")
    if args.mode not in ("off", "quality+qattn") and args.qattn_window_mode == "turn":
        parser.error("--qattn-window-mode turn requires quality+qattn")
    if not (0 < args.scored < args.tail < args.length and args.length >= args.min_prompt and
            args.qattn_window > 0 and (args.qattn_window_mode == "turn" or args.qattn_window <= args.length - args.scored) and
            args.qattn_turn_cap > 0 and args.header_rows >= 0 and (args.limit is None or args.limit >= 2)):
        parser.error("require 0 < scored < tail < length, length >= min-prompt, a valid window, and limit >= 2")
    print(json.dumps(run(args), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
