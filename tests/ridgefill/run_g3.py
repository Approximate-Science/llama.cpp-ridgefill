#!/usr/bin/env python3
"""Run paired G3 NLL, Qwen thinking-off needles, TTFT, and native prefill/decode benchmarks.

One server and one build supply both arms. Last-512 NLL uses raw next-token logprobs only, with
a fixed approximate prefix. Needle answer suffixes are removed before applying the chat template.
Unavailable modes fail by name; omitted metrics are recorded as SKIPPED. No quality verdict is
inferred from a completed measurement or a degenerate interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace

from eval_client import Client, EvalError, evaluate, measure_bench, measure_ttft, paired_summary, read_items, write_record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def projector_tail_min(path: Path) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
    import gguf
    reader = gguf.GGUFReader(path)
    field = reader.fields.get("adapter.ridgefill.tail_min")
    if field is None or field.types != [gguf.GGUFValueType.UINT32] or len(field.data) != 1:
        raise EvalError("G3 requires scalar UINT32 adapter.ridgefill.tail_min")
    value = int(field.parts[field.data[0]][0])
    if not 0 < value <= 2147483647:
        raise EvalError("adapter.ridgefill.tail_min must be a positive int32")
    return value


def build_info(binary: Path) -> dict:
    result = subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=True, timeout=30)
    version = result.stdout + result.stderr
    match = re.search(r"build (\d+), commit ([0-9a-z]+)", version)
    if match is None:
        raise EvalError(f"cannot identify the build of {binary}")
    return dict(build=int(match[1]), commit=match[2], version=version.strip())


def native_options(args, cache_type: str, include_paths: bool = True) -> list[str]:
    options = ["-ngl", str(args.gpu_layers), "-t", str(args.threads),
            "-b", str(args.batch), "-ub", str(args.ubatch), "-ctk", cache_type, "-ctv", cache_type,
            "-fa", args.flash_attn,
            "--ridgefill-share", str(args.share), "--ridgefill-digits", args.digits,
            "--ridgefill-alpha", str(args.alpha), "--ridgefill-qattn-share", str(args.qattn_share),
            "--ridgefill-qattn-window", str(args.qattn_window)] + (["--ridgefill-qattn-blind"] if args.qattn_blind else [])
    if args.s1_window != 0 or args.s1_factor != 1 or args.vtail != "off" or args.vtail_rows != 512:
        for option in ("s1_window", "s1_factor", "vtail", "vtail_rows"):
            options += ["--ridgefill-" + option.replace("_", "-"), str(getattr(args, option))]
    if include_paths:
        options += ["-m", str(args.model), "--ridgefill", str(args.projector), "--ridgefill-tail", str(args.tail)]
    return options


def measure(args, client: Client, server_props: dict, metadata: dict) -> list[dict]:
    summaries = []
    for metric in ("nll", "needles", "ttft", "bench_prefill", "bench_decode"):
        if metric not in args.metrics:
            summaries.append(dict(metric=metric, status="SKIPPED", reason="not requested by --metrics"))
            continue
        print(json.dumps(dict(stage=metric, status="RUNNING")), flush=True)
        kind = {"needles": "needle", "bench_prefill": "bench", "bench_decode": "bench"}.get(metric, metric)
        settings = SimpleNamespace(**vars(args))
        settings.command = kind
        settings.data = args.needles if metric == "needles" else args.docs
        settings.predict = args.decode_tokens if metric == "bench_decode" else (0 if kind == "bench" else 1)
        settings.lengths = [0] if metric == "bench_decode" else args.lengths
        settings.binary = args.bin_dir / "llama-bench"
        settings.bench_arg = native_options(args, args.bench_cache_type, include_paths=False) + args.bench_arg
        settings.bench_arg += ["--no-warmup"] if args.no_warmup else []
        path = args.out / (metric + ".jsonl")
        with path.open("x") as file:
            write_record(file, dict(kind="metadata", metric=metric, server=server_props, **metadata))
            if kind == "bench":
                if args.mode == "quality+qattn":
                    raise EvalError("llama-bench: quality+qattn requires the server's question-attention sweep; omit bench metrics explicitly")
                records = measure_bench(settings, file)
            elif metric == "ttft":
                records = measure_ttft(settings, client, file)
            else:
                records = evaluate(settings, client, file)
            groups = {length: [row for row in records if row["length"] == length] for length in settings.lengths} if kind in ("bench", "ttft") else {None: records}
            for length, rows in groups.items():
                summary = dict(metric=metric, length=length, status="MEASURED", **paired_summary(rows, kind, args.kva))
                summaries.append(summary)
                write_record(file, dict(kind="summary", result=summary))
    return summaries


def run(args) -> dict:
    args.out.mkdir(parents=True, exist_ok=False)
    server = None
    try:
        versions = {name: build_info(args.bin_dir / name) for name in ("llama-server", "llama-bench")}
        if versions["llama-server"]["commit"] != versions["llama-bench"]["commit"] or versions["llama-server"]["build"] != versions["llama-bench"]["build"]:
            raise EvalError("llama-server and llama-bench must come from the same build")
        for path in (args.model, args.projector, args.kva / "finals/stats.py"):
            if not path.is_file():
                raise EvalError(f"required input is missing: {path}")
        if args.tail_min != projector_tail_min(args.projector):
            raise EvalError("--tail-min does not match adapter.ridgefill.tail_min")
        for path in (args.docs, args.needles):
            if len(read_items(path, args.limit)) < 2:
                raise EvalError(f"{path}: at least two paired units are required for BCa")
        if args.ctx_size <= max([args.length, *args.lengths]) + args.decode_tokens:
            raise EvalError("--ctx-size must exceed the longest prompt plus --decode-tokens")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        startup_mode = args.mode if args.s1_window > 0 or args.vtail == "on" else "speed"
        command = [str(args.bin_dir / "llama-server"), *native_options(args, args.cache_type), "--ridgefill-mode", startup_mode,
                   "--ridgefill-min-prompt", str(args.min_prompt), "-c", str(args.ctx_size), "-np", "1", "--host", "127.0.0.1",
                   "--port", str(args.port), "--offline", "--jinja", "--reasoning", "off"]
        command += ["--chat-template-file", str(args.chat_template_file)] if args.chat_template_file else []
        command += ["--no-warmup"] if args.no_warmup else []
        command += args.server_arg
        client = Client(f"http://127.0.0.1:{args.port}", args.timeout)
        with (args.out / "server.log").open("w") as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            health = Client(client.url, min(5, args.startup_timeout))
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if server.poll() is not None:
                    raise EvalError(f"llama-server exited with {server.returncode}; see {args.out / 'server.log'}")
                try:
                    health.request("/health")
                    break
                except (urllib.error.URLError, EvalError):
                    if time.monotonic() >= deadline:
                        raise EvalError("llama-server startup timed out")
                    time.sleep(0.1)
            props = client.request("/props")
            if Path(props["model_path"]).resolve() != args.model or props["total_slots"] != 1:
                raise EvalError("server model/slot configuration differs from this paired run")
            metadata = dict(build_label=args.build_label, versions=versions, machine=platform.node(),
                            args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                            server_command=command, model=dict(path=str(args.model), bytes=args.model.stat().st_size),
                            projector_sha256=sha256(args.projector), docs_sha256=sha256(args.docs), needles_sha256=sha256(args.needles),
                            stats_sha256=sha256(args.kva / "finals/stats.py"),
                            chat_template_sha256=hashlib.sha256(props["chat_template"].encode()).hexdigest())
            (args.out / "provenance.json").write_text(json.dumps(dict(server=props, **metadata), indent=2) + "\n")
            summaries = measure(args, client, props, metadata)
            result = dict(result="COMPLETE", scope="paired measurements; no quality or noninferiority verdict", mode=args.mode,
                          build=versions["llama-server"], summaries=summaries)
            (args.out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            return result
    except Exception as error:
        (args.out / "FAILED.json").write_text(json.dumps(dict(result="FAILED", reason=str(error))) + "\n")
        raise
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bin-dir", "model", "projector", "docs", "needles", "out", "kva"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("off", "speed", "quality", "quality+qattn"), required=True)
    parser.add_argument("--build-label", required=True)
    parser.add_argument("--metrics", type=lambda value: value.split(','), default=["nll", "needles", "ttft", "bench_prefill", "bench_decode"])
    parser.add_argument("--length", type=int, default=16384)
    parser.add_argument("--lengths", type=lambda value: [int(x) for x in value.split(',')], default=[8192, 16384, 32768])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tail", type=int, default=4096)
    parser.add_argument("--tail-min", type=int, default=512)
    parser.add_argument("--min-prompt", type=int, default=8192)
    parser.add_argument("--ridgefill-s1-window", "--s1-window", dest="s1_window", type=int, default=0)
    parser.add_argument("--ridgefill-s1-factor", "--s1-factor", dest="s1_factor", type=float, default=1.0)
    parser.add_argument("--ridgefill-vtail", "--vtail", dest="vtail", choices=("on", "off"), default="off")
    parser.add_argument("--ridgefill-vtail-rows", "--vtail-rows", dest="vtail_rows", type=int, default=512)
    parser.add_argument("--share", type=float, default=0.5)
    parser.add_argument("--digits", choices=("on", "off"), default="on")
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--qattn-share", type=float, default=0.2)
    parser.add_argument("--qattn-window", type=int, default=64)
    parser.add_argument("--qattn-blind", action="store_true")
    parser.add_argument("--ctx-size", type=int, default=33792)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--cache-type", default="f16")
    parser.add_argument("--bench-cache-type", default="f16")
    parser.add_argument("--flash-attn", choices=("off", "on", "auto"), default="auto")
    parser.add_argument("--port", type=int, default=18089)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--chat-template-file", type=Path)
    parser.add_argument("--server-arg", action="append", default=[], help="one native argument; use --server-arg=FLAG for flags starting with '-'")
    parser.add_argument("--bench-arg", action="append", default=[])
    args = parser.parse_args()
    if not (0 <= args.s1_window <= 2147483647 and 1 <= args.vtail_rows <= 2147483647 and math.isfinite(args.s1_factor) and args.s1_factor >= 0):
        parser.error("invalid --ridgefill-s1-window/--ridgefill-s1-factor/--ridgefill-vtail-rows")
    if (args.s1_window > 0 or args.vtail == "on") and args.mode not in ("off", "quality+qattn"):
        parser.error("S1/vtail requires the quality+qattn two-sweep flow")
    for name in ("bin_dir", "model", "projector", "docs", "needles", "out", "kva", "chat_template_file"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if not args.metrics or len(set(args.metrics)) != len(args.metrics) or set(args.metrics) - {"nll", "needles", "ttft", "bench_prefill", "bench_decode"}:
        parser.error("--metrics must name distinct nll,needles,ttft,bench_prefill,bench_decode stages")
    if args.length < 513 or not args.lengths or any(x < 1 for x in args.lengths) or args.repeats < 3:
        parser.error("--length must be >=513, --lengths positive, and --repeats >=3")
    if min(args.tail, args.tail_min, args.ctx_size, args.threads, args.batch, args.ubatch, args.qattn_window, args.decode_tokens) < 1 or args.min_prompt < 0:
        parser.error("tail/context/batch/thread/window/decode counts must be positive; min-prompt nonnegative")
    if args.limit is not None and args.limit < 2:
        parser.error("--limit must be at least 2 for BCa")
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (args.share, args.alpha, args.qattn_share)):
        parser.error("share/alpha/qattn-share must be in [0,1]")
    if not 0 < args.port < 65536 or not all(math.isfinite(value) and value > 0 for value in (args.timeout, args.startup_timeout)):
        parser.error("invalid port or timeout")
    result = run(args)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (EvalError, OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(json.dumps(dict(result="FAILED", reason=str(error))), file=sys.stderr)
        sys.exit(1)
