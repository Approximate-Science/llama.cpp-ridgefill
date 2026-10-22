#!/usr/bin/env python3
"""Compute-buffer gate: RidgeFill must not grow the compute buffer with prompt length, and the load-time reserve must
cover every RidgeFill graph. Earlier gates checked correctness and speed only and missed a probe whose buffer grew as
n_kv x window x heads (9.4 GiB at 32K on Qwen3.8-27B, re-allocated mid-request).

Runs llama-ridgefill-score on the tiny qwen35 fixture, stock (mode off) and each RidgeFill mode, at several prompt
lengths with the same context size, batch and ubatch, and parses llama.cpp's own lines:
  sched_reserve: <buft> compute buffer size = X MiB                       (reserved at load)
  ~llama_context: <buft> compute buffer size is X MiB, matches expectation of Y MiB
  ~llama_context: <buft> compute buffer size of X MiB, does not match expectation of Y MiB
FAIL when: any "does not match expectation" line (a graph needed more than the reserve: re-allocation mid-request);
a mode's buffer exceeds stock's at the same length by more than --bound-mib; the excess over stock grows with length
by more than --growth-mib between the shortest and the longest length (an n_kv x n_tokens x n_head term would);
the Tier-3 question-mass graph (its own scheduler, allocated per prompt: "RidgeFill question-mass graph: <backend>
compute buffer X MiB") exceeds --bound-mib or grows across lengths by more than --growth-mib;
a run fails or its report shows the mode did not run (nothing filled, no S1 window, no vtail rows).
Fail at load (Linux, RLIMIT_AS): the smallest address-space limit at which stock (mode off) runs is searched; with
16 MiB more (less than RidgeFill's reserve excess) a RidgeFill mode must refuse at load, by name ("RidgeFill refused:
the compute buffers of the RidgeFill graphs"), before any decode; with 256 MiB more it must run (positive control).
Usage: check_compute_buffer.py --binary SCORER --fixture DIR --out NEW_DIR [--lengths 8192,16384,32768]
  DIR holds tiny-qwen35-f32.gguf and projector-s4-f32.gguf. A real model: --model GGUF --projector GGUF --vocab N
  --gpu-layers 99 (random token IDs below N; the fail-at-load check needs the tiny CPU fixture: --no-load-check)."""
import argparse, json, os, random, re, subprocess, sys
from pathlib import Path

MODES = {
    "off": ["--mode", "off"],
    "speed": ["--mode", "speed"],
    "quality": ["--mode", "quality"],
    "qattn": ["--mode", "quality+qattn"],
    "s1": ["--mode", "quality+qattn", "--ridgefill-s1-window", "256", "--special-exact", "on"],
    "vtail": ["--mode", "quality+qattn", "--ridgefill-vtail", "on", "--ridgefill-vtail-rows", "512"],
    "s1+vtail": ["--mode", "quality+qattn", "--ridgefill-s1-window", "256", "--special-exact", "on",
                 "--ridgefill-vtail", "on", "--ridgefill-vtail-rows", "512"],
}
RESERVE = re.compile(r"sched_reserve:\s+(\S+) compute buffer size =\s+([0-9.]+) MiB")
QMASS = re.compile(r"RidgeFill question-mass graph: (\S+) compute buffer ([0-9.]+) MiB")
FINAL = re.compile(r"~llama_context:\s+(\S+) compute buffer size (?:is|of)\s+([0-9.]+) MiB, (matches|does not match) expectation of\s+([0-9.]+) MiB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--fixture", type=Path, help="tiny fixture directory (default model / projector)")
    ap.add_argument("--model", type=Path)
    ap.add_argument("--projector", type=Path)
    ap.add_argument("--vocab", type=int, default=300, help="random token IDs in [3, vocab)")
    ap.add_argument("--gpu-layers", type=int, default=0)
    ap.add_argument("--ctx-size", type=int, default=0, help="default: the longest length")
    ap.add_argument("--cache-type", default="f16")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lengths", default="8192,16384,32768")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--flash-attn", default="on", help="comma list of on/off")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--ubatch", type=int, default=512)
    ap.add_argument("--bound-mib", type=float, default=512.0, help="max excess over stock at any length")
    ap.add_argument("--growth-mib", type=float, default=4.0, help="max growth of the excess over stock across lengths")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-load-check", action="store_true", help="skip the fail-at-load check (reported SKIPPED)")
    args = ap.parse_args()
    if not args.model:
        args.model = args.fixture / "tiny-qwen35-f32.gguf" if args.fixture else None
    if not args.projector:
        args.projector = args.fixture / "projector-s4-f32.gguf" if args.fixture else None
    if not args.model or not args.projector:
        sys.exit("--fixture, or --model and --projector, are required")
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"--out {args.out} must be a new or empty directory")
    args.out.mkdir(parents=True, exist_ok=True)
    lengths = [int(x) for x in args.lengths.split(",")]
    modes = args.modes.split(",")
    if "off" not in modes or len(lengths) < 2 or any(m not in MODES for m in modes):
        sys.exit("--modes must include off and known modes; --lengths needs at least two lengths")

    records = []
    def record(name, ok, **kw):
        e = dict(check=name, result="PASS" if ok else "FAIL", **kw); records.append(e); print(json.dumps(e), flush=True)

    rng = random.Random(1234)
    table = {}
    qtable = {}
    for fa in args.flash_attn.split(","):
        for n in lengths:
            doc = args.out / f"doc-{n}.jsonl"
            if not doc.exists():
                doc.write_text(json.dumps(dict(name=f"rand{n}", ids=[rng.randrange(3, args.vocab) for _ in range(n)])) + "\n")
            for mode in modes:
                tag = f"fa{fa}-{n}-{mode}"
                out = args.out / f"{tag}.jsonl"
                cmd = [str(args.binary), "--model", str(args.model), "--projector", str(args.projector), "--input", str(doc),
                       "--output", str(out), "--ctx-size", str(args.ctx_size or max(lengths)), "--batch", str(args.batch),
                       "--ubatch", str(args.ubatch), "--flash-attn", fa, "--cache-type", args.cache_type, "--threads", str(args.threads),
                       "--gpu-layers", str(args.gpu_layers),
                       "--tail", "4096", "--scored", "512", "--min-prompt", "1024", *MODES[mode]]
                r = subprocess.run(cmd, capture_output=True, text=True)
                log = r.stdout + r.stderr
                (args.out / f"{tag}.log").write_text(" ".join(cmd) + "\n" + log)
                reserve = {m.group(1): float(m.group(2)) for m in RESERVE.finditer(log)}
                final = {m.group(1): (float(m.group(2)), m.group(3), float(m.group(4))) for m in FINAL.finditer(log)}
                ran, why = r.returncode == 0 and bool(final), ""
                if ran and mode != "off":
                    rep = json.loads(out.read_text().splitlines()[0]).get("ridgefill", {})
                    ran = rep.get("n_filled", 0) > 0
                    if "s1" in mode:
                        ran = ran and bool(rep.get("s1", {}).get("window"))
                    if "vtail" in mode:
                        ran = ran and rep.get("vtail", {}).get("rows", 0) > 0
                    why = "" if ran else "the report shows the mode did not run"
                record(f"{tag}-ran", ran, rc=r.returncode, why=why or None)
                mismatch = {b: v for b, v in final.items() if v[1] != "matches"}
                record(f"{tag}-reserve-covers-graph", ran and not mismatch,
                       final_mib={b: v[0] for b, v in final.items()}, expected_mib={b: v[2] for b, v in final.items()},
                       reserve_mib=reserve)
                table[(fa, n, mode)] = {b: v[0] for b, v in final.items()}
                if mode != "off" and MODES[mode][1] == "quality+qattn":
                    q = [float(m.group(2)) for m in QMASS.finditer(log)]
                    record(f"{tag}-question-mass-buffer-reported", ran and bool(q), mib=max(q) if q else None)
                    qtable[(fa, n, mode)] = max(q) if q else None
        for mode in modes:
            if mode == "off":
                continue
            excess = {}
            for n in lengths:
                s, m = table.get((fa, n, "off"), {}), table.get((fa, n, mode), {})
                if s and m and set(s) == set(m):
                    excess[n] = max(m[b] - s[b] for b in s)
            ok = len(excess) == len(lengths)
            record(f"fa{fa}-{mode}-excess-over-stock-bounded", ok and max(excess.values()) <= args.bound_mib,
                   excess_mib=excess, bound_mib=args.bound_mib)
            growth = excess[lengths[-1]] - excess[lengths[0]] if ok else None
            record(f"fa{fa}-{mode}-excess-does-not-grow-with-length", ok and growth <= args.growth_mib,
                   growth_mib=growth, limit_mib=args.growth_mib)

        for mode in modes:
            if mode == "off" or MODES[mode][1] != "quality+qattn":
                continue
            q = {n: qtable.get((fa, n, mode)) for n in lengths}
            ok = all(v is not None for v in q.values())
            record(f"fa{fa}-{mode}-question-mass-buffer-bounded", ok and max(q.values()) <= args.bound_mib
                   and q[lengths[-1]] - q[lengths[0]] <= args.growth_mib, mib=q, bound_mib=args.bound_mib, growth_limit_mib=args.growth_mib)

    load_check = "SKIPPED: --no-load-check" if args.no_load_check else None
    if not args.no_load_check:
        try:
            import resource
        except ImportError:
            resource = None
        if resource is None or not sys.platform.startswith("linux"):
            load_check = "SKIPPED: needs RLIMIT_AS (Linux)"
        else:
            n = lengths[0]
            def limited(mode, kib):
                out = args.out / f"load-{mode}-{kib}.jsonl"
                cmd = [str(args.binary), "--model", str(args.model), "--projector", str(args.projector),
                       "--input", str(args.out / f"doc-{n}.jsonl"), "--output", str(out), "--ctx-size", str(args.ctx_size or max(lengths)), "--batch", str(args.batch), "--ubatch", str(args.ubatch),
                       "--flash-attn", "on", "--cache-type", "f16", "--threads", "1",
                       "--tail", "4096", "--scored", "512", "--min-prompt", "1024", *MODES[mode]]
                lim = lambda: resource.setrlimit(resource.RLIMIT_AS, (kib*1024, kib*1024))
                r = subprocess.run(cmd, capture_output=True, text=True, preexec_fn=lim, env=dict(os.environ, OMP_NUM_THREADS="1"))
                (args.out / f"load-{mode}-{kib}.log").write_text(r.stdout + r.stderr)
                return r.returncode, r.stdout + r.stderr
            lo, hi = 64*1024, 64*1024*1024 // 1024 * 16  # KiB: 64 MiB .. 16 GiB
            if limited("off", hi)[0] != 0:
                load_check = "FAIL: stock does not run at the upper limit"
            else:
                while hi - lo > 1024:
                    mid = (lo + hi)//2
                    lo, hi = (lo, mid) if limited("off", mid)[0] == 0 else (mid, hi)
                rc_off, _ = limited("off", hi + 16*1024)
                rc_rf, log_rf = limited("speed", hi + 16*1024)
                rc_ok, _ = limited("speed", hi + 256*1024)
                named = "RidgeFill refused: the compute buffers of the RidgeFill graphs" in log_rf
                decoded = "NLL=" in log_rf
                record("fail-at-load-ridgefill-refused-by-name", rc_off == 0 and rc_rf != 0 and named and not decoded and rc_ok == 0,
                       stock_min_kib=hi, limit_kib=hi + 16*1024, stock_rc=rc_off, ridgefill_rc=rc_rf, named=named,
                       decoded_before_refusal=decoded, positive_control_rc=rc_ok)

    rows = [dict(fa=fa, n=n, mode=mode, mib=v, question_mass_mib=qtable.get((fa, n, mode))) for (fa, n, mode), v in sorted(table.items())]
    s = dict(gate="compute-buffer", checks=len(records), passed=sum(r["result"] == "PASS" for r in records), load_check=load_check)
    s["result"] = ("PASS" if s["passed"] == s["checks"] and not (load_check or "").startswith("FAIL") else "FAIL") if records else "SKIPPED"
    (args.out / "results.json").write_text(json.dumps(dict(summary=s, checks=records, buffers=rows), indent=2) + "\n")
    print(json.dumps(dict(buffers=rows)))
    print(json.dumps(s))
    sys.exit(0 if s["result"] == "PASS" else 1)


if __name__ == "__main__":
    main()
