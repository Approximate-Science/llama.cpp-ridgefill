#!/usr/bin/env python3
"""G0 for llama-server (RidgeFill off is stock): the fork's server and upstream's, both without --ridgefill, on the
same model and flags. Usage: g0_server.py UPSTREAM_BIN FORK_BIN MODEL TOKENS_BIN OUT_DIR [PARALLEL]
Responses: 3 prompts (64 / 300 / 700 tokens of TOKENS_BIN) + a repeat of the last (prompt-cache hit), greedy,
16 tokens, top-5 probabilities: every response field except `timings` must be equal; /props must be equal except
build fields (the per-process random media marker is pinned through LLAMA_MEDIA_MARKER in both). Signals (behaviour report + checks): SIGTERM idle, SIGTERM during a long generation, SIGINT twice
during a generation: exit code and time to exit, for both servers; the stock exit codes must match.
A request that ran nothing is FAIL."""
import json, os, signal, socket, subprocess, sys, time, urllib.request
from pathlib import Path

if len(sys.argv) < 6:
    sys.exit(__doc__)
UP, FORK, MODEL, TOKENS, OUT = (Path(a).resolve() for a in sys.argv[1:6])
PAR = int(sys.argv[6]) if len(sys.argv) > 6 else 1
OUT.mkdir(parents=True, exist_ok=True)
raw = TOKENS.read_bytes()
TOK = [int.from_bytes(raw[i:i + 4], "little", signed=True) for i in range(0, len(raw), 4)]
PROMPTS = [TOK[:64], TOK[:300], TOK[:700], TOK[:700]]
records = []

def record(name, ok, **kw):
    e = dict(check=name, result="PASS" if ok else "FAIL", **kw); records.append(e); print(json.dumps(e), flush=True)

def free_port():
    # a fixed port can be held by a stray server, and a health probe would then talk to the wrong process
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]

def start(bin_dir, port, tag, ctx=None):
    log = open(OUT / f"server-{tag}.txt", "w")
    p = subprocess.Popen([str(bin_dir / "llama-server"), "-m", str(MODEL), "-ngl", "0", "-t", "4", "--no-warmup", "-c", str(ctx or 1024 * PAR),
                          "-np", str(PAR), "--host", "127.0.0.1", "--port", str(port)], stdout=log, stderr=subprocess.STDOUT,
                         env=dict(os.environ, LLAMA_MEDIA_MARKER="<__media_g0__>"))  # random per process otherwise
    for _ in range(300):
        if p.poll() is not None:
            raise RuntimeError(f"{tag} server exited at startup: {(OUT / f'server-{tag}.txt').read_text()[-400:]}")
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200:
                return p, log
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"{tag} server did not start")

def post(port, path, body, timeout=300):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

def responses(bin_dir, port, tag):
    p, log = start(bin_dir, port, tag)
    out = [post(port, "/completion", dict(prompt=pr, n_predict=16, temperature=0, n_probs=5, cache_prompt=True, ignore_eos=True, return_tokens=True)) for pr in PROMPTS]
    props = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=10).read())
    p.terminate(); p.wait(timeout=30); log.close()
    return out, props

def strip(r):
    return {k: v for k, v in r.items() if k != "timings"}

def strip_props(p):
    return {k: v for k, v in p.items() if k not in ("build_info",)}

ra, pa = responses(UP, free_port(), "up")
rb, pb = responses(FORK, free_port(), "fork")
for i, (a, b) in enumerate(zip(ra, rb)):
    same = json.dumps(strip(a), sort_keys=True) == json.dumps(strip(b), sort_keys=True)
    diff = sorted(k for k in set(a) | set(b) if k != "timings" and a.get(k) != b.get(k))
    # probabilities are withheld for partial-UTF-8 byte tokens (a random tiny model emits many): the generated ids
    # (return_tokens) are the guard that something was generated and compared
    record(f"response-{i}-{len(PROMPTS[i])}tok{'-cachehit' if i == 3 else ''}", same and len(a.get("tokens") or []) == 16,
           differing_fields=diff, cached=a.get("tokens_cached"), n_probs_entries=len(a.get("completion_probabilities") or []))
dp = sorted(k for k in set(pa) | set(pb) if k != "build_info" and pa.get(k) != pb.get(k))
record("props-equal", json.dumps(strip_props(pa), sort_keys=True) == json.dumps(strip_props(pb), sort_keys=True), differing_fields=dp)

def sig_case(bin_dir, port, tag, sigs, busy):
    p, log = start(bin_dir, port, tag, ctx=65536 if busy else None)
    if busy:
        # a long generation running when the signal arrives
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                     data=json.dumps(dict(prompt=TOK[:700], n_predict=60000, temperature=0, ignore_eos=True)).encode(),
                                     headers={"Content-Type": "application/json"})
        import threading
        th = threading.Thread(target=lambda: (lambda: urllib.request.urlopen(req, timeout=60))() if True else None, daemon=True)
        th.start(); time.sleep(0.3)
        slots = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/slots", timeout=10).read())
        if not any(sl.get("is_processing") for sl in slots):
            raise RuntimeError(f"{tag}: the generation is not running when the signal is sent (the busy case would test nothing)")
    t0 = time.time()
    for index, s in enumerate(sigs):
        p.send_signal(s)
        if index + 1 < len(sigs):
            # Wait for the first handler; back-to-back identical signals can coalesce.
            deadline = time.monotonic() + 1.0
            while "cleaning up before exit" not in (OUT / f"server-{tag}.txt").read_text():
                if p.poll() is not None or time.monotonic() >= deadline:
                    p.kill(); p.wait(); log.close()
                    raise RuntimeError(f"{tag}: first signal was not observed while the server remained busy")
                time.sleep(0.001)
            if p.poll() is not None:
                log.close()
                raise RuntimeError(f"{tag}: first signal ended the busy server before the second signal")
    try:
        rc = p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        p.kill(); rc = "hang"
    dt = time.time() - t0
    log.close()
    text = (OUT / f"server-{tag}.txt").read_text()
    return rc, round(dt, 3), "Received second interrupt" in text

behaviour = {}
for name, sigs, busy in (("sigterm-idle", [signal.SIGTERM], False), ("sigterm-busy", [signal.SIGTERM], True),
                         ("sigint-twice-busy", [signal.SIGINT, signal.SIGINT], True)):
    res = {}
    for tag, b in (("up", UP), ("fork", FORK)):
        res[tag] = dict(zip(("rc", "seconds", "second_interrupt_message"), sig_case(b, free_port(), f"{name}-{tag}", sigs, busy)))
    behaviour[name] = res
    # stock behaviour preserved: the same outcome as upstream (a server that waits for an in-flight request on SIGTERM
    # does so in both; "hang" = still running 20 s after the signal)
    if name == "sigint-twice-busy":
        # the second interrupt must terminate at once (exit 1). Upstream calls exit() and takes locks inside the
        # signal handler: its outcome is recorded (crash / abort / hang observed), not required
        record(f"signal-{name}-fork-exits-1-promptly", res["fork"]["rc"] == 1 and res["fork"]["seconds"] < 1.0, **res)
    else:
        record(f"signal-{name}-same-outcome-as-upstream", res["up"]["rc"] == res["fork"]["rc"], **res)

s = dict(gate="G0-server", parallel=PAR, model=MODEL.name, checks=len(records), passed=sum(r["result"] == "PASS" for r in records))
s["result"] = "PASS" if s["passed"] == s["checks"] else "FAIL"
(OUT / "results.json").write_text(json.dumps(dict(summary=s, checks=records, signals=behaviour), indent=2) + "\n")
print(json.dumps(s)); sys.exit(0 if s["result"] == "PASS" else 1)
