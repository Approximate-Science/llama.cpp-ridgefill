#!/usr/bin/env python3
"""G5 refusal controls, L-A's independent instrument (written from L-B's claims, not from its checker).
Usage: g5_refusals.py BIN_DIR QWEN35_FIXTURE_DIR DENSE_FIXTURE_DIR OUT_DIR [PORT]
  BIN_DIR: llama-server / llama-bench of the build under test; QWEN35_FIXTURE_DIR: tiny-qwen35-f32.gguf,
  projector-s3-f32.gguf, tokens.bin; DENSE_FIXTURE_DIR: qproj-s2-f32.gguf (a dense qwen3 projector).
Checks (L-B's 15 claims + 2 of L-A's): named startup refusals (LoRA, control vector, embeddings, --cache-reuse,
-sm row), named bench refusals (-sm row, embeddings), a stock missing-LoRA control (the normal loader error, no
RidgeFill message), a per-request n_cache_reuse refusal, the fill active on a 700-token prompt (500 filled rows),
a filled-slot save refused with no file written, an off request exact (0 filled), an exact-slot save that works and
writes a nonempty file; L-A: a dense projector on the qwen35 model refused by architecture, quality with alpha 1
refused by the rho limitation. A case that ran nothing is FAIL, never PASS."""
import json, socket, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

if len(sys.argv) < 5:
    sys.exit(__doc__)
BIN, Q35, DENSE, OUT = (Path(a).resolve() for a in sys.argv[1:5])
PORT = int(sys.argv[5]) if len(sys.argv) > 5 else 18731
OUT.mkdir(parents=True, exist_ok=True)
MODEL, PROJ = Q35 / "tiny-qwen35-f32.gguf", Q35 / "projector-s3-f32.gguf"
TOKENS = [int.from_bytes((Q35 / "tokens.bin").read_bytes()[i:i + 4], "little", signed=True) for i in range(0, 2800, 4)]
MISSING = OUT / "does-not-exist.gguf"
BASE = ["-m", str(MODEL), "-ngl", "0", "-t", "4", "--no-warmup", "-c", "1024", "-b", "700", "-ub", "128", "-np", "1"]
RF = ["--ridgefill", str(PROJ), "--ridgefill-tail", "200", "--ridgefill-min-prompt", "1"]
records = []

def record(name, ok, **kw):
    e = dict(check=name, result="PASS" if ok else "FAIL", **kw); records.append(e); print(json.dumps(e), flush=True)

def startup(name, args, needles, forbid=(), exe="llama-server"):
    log = OUT / f"{name}.txt"
    try:
        r = subprocess.run([str(BIN / exe), *args, "--port", str(PORT)] if exe == "llama-server" else [str(BIN / exe), *args],
                           capture_output=True, text=True, timeout=60)
        text = r.stdout + r.stderr
        rc = r.returncode
    except subprocess.TimeoutExpired as e:
        text, rc = (e.stdout or b"").decode() + (e.stderr or b"").decode(), "timeout (did not refuse)"
    log.write_text(text)
    ok = rc not in (0, "timeout (did not refuse)") and all(n in text for n in needles) and not any(f in text for f in forbid)
    record(name, ok, rc=rc, needles=list(needles))

def http(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")

# 1-5: startup refusals, named
for name, extra, flag in (("lora", ["--lora", str(MISSING)], "LoRA"), ("control-vector", ["--control-vector", str(MISSING)], "control vector"),
                          ("embeddings", ["--embeddings"], "embeddings"), ("cache-reuse", ["--cache-reuse", "16"], "--cache-reuse"),
                          ("split-mode-row", ["-sm", "row"], "row")):
    startup(f"startup-refuses-{name}", [*BASE, *RF, *extra], ("RidgeFill", flag))
# 6-7: bench refusals, named
for name, extra, flag in (("split-mode-row", ["-sm", "row"], "row"), ("embeddings", ["-embd", "1"], "embeddings")):
    startup(f"bench-refuses-{name}", ["-m", str(MODEL), "--ridgefill", str(PROJ), "-p", "64", "-n", "0", "-r", "1", *extra],
            ("RidgeFill", flag), exe="llama-bench")
# 8: stock control: without a projector a missing LoRA fails in the normal loader, not with RidgeFill's refusal
startup("stock-missing-lora-normal-loader", [*BASE, "--lora", str(MISSING)], (str(MISSING),), forbid=("RidgeFill",))
# L-A 1-2: architecture mismatch, quality alpha 1
startup("startup-refuses-arch-mismatch", [*BASE, "--ridgefill", str(DENSE / "qproj-s2-f32.gguf"), "--ridgefill-min-prompt", "1"],
        ("qwen3", "qwen35"))
startup("startup-refuses-quality-alpha1", [*BASE, *RF, "--ridgefill-mode", "quality", "--ridgefill-alpha", "1"], ("rho",))

# 9-15: live server
slots = OUT / "slots"; slots.mkdir(exist_ok=True)
for f in slots.iterdir():
    f.unlink()
log = open(OUT / "server-live.txt", "w")
srv = subprocess.Popen([str(BIN / "llama-server"), *BASE, *RF, "--slot-save-path", str(slots), "--port", str(PORT), "--host", "127.0.0.1"],
                       stdout=log, stderr=subprocess.STDOUT)
try:
    for _ in range(300):
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                pass
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                break
        except Exception:
            time.sleep(0.1)
    st, r = http("/completion", dict(prompt=TOKENS, n_predict=1, n_cache_reuse=16, cache_prompt=False))
    record("request-refuses-n_cache_reuse", st >= 400 and "RidgeFill" in json.dumps(r) and "cache_reuse" in json.dumps(r), status=st)
    st, r = http("/completion", dict(prompt=TOKENS, n_predict=1, cache_prompt=True))
    record("positive-fill-active-500-of-700", st == 200 and r.get("timings", {}).get("prompt_filled_n") == 500, status=st,
           filled=r.get("timings", {}).get("prompt_filled_n"))
    st, r = http("/slots/0?action=save", dict(filename="approx.bin"))
    record("slot-save-refuses-filled-rows", st >= 400 and "RidgeFill" in json.dumps(r), status=st)
    record("refused-save-wrote-no-file", not any(slots.iterdir()), files=[f.name for f in slots.iterdir()])
    st, r = http("/completion", dict(prompt=TOKENS, n_predict=1, ridgefill=False, cache_prompt=True))
    record("negative-off-request-exact", st == 200 and r.get("timings", {}).get("prompt_filled_n", 0) == 0, status=st,
           filled=r.get("timings", {}).get("prompt_filled_n", 0))
    st, r = http("/slots/0?action=save", dict(filename="exact.bin"))
    record("negative-exact-slot-save-works", st == 200 and r.get("n_saved", 0) > 0, status=st, n_saved=r.get("n_saved"))
    f = slots / "exact.bin"
    record("exact-slot-file-nonempty", f.is_file() and f.stat().st_size > 0, bytes=f.stat().st_size if f.is_file() else 0)
finally:
    srv.terminate()
    try:
        srv.wait(timeout=20)
    except subprocess.TimeoutExpired:
        srv.kill()
    log.close()

summary = dict(gate="G5-refusals (L-A instrument)", bin=str(BIN), checks=len(records), passed=sum(r["result"] == "PASS" for r in records),
               result=("PASS" if all(r["result"] == "PASS" for r in records) else "FAIL") if records else "SKIPPED")
(OUT / "results.json").write_text(json.dumps(dict(summary=summary, checks=records), indent=2) + "\n")
print(json.dumps(summary))
sys.exit(0 if summary["result"] == "PASS" else 1)
