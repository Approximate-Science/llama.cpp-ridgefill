#!/usr/bin/env python3
"""G1 for the two-sweep dispatch (llama_ridgefill_seq_set_part: LOWER, then a LATE replay), tiny qwen35, split 3,
N = 700, T = 200, CPU F32, using L-B's compare_qwen35 (reference / read_state) unchanged.
Checks: sweep vs kva finish (speed alpha 0 and 1, quality R1 alpha 0) at 700/512, 128/128, 50/50 (final caches and
tail logits, 1e-4; a LATE replay rewrites the prefix states, so prefix dumps do not exist here); sweep vs the single
pass; the lower layers' caches byte-equal to the single pass (LOWER writes them, LATE must not touch them); the KV
cell count stays N (the replay reuses cells); h_S and window captures vs upstream's own layer-input extraction;
the replay after a stale prompt in the same sequence;
controls: wrong map fails, T >= N (all LATE rows exact) equals stock; refusals: a LOWER output, LATE without a
LOWER pass, LOWER on a non-empty sequence, FULL in the middle of a sweep, a part on a dense model. Nothing run = SKIPPED."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lb/tests/ridgefill"))
import compare_qwen35 as cq

KVA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/var/home/dylan/projects/research/kva-wt-q27")
W = HERE / "lb/out/gate1-qwen35"
OUT = HERE / "out/g1s"; OUT.mkdir(parents=True, exist_ok=True)
DUMP = HERE / "rf_dump_rs"
CFG = json.loads((W / "tiny-qwen35-hf/config.json").read_text())
SPLIT, N, T, KEEP, NE = 3, 700, 200, 200, CFG["hidden_size"]
R1 = [0, 7, 31, 127, 255, 499]
records = []

def record(name, ok, **kw):
    e = dict(check=name, result="PASS" if ok else "FAIL", **kw); records.append(e); print(json.dumps(e), flush=True)

def run(tag, b, ub, mode="speed", alpha=0, rows=None, proj="projector-s3-f32.gguf", tail=T, extra=()):
    pfx = OUT / tag
    cmd = [str(DUMP), "--model", str(W / "tiny-qwen35-f32.gguf"), "--tokens", str(W / "tokens.bin"), "--output", str(pfx),
           "--mode", mode, "--tail", str(tail), "--keep", str(KEEP), "--batch", str(b), "--ubatch", str(ub), "--alpha", str(alpha), "--cross"]
    if mode != "off":
        cmd += ["--projector", str(W / proj)]
    if rows:
        cmd += ["--rows", ",".join(map(str, rows))]
    cmd += list(extra)
    with open(str(pfx) + ".log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    return pfx, rc

def load(pfx):
    a = {"logits": np.fromfile(str(pfx) + ".logits", dtype=np.float32).reshape(KEEP, -1)}
    for k, v in cq.read_state(Path(str(pfx) + ".kv"), CFG, N).items():   # asserts N cells (no extra cells)
        a["final." + k] = v
    return a

def rel(a, b):
    return max(float(np.max(np.abs(a[k] - b[k])) / max(float(np.max(np.abs(b[k]))), 1e-30)) for k in b)

ref = {}
def reference(mode, alpha, rows):
    key = (mode, alpha, tuple(rows or ()))
    if key not in ref:
        r = cq.reference(KVA, W, SPLIT, mode, alpha, rows or [])
        ref[key] = {k: v for k, v in r.items() if not k.startswith("prefix.")}
    return ref[key]

lower = [f"final.{il}.{n}" for il in range(SPLIT) for n in ("conv", "rec", "k", "v") if f"final.{il}.{n}" in reference("speed", 0, None)]

for b, ub in ((700, 512), (128, 128), (50, 50)):
    tag = f"b{b}-ub{ub}"
    for mode, alpha, rows in (("speed", 0, None), ("speed", 1, None), ("quality", 0, R1)):
        name = f"{mode}-a{alpha}-{tag}"
        ps, rc = run("sweep-" + name, b, ub, mode, alpha, rows, extra=["--sweep"])
        p1, rc1 = run("single-" + name, b, ub, mode, alpha, rows)
        if rc or rc1:
            record(f"oracle-sweep-{name}", False, reason=f"rc sweep {rc} single {rc1}"); continue
        A, S, E = load(ps), load(p1), reference(mode, alpha, rows)
        record(f"oracle-sweep-{name}", rel(A, E) <= 1e-4, max_rel=rel(A, E), cells=N)
        record(f"sweep==single-{name}", rel(A, S) <= 1e-4, max_rel=rel(A, S), single_vs_oracle=rel(S, E))
        same = all(np.array_equal(A[k], S[k]) for k in lower)
        record(f"lower-caches-byte-equal-{name}", same, arrays=len(lower))
    pw, rc = run("sweep-wrong-" + tag, b, ub, "speed", 0, None, proj="projector-s3-wrong-f32.gguf", extra=["--sweep"])
    d = rel(load(pw), reference("speed", 0, None)) if rc == 0 else -1
    record(f"positive-wrong-map-sweep-{tag}", d > 1e-2, max_rel=d)
    pn, rc = run("sweep-tgen-" + tag, b, ub, "speed", 0, None, tail=N, extra=["--sweep"])
    ps0, rc0 = run("stock-" + tag, b, ub, "off", tail=N)
    d = rel(load(pn), load(ps0)) if rc == 0 and rc0 == 0 else -1
    record(f"negative-sweep-T>=N-vs-stock-{tag}", 0 <= d <= 1e-5, max_rel=d)

# stale states: the sequence served another prompt first (removed without zeroing the buffers); the replay must
# start the late layers' recurrent states from zero (speed: --warm re-sets p_end, which clears exact rows)
for alpha in (0, 1):
    pwm, rc = run(f"sweep-warm-a{alpha}", 128, 128, "speed", alpha, extra=["--sweep", "--warm"])
    d = rel(load(pwm), reference("speed", alpha, None)) if rc == 0 else -1
    record(f"oracle-sweep-after-stale-prompt-a{alpha}", 0 <= d <= 1e-4, max_rel=d)

# the zeroing itself: after the first replay batch (2 rows, before decay hides it) the late layers' states and the
# replayed K/V rows equal a cold context's byte for byte (with the zeroing disabled: 0.59 apart, NOTES.md)
pc, rc1 = run("late1-cold", 2, 2, "speed", extra=["--sweep"])
pw2, rc2 = run("late1-warm", 2, 2, "speed", extra=["--sweep", "--warm"])
if rc1 == 0 and rc2 == 0:
    r = subprocess.run([sys.executable, str(HERE / "cmp_late1.py"), str(W), str(pc) + ".late1.kv", str(pw2) + ".late1.kv", "2"],
                       capture_output=True, text=True)
    d = json.loads(r.stdout.strip().splitlines()[-1])
    record("replay-zeroes-stale-late-states", d["equal"] and d["arrays"] > 0, **d)
else:
    record("replay-zeroes-stale-late-states", False, reason=f"rc {rc1} {rc2}")

# captures: h_S rows and window layer inputs vs upstream's llama_get_embeddings_layer_inp of the same rows
ps, rc = run("cap-sweep", 700, 512, "speed", extra=["--sweep", "--win-layers", "1,2", "--win-n", "64"])
pi, rc2 = run("cap-stock", 700, 512, "off", tail=N, extra=["--layer-inp", "1,2,3"])
if rc == 0 and rc2 == 0:
    h = np.fromfile(str(ps) + ".h.f32", dtype=np.float32).reshape(N, NE)
    i3 = np.fromfile(str(pi) + ".inp3.f32", dtype=np.float32).reshape(N, NE)   # every row, batch order = position
    d = float(np.max(np.abs(h - i3)))
    record("capture-h_S-vs-layer_inp(3)", d == 0.0 and float(np.max(np.abs(h))) > 0, max_abs=d, rows=N)
    for l in (1, 2):
        w = np.fromfile(str(ps) + f".win{l}.f32", dtype=np.float32).reshape(64, NE)
        il = np.fromfile(str(pi) + f".inp{l}.f32", dtype=np.float32).reshape(N, NE)
        d = float(np.max(np.abs(w - il[N - 64:])))
        record(f"capture-window-layer{l}-vs-layer_inp({l})", d == 0.0, max_abs=d, rows=64)
else:
    record("captures", False, reason=f"rc {rc} {rc2}")

# refusals
pr, rc = run("refuse-lower-output", 128, 128, "speed", extra=["--sweep", "--filled-logit"])
record("refuse-lower-output", rc == 2 and "cannot be an output (no output head)" in Path(str(pr) + ".log").read_text(), rc=rc)
pr, rc = run("refuse-late-unseen", 128, 128, "speed", extra=["--late-only"])
record("refuse-late-without-lower", rc == 3 and "LATE follows a LOWER pass" in Path(str(pr) + ".log").read_text(), rc=rc)
pr, rc = run("refuse-lower-nonempty", 128, 128, "speed", extra=["--sweep", "--lower-twice"])
record("refuse-lower-on-nonempty-sequence", rc == 3 and "LOWER starts on an empty sequence" in Path(str(pr) + ".log").read_text(), rc=rc)
pr, rc = run("refuse-full-midsweep", 128, 128, "speed", extra=["--sweep", "--full-midsweep"])
record("refuse-full-reset-mid-sweep", rc == 3 and "before FULL" in Path(str(pr) + ".log").read_text(), rc=rc)
F = Path("/var/mnt/qwen-storage/dylan/AI-Work/llamacpp-ridgefill-20261008/gate1/work")
rc = subprocess.run([str(DUMP), "--model", str(F / "tiny-qwen3-f32.gguf"), "--tokens", str(F / "tokens.bin"), "--output",
                     str(OUT / "refuse-dense"), "--mode", "speed", "--tail", "200", "--keep", "200", "--projector",
                     str(F / "qproj-s2-f32.gguf"), "--sweep"], capture_output=True, text=True)
record("refuse-part-dense-model", rc.returncode == 3 and "qwen35 (hybrid memory) only" in rc.stderr, rc=rc.returncode)

summary = dict(gate="G1-sweep-dispatch", result=("PASS" if all(r["result"] == "PASS" for r in records) else "FAIL") if records else "SKIPPED",
               checks=len(records), passed=sum(r["result"] == "PASS" for r in records))
(HERE / "results-g1-sweep-dispatch.json").write_text(json.dumps(dict(summary=summary, checks=records), indent=2) + "\n")
print(json.dumps(summary))
sys.exit(0 if summary["result"] == "PASS" else 1)
