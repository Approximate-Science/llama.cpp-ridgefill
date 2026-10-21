#!/usr/bin/env python3
"""G1 for the S1 window selector and vtail (C API v2) on the tiny qwen35 (L-B's fixture, projector split 4), vs lane T's
kva references (q27/engine_ref_s1_vtail.py; see the reference directory's MANIFEST.json). Usage:
  g1_s1_vtail.py REF_DIR      (REF_DIR: <prompt>-<arm>.npz, MANIFEST.json, SHA256SUMS; machine paths are arguments)
Per prompt (fixture, selchat) and arm, the engine's two-sweep flow through rf_dump_rs:
  T3s             LOWER -> LATE with the reference's T3s rows (the arm vtail is measured against)
  T3s-vtail-k{0,72}  + vtail with c = T - k rows (k0: 200, k72: 128)
  S1gw            LOWER -> S1_FILL [0, P) -> S1_PROBE [N - 64, N) -> rows = top n_rows_ref by the engine's mass,
                  united with the special rows -> LATE
  S1gw-vtail-k{0,72}
Checks: logits, final.* (every layer) and prefix.* (late layers, cells [0, P), before vtail) <= 1e-4 relative to the
reference's max |value|; S1 mass <= 1e-4; S1 rows equal (or a tie at the k-th mass, reported); vtail scales <= 1e-4.
Controls: vtail changes the late V and the logits (vs T3s); k0 and k72 give different scales; S1 rows differ from the
T3s rows. Refusals by name: blind selection (window start in the prefix), a window longer than the tail, LATE after
S1_FILL without a probe, vtail with a first tail ubatch shorter than c, vtail on a single-pass tail."""
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
if len(sys.argv) < 2:
    sys.exit(__doc__)
REF = Path(sys.argv[1])
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "lb/tests/ridgefill")]
import compare_qwen35 as cq

W = HERE / "lb/out/gate1-qwen35"
MODEL, PROJ = W / "tiny-qwen35-f32.gguf", W / "projector-s4-f32.gguf"
TOKENS = dict(fixture=W / "tokens.bin", selchat=HERE / "out/selchat/fixture/tokens.bin")
OUT = HERE / "out/s1"; OUT.mkdir(parents=True, exist_ok=True)
CONFIG = json.loads((W / "tiny-qwen35-hf/config.json").read_text())
SPLIT, P, N, TOL = 4, 500, 700, 1e-4
manifest = json.loads((REF / "MANIFEST.json").read_text())

records = []
def record(name, ok, **kw):
    e = dict(check=name, result="PASS" if ok else "FAIL", **kw); records.append(e); print(json.dumps(e), flush=True)

# the references are the files lane T published, and the inputs are the ones it used
sums = dict(line.split()[::-1] for line in (REF / "SHA256SUMS").read_text().splitlines() if line.strip())
bad = [f for f, h in sums.items() if hashlib.sha256((REF / f).read_bytes()).hexdigest() != h]
record("reference-sha256sums", not bad and len(sums) >= 12, files=len(sums), mismatched=bad)
inp = manifest["inputs"]
for key, path in (("tiny-qwen35-f32.gguf", MODEL), ("tokens:fixture", TOKENS["fixture"]), ("tokens:selchat", TOKENS["selchat"])):
    record(f"input-{key}", hashlib.sha256(path.read_bytes()).hexdigest() == inp[key])

def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-30))

def run(prompt, tag, *extra, ubatch=512):
    pfx = OUT / f"{prompt}-{tag}"
    r = subprocess.run([str(HERE / "rf_dump_rs"), "--model", str(MODEL), "--tokens", str(TOKENS[prompt]), "--output", str(pfx),
                        "--mode", "quality", "--tail", "200", "--keep", "200", "--ubatch", str(ubatch), "--projector", str(PROJ),
                        "--sweep", "--sweep-boundary", *map(str, extra)], capture_output=True, text=True)
    (OUT / f"{prompt}-{tag}.log").write_text(r.stdout + r.stderr)
    return pfx, r

def engine_arrays(pfx):
    a = {"logits": np.fromfile(str(pfx) + ".logits", dtype=np.float32).reshape(200, 300)}
    for k, v in cq.read_state(Path(str(pfx) + ".kv"), CONFIG, N).items():
        a["final." + k] = v
    # the LATE replay's state at P: every LOWER cell exists, the late caches of [0, P) are the fill's
    for k, v in cq.read_state(Path(str(pfx) + ".prefix.kv"), CONFIG, N).items():
        if int(k.split(".")[0]) >= SPLIT:
            a["prefix." + k] = v[:, :P] if k.endswith((".k", ".v")) else v
    return a

state_keys = lambda ref: [k for k in ref.files if k == "logits" or k.startswith(("final.", "prefix."))]
eng = {}
for prompt in ("fixture", "selchat"):
    for arm in ("T3s", "T3s-vtail-k0", "T3s-vtail-k72", "S1gw", "S1gw-vtail-k0", "S1gw-vtail-k72"):
        ref = np.load(REF / f"{prompt}-{arm}.npz")
        extra = []
        if arm.startswith("T3s"):
            extra += ["--rows", ",".join(map(str, sorted(ref["rows"].tolist())))]
        else:
            extra += ["--s1-window", manifest["window"], "--s1-k", int(ref["n_rows_ref"]), "--s1-special", 1]
        if "vtail" in arm:
            extra += ["--vtail", 200 - int(arm.rsplit("-k", 1)[1])]
        pfx, r = run(prompt, arm, *extra)
        name = f"{prompt}-{arm}"
        if r.returncode != 0:
            record(name + "-ran", False, rc=r.returncode, log=(r.stdout + r.stderr)[-300:]); continue
        a = engine_arrays(pfx); eng[name] = dict(a=a)
        keys = state_keys(ref)
        missing = sorted(set(keys) ^ set(a))
        errs = {k: rel(a[k], ref[k]) for k in keys if k in a and a[k].shape == ref[k].shape}
        shape_bad = [k for k in keys if k in a and a[k].shape != ref[k].shape]
        worst = max(errs, key=errs.get) if errs else None
        record(name + "-state-logits", not missing and not shape_bad and len(errs) == len(keys) and max(errs.values()) <= TOL,
               arrays=len(errs), max_rel=errs.get(worst), worst=worst, logits_rel=errs.get("logits"), missing=missing, shape=shape_bad)
        if "vtail" in arm:
            sc = np.fromfile(str(pfx) + ".vtail.f32", dtype=np.float32)
            want = ref["vtail_scales"].ravel()
            ok = sc.shape == want.shape and rel(sc.astype(np.float64), want) <= TOL
            record(name + "-vtail-scales", ok, engine=sc.tolist(), reference=want.tolist())
            eng[name]["scales"] = sc
        if arm.startswith("S1gw"):
            m = np.fromfile(str(pfx) + ".mass_s1.f64")
            record(name + "-s1-mass", m.shape == ref["mass_s1"].shape and rel(m, ref["mass_s1"]) <= TOL,
                   max_rel=rel(m, ref["mass_s1"]) if m.shape == ref["mass_s1"].shape else None, n=int(m.size))
            rows = np.fromfile(str(pfx) + ".rows.i32", dtype=np.int32)
            want = np.sort(ref["rows"])
            same = np.array_equal(rows, want)
            # a tie at the cut: the reference mass of the k-th and (k+1)-th row by rank
            srt = np.sort(ref["mass_s1"])[::-1]; k = int(ref["n_rows_ref"])
            gap = float(srt[k - 1] - srt[k]) if k < P else None
            record(name + "-s1-rows", same, n_engine=int(rows.size), n_reference=int(want.size), cut_gap=gap,
                   diff=sorted(set(rows.tolist()) ^ set(want.tolist()))[:8])
            eng[name]["rows"] = rows

# chunking: S1 with n_ubatch 32 (the probe window in 2 ubatches: mass summed across them, the window's GDN state carried),
# and vtail with c = n_ubatch = 128 (the first tail ubatch holds exactly c rows, the second one sees the scaled V)
for prompt, arm, extra, ub in (("selchat", "S1gw", ["--s1-window", 64, "--s1-k", 220, "--s1-special", 1], 32),
                               ("fixture", "T3s-vtail-k72", ["--vtail", 128], 128)):
    ref = np.load(REF / f"{prompt}-{arm}.npz")
    if arm.startswith("T3s"):
        extra = ["--rows", ",".join(map(str, sorted(ref["rows"].tolist())))] + extra
    pfx, r = run(prompt, f"{arm}-ub{ub}", *extra, ubatch=ub)
    if r.returncode != 0:
        record(f"{prompt}-{arm}-ubatch{ub}-ran", False, rc=r.returncode); continue
    a = engine_arrays(pfx); keys = state_keys(ref)
    errs = {k: rel(a[k], ref[k]) for k in keys}
    extra_ok, info = True, {}
    if "vtail" in arm:
        sc = np.fromfile(str(pfx) + ".vtail.f32", dtype=np.float32)
        info["scales_rel"] = rel(sc.astype(np.float64), ref["vtail_scales"].ravel())
        extra_ok = info["scales_rel"] <= TOL
    if arm.startswith("S1gw"):
        m = np.fromfile(str(pfx) + ".mass_s1.f64"); rows = np.fromfile(str(pfx) + ".rows.i32", dtype=np.int32)
        info.update(mass_rel=rel(m, ref["mass_s1"]), rows_equal=bool(np.array_equal(rows, np.sort(ref["rows"]))))
        extra_ok = extra_ok and info["mass_rel"] <= TOL and info["rows_equal"]
    record(f"{prompt}-{arm}-ubatch{ub}", max(errs.values()) <= TOL and extra_ok,
           max_rel=max(errs.values()), **info)

# controls: vtail has an effect; the two c values differ; S1 picks other rows than T3s
for prompt in ("fixture", "selchat"):
    base, vt = eng.get(f"{prompt}-T3s"), eng.get(f"{prompt}-T3s-vtail-k0")
    if base and vt:
        record(f"control-{prompt}-vtail-changes-late-v-and-logits", rel(vt["a"]["final.7.v"], base["a"]["final.7.v"]) > 10 * TOL
               and rel(vt["a"]["logits"], base["a"]["logits"]) > 10 * TOL, v_rel=rel(vt["a"]["final.7.v"], base["a"]["final.7.v"]),
               logits_rel=rel(vt["a"]["logits"], base["a"]["logits"]))
    k0, k72 = eng.get(f"{prompt}-T3s-vtail-k0"), eng.get(f"{prompt}-T3s-vtail-k72")
    if k0 and k72:
        record(f"control-{prompt}-c-changes-scales", float(np.max(np.abs(k0["scales"] - k72["scales"]))) > 10 * TOL)
    s1 = eng.get(f"{prompt}-S1gw")
    if s1:
        t3 = np.load(REF / f"{prompt}-T3s.npz")["rows"]
        record(f"control-{prompt}-s1-rows-differ-from-t3s", not np.array_equal(s1["rows"], np.sort(t3)),
               shared=len(set(s1["rows"].tolist()) & set(t3.tolist())))

# refusals by name (the run must fail and the log must name the reason)
def refusal(name, needle, *extra, ubatch=512, sweep=True):
    pfx = OUT / f"refuse-{name}"
    cmd = [str(HERE / "rf_dump_rs"), "--model", str(MODEL), "--tokens", str(TOKENS["fixture"]), "--output", str(pfx), "--mode",
           "quality", "--tail", "200", "--keep", "200", "--ubatch", str(ubatch), "--projector", str(PROJ), *map(str, extra)]
    if sweep:
        cmd += ["--sweep", "--sweep-boundary"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    text = r.stdout + r.stderr
    (OUT / f"refuse-{name}.log").write_text(text)
    record(f"refuses-{name}", r.returncode not in (0,) and needle in text, rc=r.returncode, needle=needle)

refusal("s1-blind-window-in-prefix", "blind selection", "--s1-window", 64, "--s1-k", 10, "--s1-begin", 450)
refusal("s1-window-longer-than-tail", "blind selection", "--s1-window", 300, "--s1-k", 10)
refusal("late-after-s1-fill-without-probe", "LATE follows a LOWER pass", "--s1-window", 64, "--s1-k", 10, "--s1-late-early")
refusal("vtail-first-tail-ubatch-short", "vtail needs the first tail ubatch", "--vtail", 200, ubatch=128)
refusal("vtail-single-pass", "vtail requires the two-sweep flow", "--vtail", 200, "--batch", 128, sweep=False)

s = dict(gate="G1-s1-vtail", reference=str(REF), checks=len(records), passed=sum(r["result"] == "PASS" for r in records))
s["result"] = ("PASS" if s["passed"] == s["checks"] else "FAIL") if records else "SKIPPED"
(HERE / "results-g1-s1-vtail.json").write_text(json.dumps(dict(summary=s, checks=records), indent=2) + "\n")
print(json.dumps(s)); sys.exit(0 if s["result"] == "PASS" else 1)
