#!/usr/bin/env python3
"""G1 quality at alpha 0 (tiny qwen35, split 3, N=700, T=200): engine vs kva finish(exact_rows=R), using L-B's
compare_qwen35.reference / compare unchanged. L-A's driver until L-B's harness handles quality (its T >= N control
passes --rows with no prefix, and it runs alpha 1, which the engine refuses until the rho-scaled correction exists).
Checks per chunk config: oracle agreement (1e-4); quality != speed (the rows R change the caches, else the gate tests
nothing); wrong map fails; T >= N byte-identical to stock. Also: a second R (adjacent rows, ubatch edges) agrees; the R1
run against the R2 oracle fails; FA on agrees at 1e-2; two sequences in the same ubatches (R1 + R2, R1 + speed)
agree per sequence; alpha 1 refused by name. Nothing run = SKIPPED, never PASS."""
import json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lb/tests/ridgefill"))
import compare_qwen35 as cq
import numpy as np

KVA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/var/home/dylan/projects/research/kva-wt-q27")
W = HERE / "lb/out/gate1-qwen35"
OUT = HERE / "out/g1q"; OUT.mkdir(parents=True, exist_ok=True)
DUMP = HERE / "rf_dump_q35"
records = []

def record(name, ok, **kw):
    e = dict(check=name, result="PASS" if ok else "FAIL", **kw); records.append(e); print(json.dumps(e), flush=True)

def invoke(tag, b, ub, mode, rows=None, alpha=0, proj="projector-s3-f32.gguf", tail=200):
    pfx = OUT / tag
    cmd = [str(DUMP), "--model", str(W / "tiny-qwen35-f32.gguf"), "--tokens", str(W / "tokens.bin"), "--output", str(pfx),
           "--mode", mode, "--tail", str(tail), "--keep", "200", "--batch", str(b), "--ubatch", str(ub), "--alpha", str(alpha)]
    if mode != "off":
        cmd += ["--projector", str(W / proj)]
    if rows:
        cmd += ["--rows", ",".join(map(str, rows))]
    with open(str(pfx) + ".log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    return pfx, rc

R1 = [0, 7, 31, 127, 255, 499]          # L-B's hand-picked rows (first, last prefix row, chunk-straddling)
R2 = [3, 64, 65, 66, 200, 383, 384, 450] # adjacent rows and rows at ubatch edges (128, 384, 512 boundaries)
def as_seq0(pfx, seq):
    """Copy of seq's dumps with the sequence id rewritten to 0 in the header and in every KV cell (L-B's parser reads
    seq 0 only); the payload bytes are untouched. Returns the new prefix."""
    import struct, shutil
    dst = Path(f"{pfx}.s{seq}as0")
    shutil.copyfile(f"{pfx}.s{seq}.logits", str(dst) + ".logits")
    for suf in (".kv", ".prefix.kv"):
        src = Path(f"{pfx}.s{seq}{suf}")
        if not src.exists():
            continue
        b = bytearray(src.read_bytes())
        magic, sid = struct.unpack_from("<Ii", b, 0)
        assert magic == 0xaf143cd8 and sid == seq
        struct.pack_into("<i", b, 4, 0)
        n_stream, cells = struct.unpack_from("<II", b, 8)
        assert n_stream == 1
        off = 16
        for _ in range(cells):
            pos, nseq = struct.unpack_from("<iI", b, off); off += 8
            off += 12                                    # x, y, token (M-RoPE cell metadata)
            for k in range(nseq):
                assert struct.unpack_from("<i", b, off)[0] == seq
                struct.pack_into("<i", b, off, 0); off += 4
        (Path(str(dst) + suf)).write_bytes(bytes(b))
    return dst

ref = {}
def reference(mode, rows):
    key = (mode, tuple(rows or ()))
    if key not in ref:
        ref[key] = cq.reference(KVA, W, 3, mode, 0, rows or [])
    return ref[key]

for b, ub in ((700, 512), (128, 128), (50, 50)):
    tag = f"b{b}-ub{ub}"
    for name, rows in (("R1", R1), ("R2", R2)):
        p, rc = invoke(f"q-{name}-{tag}", b, ub, "quality", rows)
        if rc:
            record(f"oracle-quality-{name}-{tag}", False, reason=f"rf_dump rc {rc}"); continue
        c = cq.compare(W, p, reference("quality", rows), 3, False)
        worst = max(c["relative_errors"], key=c["relative_errors"].get)
        record(f"oracle-quality-{name}-{tag}", c["result"] == "PASS", max_rel=c["max_relative_error"], worst=worst, arrays=c["arrays"])
    ps, _ = invoke(f"s-{tag}", b, ub, "speed")
    c = cq.compare(W, OUT / f"q-R1-{tag}", reference("speed", None), 3, False)
    record(f"quality!=speed-R1-{tag}", c["result"] == "FAIL" and c["max_relative_error"] > 1e-2, max_rel=c["max_relative_error"])
    pw, rc = invoke(f"wrong-{tag}", b, ub, "quality", R1, proj="projector-s3-wrong-f32.gguf")
    c = cq.compare(W, pw, reference("quality", R1), 3, False) if rc == 0 else dict(result="PASS", max_relative_error=0)
    record(f"positive-wrong-map-R1-{tag}", rc == 0 and c["result"] == "FAIL" and c["max_relative_error"] > 1e-2, max_rel=c["max_relative_error"])
    st, _ = invoke(f"stock-np-{tag}", b, ub, "off", tail=700)
    npf, rc = invoke(f"np-{tag}", b, ub, "quality", None, tail=700)
    same = rc == 0 and all(Path(str(st) + s).read_bytes() == Path(str(npf) + s).read_bytes() for s in (".kv", ".logits"))
    record(f"negative-tail-ge-n-{tag}", same)

# wrong rows: the engine's R1 run against the oracle's R2 must fail (the exact rows are the right ones)
c = cq.compare(W, OUT / "q-R1-b128-ub128", reference("quality", R2), 3, False)
record("positive-wrong-rows-R1-vs-R2", c["result"] == "FAIL" and c["max_relative_error"] > 1e-2, max_rel=c["max_relative_error"])
# flash attention on (the gathered mask rows are cast to F16): 1e-2, as the Oct-8 gate for FA on the CPU
for b, ub in ((700, 512), (128, 128)):
    pfx = OUT / f"q-R1-fa-b{b}-ub{ub}"
    rc = subprocess.run([str(HERE / "rf_dump_rs"), "--model", str(W / "tiny-qwen35-f32.gguf"), "--tokens", str(W / "tokens.bin"),
                         "--output", str(pfx), "--mode", "quality", "--tail", "200", "--keep", "200", "--batch", str(b),
                         "--ubatch", str(ub), "--projector", str(W / "projector-s3-f32.gguf"), "--rows", ",".join(map(str, R1)),
                         "--fa", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    c = cq.compare(W, pfx, reference("quality", R1), 3, False, tolerance=1e-2) if rc == 0 else dict(result="FAIL", max_relative_error=-1)
    record(f"oracle-quality-R1-fa1-b{b}-ub{ub}", c["result"] == "PASS", max_rel=c["max_relative_error"], tol=1e-2)

# two sequences in the same ubatches (unified KV): per-sequence rows; seq 1 with other rows, or in speed mode
for b, ub in ((700, 512), (128, 128)):
    for label, rows1, ref1 in (("R1+R2", R2, ("quality", R2)), ("R1+speed", None, ("speed", None))):
        pfx = OUT / f"ms-{label}-b{b}-ub{ub}"
        cmd = [str(HERE / "rf_dump_rs"), "--model", str(W / "tiny-qwen35-f32.gguf"), "--tokens", str(W / "tokens.bin"),
               "--output", str(pfx), "--mode", "quality", "--tail", "200", "--keep", "200", "--batch", str(b),
               "--ubatch", str(ub), "--projector", str(W / "projector-s3-f32.gguf"), "--rows", ",".join(map(str, R1)), "--n-seq", "2"]
        if rows1:
            cmd += ["--rows1", ",".join(map(str, rows1))]
        rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        for seq, rr in ((0, ("quality", R1)), (1, ref1)):
            c = cq.compare(W, as_seq0(pfx, seq), reference(*rr), 3, False) if rc == 0 else dict(result="FAIL", max_relative_error=-1)
            record(f"oracle-2seq-{label}-seq{seq}-b{b}-ub{ub}", c["result"] == "PASS", max_rel=c["max_relative_error"])

pa, rc = invoke("alpha1", 128, 128, "quality", R1, alpha=1)
msg = Path(str(pa) + ".log").read_text()
record("alpha1-quality-refused", rc == 2 and "rho-scaled correction" in msg, rc=rc)

summary = dict(gate="G1-quality-a0", result=("PASS" if all(r["result"] == "PASS" for r in records) else "FAIL") if records else "SKIPPED",
               checks=len(records), passed=sum(r["result"] == "PASS" for r in records))
(HERE / "results-g1-qwen35-quality-a0.json").write_text(json.dumps(dict(summary=summary, checks=records), indent=2) + "\n")
print(json.dumps(summary))
sys.exit(0 if summary["result"] == "PASS" else 1)
