#!/usr/bin/env python3
"""G1 for the chat-template row rules (CHAT-PLAN H1-H3) on the tiny qwen35 (L-B's fixture), vs independent
expectations: the oracle's class_rows (lane T's q27/rf27_quality.py) + kva select.digits, and the rule
  special(id)  <=>  id in the HF tokenizer's added_tokens_decoder
  special_exact: every prefix row whose token is special;  header_rows K: rows i+1 .. i+K (< P) after a special at i
computed from the HF tokenizer, never from the GGUF. A crafted prompt (the fixture's tokens with added tokens
inserted: adjacent pairs, one as the last prefix row, one in the tail) exercises the edges. Then the masked fill with
the selected rows vs kva finish(exact_rows=R) on that prompt, and llama_ridgefill_last_marker vs a Python search.
Controls: on the fixture's own prompt (no added tokens) special_exact / header_rows change nothing; on the crafted
prompt they do. Usage: g1_select_chat.py KVA_REPO   (env ORACLE fixtures as below)."""
import json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
if len(sys.argv) < 2:
    sys.exit(f"usage: {sys.argv[0]} KVA_REPO   (machine-local paths are arguments, never defaults)")
KVA = Path(sys.argv[1])
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(KVA), str(KVA / "q27"), str(HERE / "lb/tests/ridgefill")]
from safetensors.numpy import load_file
from transformers import AutoTokenizer
from kva import select as SEL
import rf27_quality as Q
import compare_qwen35 as cq

W = HERE / "lb/out/gate1-qwen35"
OUT = HERE / "out/selchat"; OUT.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(str(W / "tiny-qwen35-hf"))
ADDED = set(tok.added_tokens_decoder)
score = load_file(str(W / "projector-s3/rowsel.safetensors"))["score"]
digit_set = SEL.digit_ids(tok)
base = np.fromfile(W / "tokens.bin", dtype=np.int32).tolist()

# crafted prompt: 700 tokens, P = 500; added tokens at 0, 10-11 (adjacent), 40, 41 (adjacent, both special), 300,
# 497, 499 (the last prefix row: its header rows fall in the tail), 600 (tail)
crafted = list(base)
for pos, tid in ((0, 1), (10, 2), (11, 1), (40, 0), (41, 2), (300, 1), (497, 2), (499, 1), (600, 2)):
    crafted[pos] = tid
cw = OUT / "fixture"; cw.mkdir(exist_ok=True)
for name in ("tiny-qwen35-f32.gguf", "projector-s3-f32.gguf", "projector-s3-wrong-f32.gguf", "tiny-qwen35-hf", "projector-s3"):
    dst = cw / name
    if not dst.exists():
        os.symlink(W / name, dst)
np.asarray(crafted, dtype=np.int32).tofile(cw / "tokens.bin")

records = []
def record(name, ok, **kw):
    e = dict(check=name, result="PASS" if ok else "FAIL", **kw); records.append(e); print(json.dumps(e), flush=True)

def expected(ids, P, share, digits, special, header):
    rows = set(Q.class_rows(ids, P, score, share))
    if digits:
        rows |= set(SEL.digits(ids, P, digit_set))
    for i in range(P):
        if ids[i] in ADDED:
            if special:
                rows.add(i)
            rows |= {j for j in range(i + 1, min(i + header, P - 1) + 1)}
    return sorted(rows)

def engine(tokens_path, share, digits, special, header, tail=200):
    pfx = OUT / f"sel-{tokens_path.parent.name}-{share}-{digits}-{special}-{header}"
    r = subprocess.run([str(HERE / "rf_dump_rs"), "--model", str(W / "tiny-qwen35-f32.gguf"), "--tokens", str(tokens_path),
                        "--output", str(pfx), "--mode", "speed", "--tail", str(tail), "--keep", "1", "--projector",
                        str(W / "projector-s3-f32.gguf"), "--select-share", str(share), "--select-digits", str(digits),
                        "--select-special", str(special), "--select-header", str(header)], capture_output=True, text=True)
    return np.fromfile(str(pfx) + ".rows.i32", dtype=np.int32).tolist() if r.returncode == 0 else None

n_added = sum(1 for t in crafted[:500] if t in ADDED)
if n_added == 0 or any(t in ADDED for t in base[:500]):
    sys.exit(json.dumps(dict(gate="G1-select-chat", result="SKIPPED", reason="crafted prompt has no added tokens, or the base prompt has some")))

got = {}
for share in (0, 0.5):
    for digits in (0, 1):
        for special in (0, 1):
            for header in (0, 1, 2, 3):
                want = expected(crafted, 500, share, digits, special, header)
                rows = engine(cw / "tokens.bin", share, digits, special, header)
                got[(share, digits, special, header)] = rows
                record(f"crafted-share{share}-digits{digits}-special{special}-header{header}", rows == want, n=len(want),
                       diff=None if rows == want else sorted(set(rows or []) ^ set(want))[:6])
# controls
record("control-crafted-special-changes-set", got[(0.5, 1, 1, 0)] != got[(0.5, 1, 0, 0)])
record("control-crafted-header-changes-set", got[(0.5, 1, 1, 2)] != got[(0.5, 1, 1, 0)])
plain_off = engine(W / "tokens.bin", 0.5, 1, 0, 0)
plain_on = engine(W / "tokens.bin", 0.5, 1, 1, 3)
record("control-no-added-tokens-rules-neutral", plain_off is not None and plain_off == plain_on, n=len(plain_off or []))
# the last prefix row is special: its header rows (500, 501, ...) are tail rows and must not be selected
record("edge-header-rows-stop-at-prefix-end", all(r < 500 for r in got[(0, 0, 1, 3)]) and 499 in got[(0, 0, 1, 3)])

# masked fill with the selected rows (special on, header 2, share 0.5, digits on) vs kva finish(exact_rows=R)
R = got[(0.5, 1, 1, 2)]
pfx = OUT / "fill-chat"
rc = subprocess.run([str(HERE / "rf_dump_rs"), "--model", str(cw / "tiny-qwen35-f32.gguf"), "--tokens", str(cw / "tokens.bin"),
                     "--output", str(pfx), "--mode", "quality", "--tail", "200", "--keep", "200", "--batch", "128", "--ubatch", "128",
                     "--projector", str(cw / "projector-s3-f32.gguf"), "--rows", ",".join(map(str, R))],
                    capture_output=True, text=True).returncode
if rc == 0:
    c = cq.compare(cw, pfx, cq.reference(KVA, cw, 3, "quality", 0, R), 3, False)
    record("oracle-fill-with-chat-rows", c["result"] == "PASS", max_rel=c["max_relative_error"], rows=len(R))
else:
    record("oracle-fill-with-chat-rows", False, rc=rc)

# last marker vs a Python search over the crafted prompt
def py_last(ids, m):
    for i in range(len(ids) - len(m), -1, -1):
        if ids[i:i + len(m)] == m:
            return i
    return -1
for m in ([2, 1], [1], [2, crafted[601]], [7, 7, 7, 7, 7], [crafted[-1]]):
    r = subprocess.run([str(HERE / "rf_dump_rs"), "--model", str(W / "tiny-qwen35-f32.gguf"), "--tokens", str(cw / "tokens.bin"),
                        "--output", str(OUT / "m"), "--mode", "speed", "--tail", "200", "--keep", "1", "--projector",
                        str(W / "projector-s3-f32.gguf"), "--last-marker", ",".join(map(str, m))], capture_output=True, text=True)
    val = int(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else None
    record(f"last-marker-{m}", val == py_last(crafted, m), engine=val, python=py_last(crafted, m))

s = dict(gate="G1-select-chat", checks=len(records), passed=sum(r["result"] == "PASS" for r in records), added_in_prefix=n_added)
s["result"] = "PASS" if s["passed"] == s["checks"] else "FAIL"
(HERE / "results-g1-select-chat.json").write_text(json.dumps(dict(summary=s, checks=records), indent=2) + "\n")
print(json.dumps(s)); sys.exit(0 if s["result"] == "PASS" else 1)
