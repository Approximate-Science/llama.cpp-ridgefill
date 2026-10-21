#!/usr/bin/env python3
"""llama_ridgefill_select_rows vs the oracle (lane T's q27/rf27_quality.py class_rows + kva.select.digits with
digit_ids from the tiny model's HF tokenizer), tiny qwen35 + projector-s3 (L-B's fixture): shares 0, 0.1, 0.25, 0.5,
1 x digits on/off x prefix 500 and 699 (tail 200 / 1). Controls: the fixture must have class matches and digit
tokens in the prefix (else SKIPPED), share 0.5 != share 0.25, digits change the set; a format-1 projector is refused."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
KVA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/var/home/dylan/projects/research/kva-wt-q27")
sys.path.insert(0, str(KVA)); sys.path.insert(0, str(KVA / "q27"))
from safetensors.numpy import load_file
from transformers import AutoTokenizer
from kva import select as SEL
import rf27_quality as Q

W = HERE / "lb/out/gate1-qwen35"
OUT = HERE / "out/sel"; OUT.mkdir(parents=True, exist_ok=True)
ids = np.fromfile(W / "tokens.bin", dtype=np.int32).tolist()
score = load_file(str(W / "projector-s3/rowsel.safetensors"))["score"]
digit_set = SEL.digit_ids(AutoTokenizer.from_pretrained(str(W / "tiny-qwen35-hf")))
records = []
def record(name, ok, **kw):
    e = dict(check=name, result="PASS" if ok else "FAIL", **kw); records.append(e); print(json.dumps(e), flush=True)

def engine(share, digits, tail, proj=W / "projector-s3-f32.gguf", model=W / "tiny-qwen35-f32.gguf", tokens=W / "tokens.bin"):
    pfx = OUT / f"s{share}-d{digits}-t{tail}"
    r = subprocess.run([str(HERE / "rf_dump_rs"), "--model", str(model), "--tokens", str(tokens), "--output", str(pfx),
                        "--mode", "speed", "--tail", str(tail), "--keep", "1", "--projector", str(proj),
                        "--select-share", str(share), "--select-digits", str(digits)], capture_output=True, text=True)
    return (np.fromfile(str(pfx) + ".rows.i32", dtype=np.int32).tolist() if r.returncode == 0 else None), r

n_match = int(np.isfinite(score[np.asarray(ids[:500])]).sum())
n_digit = len(SEL.digits(ids, 500, digit_set))
if n_match == 0 or n_digit == 0:
    print(json.dumps(dict(gate="G1-select-rows", result="SKIPPED", reason=f"fixture has {n_match} class matches, {n_digit} digit rows")))
    sys.exit(1)
got = {}
for tail in (200, 1):   # the tool needs tail >= 1
    P = 700 - tail
    for share in (0, 0.1, 0.25, 0.5, 1):
        for digits in (0, 1):
            want = set(Q.class_rows(ids, P, score, share))
            if digits:
                want |= set(SEL.digits(ids, P, digit_set))
            rows, r = engine(share, digits, tail)
            got[(share, digits, tail)] = rows
            record(f"select-rows-share{share}-digits{digits}-P{P}", rows == sorted(want), n=len(want),
                   first_diff=None if rows == sorted(want) else str(sorted(set(rows or []) ^ want)[:5]))
record("control-share-0.5!=0.25", got[(0.5, 0, 200)] != got[(0.25, 0, 200)], n=[len(got[(0.5, 0, 200)]), len(got[(0.25, 0, 200)])])
record("control-digits-change-set", got[(0.5, 1, 200)] != got[(0.5, 0, 200)])
F = Path("/var/mnt/qwen-storage/dylan/AI-Work/llamacpp-ridgefill-20261008/gate1/work")
rows, r = engine(0.5, 1, 200, proj=F / "qproj-s2-f32.gguf", model=F / "tiny-qwen3-f32.gguf", tokens=F / "tokens.bin")
record("refuse-format1-projector", rows is None and r.returncode == 3 and "no row table" in r.stderr, rc=r.returncode)
summary = dict(gate="G1-select-rows", result="PASS" if all(x["result"] == "PASS" for x in records) else "FAIL",
               checks=len(records), passed=sum(x["result"] == "PASS" for x in records), class_matches=n_match, digit_rows=n_digit)
(HERE / "results-g1-select-rows.json").write_text(json.dumps(dict(summary=summary, checks=records), indent=2) + "\n")
print(json.dumps(summary))
sys.exit(0 if summary["result"] == "PASS" else 1)
