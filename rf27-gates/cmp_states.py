"""Max relative error between two rf_dump outputs (logits, final state, prefix state), parsed with L-B's
compare_qwen35.read_state. Usage: cmp_states.py <work> <prefixA> <prefixB> <n> <prefix_len> <keep>.
Prints one JSON line. A missing or empty file raises (no result is not a pass)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent / "lb/tests/ridgefill"))
import compare_qwen35 as c
work, a, b, n, p, keep = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
cfg = json.loads((work / "tiny-qwen35-hf/config.json").read_text())
def load(pfx):
    d = {"logits": np.fromfile(pfx + ".logits", dtype=np.float32)}
    for k, v in c.read_state(Path(pfx + ".kv"), cfg, n).items(): d["final." + k] = v
    if p > 0:
        for k, v in c.read_state(Path(pfx + ".prefix.kv"), cfg, p).items(): d["prefix." + k] = v
    return d
A, B = load(a), load(b)
assert set(A) == set(B) and A["logits"].size == keep * 300
err = {k: float(np.max(np.abs(A[k] - B[k])) / max(float(np.max(np.abs(B[k]))), 1e-30)) for k in A}
worst = max(err, key=err.get)
print(json.dumps(dict(max_rel=err[worst], worst=worst, arrays=len(err))))
