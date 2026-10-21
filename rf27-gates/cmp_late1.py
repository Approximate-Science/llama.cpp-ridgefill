"""Warm vs cold state after the first LATE (replay) batch of `rows` positions: the late layers' GDN conv/state and the
late attention K/V rows < rows must be byte-equal (the replay starts the late recurrent states from zero); the K/V
rows >= rows are not replayed yet and hold whatever the cells held. Usage: cmp_late1.py <work> <cold> <warm> <rows>"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent / "lb/tests/ridgefill"))
import compare_qwen35 as cq
W, a, b, rows = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
cfg = json.loads((W / "tiny-qwen35-hf/config.json").read_text())
A, B = cq.read_state(Path(a), cfg, 700), cq.read_state(Path(b), cfg, 700)
split = 3
diff = {}
for k in A:
    il, name = int(k.split(".")[0]), k.split(".")[1]
    if il < split:
        continue
    x, y = (A[k][:, :rows], B[k][:, :rows]) if name in ("k", "v") else (A[k], B[k])
    diff[k] = float(np.max(np.abs(x - y)))
print(json.dumps(dict(equal=all(v == 0.0 for v in diff.values()), max_abs=max(diff.values()), arrays=len(diff))))
