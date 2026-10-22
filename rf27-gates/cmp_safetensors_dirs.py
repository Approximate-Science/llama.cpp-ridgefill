#!/usr/bin/env python3
"""Compare two projector package folders tensor by tensor: for every *.safetensors file, the tensor names, dtypes,
shapes and raw bytes, and the header metadata; for ridgefill.json, the parsed JSON (and which keys differ).
Usage: cmp_safetensors_dirs.py DIR_A DIR_B"""
import json, struct, sys
from pathlib import Path

def read(path):
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    meta = header.pop("__metadata__", {})
    data = raw[8 + n:]
    tensors = {k: (v["dtype"], v["shape"], data[v["data_offsets"][0]:v["data_offsets"][1]]) for k, v in header.items()}
    return meta, tensors, raw[8:8 + n]

a, b = Path(sys.argv[1]), Path(sys.argv[2])
names = sorted({p.name for p in a.glob("*.safetensors")} | {p.name for p in b.glob("*.safetensors")})
bad, header_only = [], []
for n in names:
    if not (a / n).exists() or not (b / n).exists():
        bad.append((n, "missing")); continue
    ma, ta, ha = read(a / n); mb, tb, hb = read(b / n)
    diff = [k for k in sorted(set(ta) | set(tb)) if ta.get(k) != tb.get(k)]
    if diff or ma != mb:
        bad.append((n, dict(tensors=diff[:5], metadata=None if ma == mb else {k: (ma.get(k), mb.get(k)) for k in set(ma) | set(mb) if ma.get(k) != mb.get(k)})))
    elif ha != hb:
        header_only.append(n)
ja, jb = json.loads((a / "ridgefill.json").read_text()), json.loads((b / "ridgefill.json").read_text())
def flat(d, p=""):
    return {p + k: v for k0, v0 in d.items() for k, v in (flat(v0, k0 + ".").items() if isinstance(v0, dict) else [(k0, v0)])}
fa, fb = flat(ja), flat(jb)
json_diff = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
print(json.dumps(dict(files=len(names), tensor_or_metadata_differences=bad, identical_tensors_header_bytes_differ=len(header_only),
                      ridgefill_json_keys_differing=json_diff[:40], n_json_keys_differing=len(json_diff)), indent=1))
