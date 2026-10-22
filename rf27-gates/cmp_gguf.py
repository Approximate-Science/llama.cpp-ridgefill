#!/usr/bin/env python3
"""Compare two GGUF files: every tensor (name, type, shape, raw bytes) and every metadata key (value); lists the keys
that differ. Usage: cmp_gguf.py A.gguf B.gguf"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
from gguf import GGUFReader

def load(p):
    r = GGUFReader(p)
    kv = {}
    for f in r.fields.values():
        if f.name.startswith("GGUF."):
            continue
        kv[f.name] = [bytes(f.parts[i]).hex() for i in f.data]
    t = {x.name: (int(x.tensor_type), [int(d) for d in x.shape], bytes(x.data.tobytes())) for x in r.tensors}
    return kv, t

ka, ta = load(sys.argv[1]); kb, tb = load(sys.argv[2])
tdiff = [n for n in sorted(set(ta) | set(tb)) if ta.get(n) != tb.get(n)]
kdiff = sorted(k for k in set(ka) | set(kb) if ka.get(k) != kb.get(k))
print(json.dumps(dict(tensors=len(ta), tensors_differing=tdiff, metadata_keys=len(ka), metadata_keys_differing=kdiff), indent=1))
