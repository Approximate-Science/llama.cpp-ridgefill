"""Check fixture/oracle preparation controls and export split-4 question mass; this is not engine G1."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--kva", type=Path, required=True)
parser.add_argument("--work", type=Path, default=root / "out/gate1-qwen35")
args = parser.parse_args()
work = args.work.resolve()
sys.path.insert(0, str(root / "tests/ridgefill"))
sys.path.insert(0, str(args.kva.resolve()))
from compare_qwen35 import reference
from kva.model import Qwen35
from kva.kva import run_lower
from kva.select import digit_ids, qattn_layers, top_rows, window_mass
import convert_ridgefill_to_gguf as converter

torch.set_num_threads(4)
zero = reference(args.kva, work, 3, "speed", 0, [])
one = reference(args.kva, work, 3, "speed", 1, [])
_, correction, _ = converter.read_safetensors(work / "projector-s3/correction.safetensors")
orientation_errors = []
for name, st in correction.items():
    layer = int(name.split(".")[1])
    delta = one[f"prefix.{layer}.rec"] - zero[f"prefix.{layer}.rec"]
    np.testing.assert_allclose(delta, st.swapaxes(-1, -2), atol=1e-6, rtol=1e-5)
    orientation_errors.append(float(np.max(np.abs(delta - st))))
assert max(orientation_errors) > 0.01

model = Qwen35(work / "tiny-qwen35-hf", device="cpu", dtype=torch.float32)
tokens = np.fromfile(work / "tokens.bin", dtype=np.int32)
ids = torch.tensor(tokens, dtype=torch.long)
inputs = {}
with torch.no_grad():
    hs, cache = run_lower(model, ids, [4], hook=lambda layer, hidden: inputs.update({layer: hidden.clone()}))
    assert qattn_layers(model, 3, [3, 7]) == []
    assert qattn_layers(model, 4, [3, 7]) == [3]
    mass = window_mass(model, {3: inputs[3][-64:]}, cache, [3])
    explicit = window_mass(model, {3: inputs[3][-64:]}, cache, [3], end=700)
    blind = window_mass(model, {3: inputs[3][436:500]}, cache, [3], end=500)
    torch.testing.assert_close(mass, explicit, rtol=0, atol=0)
    assert abs(float(mass.sum()) - 256) < 1e-4
    assert float((mass - blind).abs().max()) > 1e-4
    rows = top_rows(mass, 500, 100)
    assert len(rows) == 100
    np.savez(work / "qattn-s4-oracle.npz", tokens=tokens, layer3_input=inputs[3][-64:].numpy(),
             keys=cache[3]["k"].numpy(), h_s=hs[4].numpy(), mass=mass.numpy(), rows=np.array(rows))
tokenizer = AutoTokenizer.from_pretrained(work / "tiny-qwen35-hf", local_files_only=True)
digits = digit_ids(tokenizer)
assert 269 in digits and 259 not in digits
print(json.dumps(dict(result="PASS", scope="fixture/oracle preparation only; engine G1 not run",
                      split3_alpha1_arrays=len(one), correction_orientation_control=max(orientation_errors),
                      qattn_layers=[3], qattn_mass_sum=float(mass.sum()), qattn_rows=len(rows),
                      question_window_control=float((mass - blind).abs().max()), digit_token=269)))
