#!/usr/bin/env python3
"""Round-trip a format-1 package through GGUF, including wrong-map and head-order controls.

Q8_0 has a separate fp16 scale for each 32 weights. Compare against its rounding error bound,
not against the release's original bf16 fit. No model execution or torch is required here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import convert_ridgefill_to_gguf as converter

gguf = converter.gguf


def read_tensor(tensor) -> np.ndarray:
    return gguf.quants.dequantize(tensor.data, tensor.tensor_type).reshape(tuple(reversed(tensor.shape)))


def map_within_bound(actual: np.ndarray, expected: np.ndarray, qtype) -> tuple[bool, float]:
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        return False, float("inf")
    error = np.abs(actual - expected)
    if qtype == gguf.GGMLQuantizationType.Q8_0:
        blocks = expected.reshape(-1, 32)
        scale = np.abs(blocks).max(axis=1) / 127
        rounded = scale.astype(np.float16).astype(np.float32)
        bound = scale / 2 + 127 * np.abs(rounded - scale)
        ok = bool(np.all(error.reshape(-1, 32).max(axis=1) <= bound + 1e-6))
    elif qtype == gguf.GGMLQuantizationType.F32:
        ok = bool(np.array_equal(actual, expected))
    elif qtype == gguf.GGMLQuantizationType.F16:
        ok = bool(np.array_equal(actual, expected.astype(np.float16).astype(np.float32)))
    elif qtype == gguf.GGMLQuantizationType.BF16:
        # bf16 round-to-nearest-even, independent of gguf-py's quantizer.
        bits = expected.view(np.uint32)
        rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) << 16
        ok = bool(np.array_equal(actual, rounded.view(np.float32)))
    else:
        raise ValueError(f"unsupported map type {qtype}")
    return ok, float(error.max())


def verify(folder: Path, output: Path, model_gguf: Path) -> dict:
    manifest, hashes = converter.read_package(folder)
    reader = gguf.GGUFReader(output)
    tensors = {t.name: t for t in reader.tensors}
    m = manifest["model"]["meta"]
    prefix = "adapter.ridgefill."
    expected_meta = {
        "general.architecture": "qwen35", "general.type": "adapter", "adapter.type": "ridgefill",
        prefix + "split": manifest["split"], prefix + "n_layer": m["num_hidden_layers"],
        prefix + "format": 2,
        prefix + "n_embd": m["hidden_size"], prefix + "source_manifest_sha256": hashes["ridgefill.json"],
        prefix + "tokenizer_sha256": manifest["model"]["tokenizer_json_sha256"],
        prefix + "authors": manifest["authors"],
        prefix + "vocab_sha256": converter.model_vocab_sha256(model_gguf, m["vocab_size"]),
        prefix + "digits_default": manifest["rowsel"]["digits_default"],
        prefix + "model_layer_types": manifest["model"]["layer_types"],
        prefix + "tail_default": manifest["tail"]["default"], prefix + "tail_min": manifest["tail"]["min"],
    }
    expected_meta.update({prefix + k: manifest[k] for k in ("name", "license", "doi")})
    expected_meta.update({prefix + "model_meta." + k: v for k, v in m.items()})
    expected_meta.update({prefix + "source." + k: v for k, v in hashes.items()})
    for key, expected in expected_meta.items():
        if reader.fields[key].contents() != expected:
            raise AssertionError(f"metadata {key}: {reader.fields[key].contents()!r} != {expected!r}")
    for key, section in (("alpha_default", "correction"), ("share_default", "rowsel"), ("qattn_default", "rowsel")):
        np.testing.assert_equal(reader.fields[prefix + key].contents(), float(np.float32(manifest[section][key])))
    if json.loads(reader.fields[prefix + "source_manifest"].contents()) != manifest:
        raise AssertionError("source_manifest did not preserve the manifest")

    expected_names = {"ridgefill.rowsel.score"}
    maximum_error, maps, states = 0.0, 0, 0
    wrong_map_checked = False
    previous = None
    head_control = False
    orientation_control = False
    for il in range(manifest["split"], m["num_hidden_layers"]):
        _, source, _ = converter.read_safetensors(folder / manifest["projector"]["files"][str(il)])
        width = m["hidden_size"]
        codes = source[f"proj.{il}.codes"]
        scales = source[f"proj.{il}.scale"]
        expected = codes.astype(np.float32) * scales[:, np.arange(width) // 128]
        weight_name, bias_name = f"blk.{il}.ridgefill.weight", f"blk.{il}.ridgefill.bias"
        expected_names.update((weight_name, bias_name))
        weight = tensors[weight_name]
        actual = read_tensor(weight)
        ok, error = map_within_bound(actual, expected, weight.tensor_type)
        if not ok:
            raise AssertionError(f"{weight_name}: map exceeds {weight.tensor_type.name} error bound, max abs {error}")
        maximum_error = max(maximum_error, error)
        np.testing.assert_array_equal(read_tensor(tensors[bias_name]), source[f"proj.{il}.bias"])
        if previous is not None and not np.array_equal(previous, expected):
            wrong_ok, _ = map_within_bound(previous, expected, weight.tensor_type)
            if wrong_ok:
                raise AssertionError("positive wrong-map control was not rejected")
            wrong_map_checked = True
        previous = actual
        maps += 1

    _, correction, _ = converter.read_safetensors(folder / "correction.safetensors")
    nk, nv = m["linear_num_key_heads"], m["linear_num_value_heads"]
    # Each tiled head chooses (key_head, value_within_key) from the HF grouped order.
    indices = [(head % nk) * (nv // nk) + head // nk for head in range(nv)]
    for name, source_st in correction.items():
        il = int(name.split(".")[1])
        ggname = f"blk.{il}.ridgefill.st"
        expected_names.add(ggname)
        actual = read_tensor(tensors[ggname])
        np.testing.assert_array_equal(actual, source_st[indices])
        transposed = source_st[indices].swapaxes(-1, -2)
        if not np.array_equal(actual, transposed):
            orientation_control = True
        if not np.array_equal(source_st, source_st[indices]):
            if np.array_equal(actual, source_st):
                raise AssertionError("positive wrong-head-order control was not rejected")
            head_control = True
        states += 1
    _, rowsel, _ = converter.read_safetensors(folder / "rowsel.safetensors")
    np.testing.assert_array_equal(read_tensor(tensors["ridgefill.rowsel.score"]), rowsel["score"])
    if set(tensors) != expected_names:
        raise AssertionError(f"unexpected tensor set: {set(tensors) ^ expected_names}")
    if not maps or not states or not wrong_map_checked or not head_control or not orientation_control:
        raise ValueError("SKIPPED: maps, correction or nontrivial controls missing; no passing verdict")
    return dict(result="PASS", maps=maps, states=states, tensors=len(tensors), max_map_abs_error=maximum_error,
                metadata="PASS", bias_and_score_exact="PASS", wrong_map_control="PASS", head_order_control="PASS", orientation_control="PASS",
                vocab_sha256=reader.fields[prefix + "vocab_sha256"].contents(),
                source_sha256=hashes["ridgefill.json"], output_sha256=converter.file_sha256(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--model-gguf", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.package, args.gguf, args.model_gguf), sort_keys=True))
