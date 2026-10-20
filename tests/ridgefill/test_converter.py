"""Torch-free converter checks: all output types, legacy compatibility and named refusals."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from safetensors import serialize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import convert_ridgefill_to_gguf as converter
from check_converter import verify


def save_tensors(path: Path, tensors: dict, metadata: dict | None = None) -> None:
    data = {}
    for name, (dtype, array) in tensors.items():
        name_dtype = {"I8": "int8", "BF16": "bfloat16", "F32": "float32"}[dtype]
        data[name] = dict(dtype=name_dtype, shape=list(array.shape), data=array.tobytes())
    path.write_bytes(serialize(data, metadata=metadata))


def seal(folder: Path, manifest: dict) -> None:
    manifest["files"] = {p.name: converter.file_sha256(p) for p in sorted(folder.glob("*.safetensors"))}
    (folder / "ridgefill.json").write_text(json.dumps(manifest))
    paths = sorted(folder.glob("*.safetensors")) + [folder / "ridgefill.json"]
    (folder / "SHA256SUMS").write_text("".join(f"{converter.file_sha256(p)}  {p.name}\n" for p in paths))


def make_package(folder: Path) -> dict:
    rng = np.random.default_rng(123)
    meta = dict(num_hidden_layers=3, hidden_size=64, vocab_size=32, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, intermediate_size=160, full_attention_interval=4,
                linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16)
    manifest = dict(format=1, name="test-projector", authors=["Test Author"], license="Apache-2.0", doi="test-doi",
                    model=dict(model_type="qwen3_5_text", meta=meta, tokenizer_json_sha256="0" * 64,
                               layer_types="linear_attention linear_attention full_attention"),
                    split=1, stream_width=64, block_in_width=64, layers={"1": "recurrent", "2": "full_attn"},
                    tail=dict(default=4096, min=512), correction=dict(alpha_default=0.0, layers=[1], layout="[heads, V, K]"),
                    rowsel=dict(share_default=0.5, qattn_default=0.2, digits_default=True),
                    projector=dict(encoding="i8*bf16[1x128]", files={str(i): f"proj8.L{i}.safetensors" for i in (1, 2)}))
    for il in (1, 2):
        codes = rng.integers(-127, 128, size=(64, 64), dtype=np.int8)
        scales = (rng.uniform(0.0001, 0.002, (64, 1)).astype(np.float32).view(np.uint32) >> 16).astype("<u2")
        bias = (rng.normal(0, 0.01, 64).astype(np.float32).view(np.uint32) >> 16).astype("<u2")
        save_tensors(folder / f"proj8.L{il}.safetensors",
                     {f"proj.{il}.codes": ("I8", codes), f"proj.{il}.scale": ("BF16", scales), f"proj.{il}.bias": ("BF16", bias)})
    save_tensors(folder / "correction.safetensors", {"st.1": ("F32", rng.normal(size=(4, 16, 16)).astype("<f4"))})
    score = rng.normal(size=32).astype("<f4")
    score[::3] = -np.inf
    save_tensors(folder / "rowsel.safetensors", {"score": ("F32", score)})
    seal(folder, manifest)
    return manifest


class ConverterTests(unittest.TestCase):
    def setUp(self):
        output = Path(__file__).resolve().parents[2] / "out"
        output.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=output)
        self.folder = Path(self.temporary.name)
        self.manifest = make_package(self.folder)
        self.output = self.folder / "projector.gguf"
        self.model_gguf = self.folder / "model.gguf"
        self.tokens = [f"token-{i}" for i in range(32)]
        writer = converter.gguf.GGUFWriter(self.model_gguf, "qwen35")
        writer.add_token_list(self.tokens)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def tearDown(self):
        self.temporary.cleanup()

    def convert(self, outtype="q8_0", base=None):
        converter.convert_package(argparse.Namespace(projector=self.folder, base=base, outtype=outtype, outfile=self.output, model_gguf=self.model_gguf))

    def test_output_types_and_controls(self):
        for outtype in converter.OUTTYPES:
            with self.subTest(outtype=outtype):
                self.convert(outtype)
                result = verify(self.folder, self.output, self.model_gguf)
                self.assertEqual(result["result"], "PASS")
                self.assertEqual(result["maps"], 2)
                self.assertEqual(result["states"], 1)

    def test_vocab_hash(self):
        expected = hashlib.sha256(b"".join(token.encode("utf-8") + b"\0" for token in self.tokens)).hexdigest()
        self.assertEqual(converter.model_vocab_sha256(self.model_gguf, 32), expected)
        without_terminators = hashlib.sha256("".join(self.tokens).encode("utf-8")).hexdigest()
        self.assertNotEqual(expected, without_terminators)
        with self.assertRaisesRegex(ValueError, "tokenizer.ggml.tokens count"):
            converter.model_vocab_sha256(self.model_gguf, 31)

    def test_model_gguf_required(self):
        with self.assertRaisesRegex(ValueError, "--model-gguf is required"):
            converter.convert_package(argparse.Namespace(projector=self.folder, base=None, outtype="q8_0", outfile=self.output, model_gguf=None))

    def test_corruption_refused_and_output_preserved(self):
        self.output.write_bytes(b"previous output")
        path = self.folder / "proj8.L1.safetensors"
        path.write_bytes(path.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "proj8.L1.safetensors: sha256 mismatch"):
            self.convert()
        self.assertEqual(self.output.read_bytes(), b"previous output")

    def test_missing_file_refused(self):
        (self.folder / "correction.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "correction.safetensors: missing"):
            self.convert()

    def test_wrong_correction_shape_refused(self):
        save_tensors(self.folder / "correction.safetensors", {"st.1": ("F32", np.zeros((4, 16, 15), dtype="<f4"))})
        seal(self.folder, self.manifest)
        with self.assertRaisesRegex(ValueError, "st.1: shape"):
            self.convert()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_name(self.output.name + ".tmp").exists())

    def test_wrong_model_metadata_refused(self):
        base = self.folder / "base"
        base.mkdir()
        config = dict(self.manifest["model"]["meta"], hidden_size=128)
        (base / "config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "model.meta.hidden_size"):
            self.convert(base=base)

    def test_wrong_tokenizer_refused(self):
        base = self.folder / "base"
        base.mkdir()
        (base / "config.json").write_text(json.dumps(self.manifest["model"]["meta"]))
        (base / "tokenizer.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "tokenizer.json sha256 mismatch"):
            self.convert(base=base)

    def test_legacy_file(self):
        source = self.folder / "legacy.safetensors"
        weights = np.arange(64 * 65, dtype=np.float32).reshape(64, 65) / 10000
        save_tensors(source, {"layer.1": ("F32", weights)}, metadata={"split": "1"})
        base = self.folder / "base"
        base.mkdir()
        (base / "config.json").write_text(json.dumps(dict(model_type="llama", num_hidden_layers=2, hidden_size=64)))
        with patch.object(sys, "argv", ["convert", str(source), "--base", str(base), "--outtype", "f32", "--outfile", str(self.output)]):
            converter.main()
        tensors = {t.name: t for t in converter.gguf.GGUFReader(self.output).tensors}
        np.testing.assert_array_equal(tensors["blk.1.ridgefill.weight"].data, weights[:, :-1])
        np.testing.assert_array_equal(tensors["blk.1.ridgefill.bias"].data, weights[:, -1])


if __name__ == "__main__":
    unittest.main()
