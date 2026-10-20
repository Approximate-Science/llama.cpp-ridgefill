#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Convert a RidgeFill projector (kva safetensors or format-1 int8 folder) to GGUF.
#
# The input file holds one tensor 'layer.<il>' per layer il >= split, shape [n_embd, n_embd + 1] (bf16):
# the input of layer il is predicted as x_il = h @ W^T + c, with W = t[:, :-1], c = t[:, -1] and h the input of
# layer split. Its metadata holds 'split'. The GGUF file holds blk.<il>.ridgefill.weight and blk.<il>.ridgefill.bias.

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from safetensors import deserialize

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

logger = logging.getLogger("ridgefill-to-gguf")

# model_type in config.json -> GGUF arch; the fill graph supports these only
ARCHS = {
    "llama": gguf.MODEL_ARCH.LLAMA,
    "qwen3": gguf.MODEL_ARCH.QWEN3,
    "qwen3_5_text": gguf.MODEL_ARCH.QWEN35,
}

OUTTYPES = {
    "f32":  gguf.GGMLQuantizationType.F32,
    "f16":  gguf.GGMLQuantizationType.F16,
    "bf16": gguf.GGMLQuantizationType.BF16,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
}

MAX_ABS_WEIGHT = 1e4  # fitted maps are far below this, larger values mean a broken or corrupted file


def read_safetensors(path: Path) -> tuple[dict[str, str], dict[str, np.ndarray], str]:
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()

    decoded = deserialize(data)
    n = int.from_bytes(data[:8], "little")
    meta = json.loads(data[8:8 + n]).get("__metadata__", {})
    tensors = {}
    for name, info in decoded:
        raw = info["data"]
        if info["dtype"] == "BF16":
            t = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        elif info["dtype"] == "F16":
            t = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        elif info["dtype"] == "F32":
            t = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        elif info["dtype"] == "I8":
            t = np.frombuffer(raw, dtype=np.int8)
        else:
            raise ValueError(f"{name}: unsupported dtype {info['dtype']}")
        tensors[name] = t.reshape(info["shape"])

    return meta, tensors, sha256


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def package_file(folder: Path, name: str) -> Path:
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"invalid package filename {name!r}")
    path = folder / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name}: missing file or symlink, refusing")
    return path


def model_vocab_sha256(path: Path, vocab_size: int) -> str:
    reader = gguf.GGUFReader(path)
    if reader.fields["general.architecture"].contents() != "qwen35":
        raise ValueError("--model-gguf: general.architecture must be qwen35")
    tokens = reader.fields.get("tokenizer.ggml.tokens")
    if tokens is None or tokens.types != [gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.STRING]:
        raise ValueError("--model-gguf: tokenizer.ggml.tokens must be a string array")
    if len(tokens.data) != vocab_size:
        raise ValueError(f"--model-gguf: tokenizer.ggml.tokens count {len(tokens.data)} != model.meta.vocab_size {vocab_size}")
    digest = hashlib.sha256()
    for index in tokens.data:
        digest.update(tokens.parts[index].tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def read_package(folder: Path) -> tuple[dict, dict[str, str]]:
    manifest_path = package_file(folder, "ridgefill.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest["format"] != 1:
        raise ValueError(f"ridgefill.json: unsupported format {manifest['format']!r}")
    if manifest["model"]["model_type"] != "qwen3_5_text":
        raise ValueError("ridgefill.json: int8 folder requires model_type qwen3_5_text")
    if manifest["projector"]["encoding"] != "i8*bf16[1x128]":
        raise ValueError("ridgefill.json: unsupported projector.encoding")

    hashes = {}
    for line in package_file(folder, "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.removeprefix("*")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or name in hashes:
            raise ValueError(f"SHA256SUMS: invalid or duplicate entry {name!r}")
        actual = file_sha256(package_file(folder, name))
        if actual != digest:
            raise ValueError(f"SHA256SUMS: {name}: sha256 mismatch")
        hashes[name] = actual
    if hashes.get("ridgefill.json") != file_sha256(manifest_path):
        raise ValueError("SHA256SUMS: missing ridgefill.json")
    for name, digest in manifest["files"].items():
        if hashes.get(name) != digest:
            raise ValueError(f"ridgefill.json files: {name}: missing checksum or sha256 mismatch")

    m = manifest["model"]["meta"]
    for key, value in m.items():
        if type(value) is not int or not 0 < value <= 0xffffffff:
            raise ValueError(f"model.meta.{key}: expected positive uint32")
    split, n_layer, width = manifest["split"], m["num_hidden_layers"], m["hidden_size"]
    if type(split) is not int or not 1 <= split < n_layer:
        raise ValueError("ridgefill.json: split out of range")
    if manifest["stream_width"] != width or manifest["block_in_width"] != width:
        raise ValueError("ridgefill.json: projector width must equal hidden_size")
    expected = {str(i) for i in range(split, n_layer)}
    if set(manifest["layers"]) != expected or set(manifest["projector"]["files"]) != expected:
        raise ValueError("ridgefill.json: layers and projector.files must cover every late layer")
    kinds = manifest["model"]["layer_types"].split()
    if len(kinds) != n_layer:
        raise ValueError("model.layer_types: incorrect layer count")
    for i, kind in enumerate(kinds):
        if kind not in ("linear_attention", "full_attention"):
            raise ValueError(f"model.layer_types: unsupported layer {i}: {kind}")
        if i >= split and manifest["layers"][str(i)] != ("recurrent" if kind == "linear_attention" else "full_attn"):
            raise ValueError(f"layers.{i}: disagrees with model.layer_types")
    nk, nv = m["linear_num_key_heads"], m["linear_num_value_heads"]
    if nv % nk:
        raise ValueError("model.meta.linear_num_value_heads: not divisible by key heads")
    if manifest["correction"]["layout"] != "[heads, V, K]":
        raise ValueError("correction.layout: expected [heads, V, K]")
    required_files = set(manifest["projector"]["files"].values()) | {"correction.safetensors", "rowsel.safetensors"}
    if not required_files <= set(manifest["files"]):
        raise ValueError("ridgefill.json files: missing projector, correction or rowsel checksums")
    for section, key in (("correction", "alpha_default"), ("rowsel", "share_default"), ("rowsel", "qattn_default")):
        value = manifest[section][key]
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{section}.{key}: expected finite value in [0, 1]")
    if type(manifest["rowsel"]["digits_default"]) is not bool:
        raise ValueError("rowsel.digits_default: expected bool")
    if not 1 <= manifest["tail"]["min"] <= manifest["tail"]["default"] <= 0xffffffff:
        raise ValueError("tail: expected 1 <= min <= default <= uint32 max")
    tokenizer_hash = manifest["model"]["tokenizer_json_sha256"]
    if len(tokenizer_hash) != 64 or any(c not in "0123456789abcdef" for c in tokenizer_hash):
        raise ValueError("model.tokenizer_json_sha256: invalid sha256")
    logger.info("verified SHA256SUMS: %d files", len(hashes))
    return manifest, hashes


def check_shape(name: str, tensor: np.ndarray, shape: tuple[int, ...]) -> None:
    if tensor.shape != shape:
        raise ValueError(f"{name}: shape {tensor.shape}, expected {shape}")
    if not np.isfinite(tensor).all() or np.max(np.abs(tensor)) > MAX_ABS_WEIGHT:
        raise ValueError(f"{name}: non-finite or |w| > {MAX_ABS_WEIGHT:g}, refusing")


def package_map(folder: Path, manifest: dict, il: int) -> tuple[np.ndarray, np.ndarray]:
    _, tensors, _ = read_safetensors(package_file(folder, manifest["projector"]["files"][str(il)]))
    prefix = f"proj.{il}."
    if set(tensors) != {prefix + k for k in ("codes", "scale", "bias")}:
        raise ValueError(f"layer {il}: expected codes, scale and bias")
    codes, scale, bias = (tensors[prefix + k] for k in ("codes", "scale", "bias"))
    width = manifest["model"]["meta"]["hidden_size"]
    check_shape(prefix + "codes", codes.astype(np.float32), (width, width))
    check_shape(prefix + "scale", scale, (width, (width + 127) // 128))
    check_shape(prefix + "bias", bias, (width,))
    if codes.dtype != np.int8 or np.any(scale < 0):
        raise ValueError(f"layer {il}: expected int8 codes and nonnegative scale")
    w = codes.astype(np.float32)
    for start in range(0, width, 128):
        w[:, start:start + 128] *= scale[:, start // 128, None]
    check_shape(prefix + "weight", w, (width, width))
    return w, bias.astype(np.float32)


def convert_package(args: argparse.Namespace) -> None:
    manifest, hashes = read_package(args.projector)
    m = manifest["model"]["meta"]
    if args.model_gguf is None:
        raise ValueError("--model-gguf is required for a format-1 int8 folder (checked vocabulary fingerprint)")
    vocab_sha256 = model_vocab_sha256(args.model_gguf, m["vocab_size"])
    if args.base:
        config = json.loads((args.base / "config.json").read_text())
        config = config.get("text_config", config)
        for key, expected in m.items():
            if config.get(key) != expected:
                raise ValueError(f"--base: model.meta.{key}: {config.get(key)!r} != {expected!r}")
        if file_sha256(args.base / "tokenizer.json") != manifest["model"]["tokenizer_json_sha256"]:
            raise ValueError("--base: tokenizer.json sha256 mismatch")
    outtype = args.outtype or "q8_0"
    qtype = OUTTYPES[outtype]
    outfile = args.outfile or args.projector.with_name(f"{args.projector.name}-{outtype}.gguf")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    temporary = outfile.with_name(outfile.name + ".tmp")
    writer = gguf.GGUFWriter(temporary, "qwen35", use_temp_file=True)
    # Keep the spool next to the output so conversion needs no writes outside its destination.
    writer.temp_file = tempfile.SpooledTemporaryFile(mode="w+b", dir=outfile.parent)
    try:
        writer.add_type(gguf.GGUFType.ADAPTER)
        writer.add_string(gguf.Keys.Adapter.TYPE, "ridgefill")
        prefix = "adapter.ridgefill."
        for key, value in (("format", 2), ("split", manifest["split"]), ("n_layer", m["num_hidden_layers"]), ("n_embd", m["hidden_size"]),
                           ("tail_default", manifest["tail"]["default"]), ("tail_min", manifest["tail"]["min"])):
            writer.add_uint32(prefix + key, value)
        for key, value in (("alpha_default", manifest["correction"]["alpha_default"]),
                           ("share_default", manifest["rowsel"]["share_default"]), ("qattn_default", manifest["rowsel"]["qattn_default"])):
            writer.add_float32(prefix + key, value)
        writer.add_bool(prefix + "digits_default", manifest["rowsel"]["digits_default"])
        for key in ("name", "license", "doi"):
            writer.add_string(prefix + key, manifest[key])
        writer.add_array(prefix + "authors", manifest["authors"])
        writer.add_string(prefix + "source_manifest_sha256", hashes["ridgefill.json"])
        writer.add_string(prefix + "tokenizer_sha256", manifest["model"]["tokenizer_json_sha256"])
        writer.add_string(prefix + "vocab_sha256", vocab_sha256)
        writer.add_string(prefix + "model_layer_types", manifest["model"]["layer_types"])
        writer.add_string(prefix + "source_manifest", json.dumps(manifest, sort_keys=True))
        for key, value in sorted(m.items()):
            writer.add_uint32(prefix + "model_meta." + key, value)
        for name, digest in sorted(hashes.items()):
            writer.add_string(prefix + "source." + name, digest)
        for il in range(manifest["split"], m["num_hidden_layers"]):
            w, b = package_map(args.projector, manifest, il)
            writer.add_tensor(f"blk.{il}.ridgefill.weight", gguf.quants.quantize(w, qtype), raw_dtype=qtype)
            writer.add_tensor(f"blk.{il}.ridgefill.bias", b)
            logger.info("blk.%d.ridgefill.weight: %s %s", il, qtype.name, list(w.shape))
        _, correction, _ = read_safetensors(package_file(args.projector, "correction.safetensors"))
        recurrent = {int(i) for i, kind in manifest["layers"].items() if kind == "recurrent"}
        if set(correction) != {f"st.{i}" for i in recurrent} or set(manifest["correction"]["layers"]) != recurrent:
            raise ValueError("correction.safetensors: expected one st tensor for each late recurrent layer")
        nk, nv = m["linear_num_key_heads"], m["linear_num_value_heads"]
        dv, dk = m["linear_value_head_dim"], m["linear_key_head_dim"]
        for il in sorted(recurrent):
            st = correction[f"st.{il}"]
            check_shape(f"st.{il}", st, (nv, dv, dk))
            # q27_package.py already transposes oracle [H, K, V] to [H, V, K]; only reorder heads here.
            st = st.reshape(nk, nv // nk, dv, dk).swapaxes(0, 1).reshape(nv, dv, dk)
            writer.add_tensor(f"blk.{il}.ridgefill.st", np.ascontiguousarray(st, dtype=np.float32))
        _, rowsel, _ = read_safetensors(package_file(args.projector, "rowsel.safetensors"))
        score = rowsel["score"]
        if score.shape != (m["vocab_size"],) or np.isnan(score).any() or np.isposinf(score).any():
            raise ValueError("rowsel.score: expected vocab_size values, finite or -inf")
        writer.add_tensor("ridgefill.rowsel.score", score.astype(np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
        os.replace(temporary, outfile)
    finally:
        if writer.temp_file is not None:
            writer.temp_file.close()
        writer.close()
        temporary.unlink(missing_ok=True)
    logger.info("wrote %s (sha256 %s)", outfile, file_sha256(outfile))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a RidgeFill projector (kva safetensors) to GGUF")
    parser.add_argument("projector", type=Path, help="projector file (.safetensors) or format-1 int8 release folder")
    parser.add_argument("--base", type=Path, help="HF model directory (required for a legacy file; verifies config and tokenizer for a folder)")
    parser.add_argument("--model-gguf", type=Path, help="base model GGUF (required for an int8 folder to fingerprint tokenizer.ggml.tokens)")
    parser.add_argument("--outfile", type=Path, help="output file (default: projector name with .gguf, plus the type)")
    parser.add_argument("--outtype", choices=list(OUTTYPES), help="type of the maps (default: q8_0 for a folder, bf16 for a legacy file)")
    parser.add_argument("--verbose", action="store_true", help="increase output verbosity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.projector.is_dir():
        convert_package(args)
        return
    if args.base is None:
        raise ValueError("--base is required for a legacy safetensors projector")
    args.outtype = args.outtype or "bf16"
    config = json.loads((args.base / "config.json").read_text())
    config = config.get("text_config", config)

    model_type = config.get("model_type")
    if model_type not in ARCHS:
        raise ValueError(f"model_type {model_type!r} is not supported (supported: {', '.join(ARCHS)})")
    arch = ARCHS[model_type]
    if model_type == "qwen3_5_text":
        raise ValueError("qwen3_5_text requires a format-1 int8 folder with model fingerprint and correction")
    n_layer = int(config["num_hidden_layers"])
    n_embd  = int(config["hidden_size"])

    meta, tensors, sha256 = read_safetensors(args.projector)
    split = int(meta["split"])
    if not 1 <= split < n_layer:
        raise ValueError(f"split {split} is out of range [1, {n_layer})")

    expected = {f"layer.{il}" for il in range(split, n_layer)}
    if set(tensors) != expected:
        raise ValueError(f"expected tensors layer.{split} .. layer.{n_layer - 1}, got {sorted(tensors)}")

    qtype = OUTTYPES[args.outtype]
    outfile = args.outfile or args.projector.with_name(f"{args.projector.stem}-{args.outtype}.gguf")

    writer = gguf.GGUFWriter(outfile, gguf.MODEL_ARCH_NAMES[arch])
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "ridgefill")
    writer.add_uint32("adapter.ridgefill.split",   split)
    writer.add_uint32("adapter.ridgefill.n_layer", n_layer)
    writer.add_uint32("adapter.ridgefill.n_embd",  n_embd)
    writer.add_string("adapter.ridgefill.source_sha256", sha256)
    writer.add_string("adapter.ridgefill.source_file", args.projector.name)
    for k, v in sorted(meta.items()):
        if k != "split":
            writer.add_string(f"adapter.ridgefill.source.{k}", str(v))

    for il in range(split, n_layer):
        t = tensors[f"layer.{il}"]
        if t.shape != (n_embd, n_embd + 1):
            raise ValueError(f"layer.{il}: shape {t.shape}, expected {(n_embd, n_embd + 1)}")
        if not np.isfinite(t).all() or np.abs(t).max() > MAX_ABS_WEIGHT:
            raise ValueError(f"layer.{il}: non-finite or |w| > {MAX_ABS_WEIGHT:g} weights, refusing")

        w = np.ascontiguousarray(t[:, :-1])
        b = np.ascontiguousarray(t[:, -1])

        writer.add_tensor(f"blk.{il}.ridgefill.weight", gguf.quants.quantize(w, qtype), raw_dtype=qtype)
        writer.add_tensor(f"blk.{il}.ridgefill.bias", b.astype(np.float32))
        logger.info(f"blk.{il}.ridgefill.weight: {qtype.name} {list(w.shape)}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()

    logger.info(f"wrote {outfile} (split {split}, {n_layer - split} maps, source sha256 {sha256})")


if __name__ == "__main__":
    main()
