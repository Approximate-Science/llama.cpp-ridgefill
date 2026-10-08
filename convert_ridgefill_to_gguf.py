#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Convert a RidgeFill projector (kva safetensors format) to GGUF.
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
from pathlib import Path

import numpy as np

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

logger = logging.getLogger("ridgefill-to-gguf")

# model_type in config.json -> GGUF arch; the fill graph supports these only
ARCHS = {
    "llama": gguf.MODEL_ARCH.LLAMA,
    "qwen3": gguf.MODEL_ARCH.QWEN3,
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

    n = int.from_bytes(data[:8], "little")
    header = json.loads(data[8:8 + n])
    meta = header.pop("__metadata__", None) or {}
    base = 8 + n

    tensors = {}
    for name, info in header.items():
        begin, end = info["data_offsets"]
        raw = data[base + begin:base + end]
        if info["dtype"] == "BF16":
            t = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif info["dtype"] == "F16":
            t = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif info["dtype"] == "F32":
            t = np.frombuffer(raw, dtype=np.float32).copy()
        else:
            raise ValueError(f"{name}: unsupported dtype {info['dtype']}")
        tensors[name] = t.reshape(info["shape"])

    return meta, tensors, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a RidgeFill projector (kva safetensors) to GGUF")
    parser.add_argument("projector", type=Path, help="projector file (.safetensors)")
    parser.add_argument("--base", type=Path, required=True, help="directory of the HF model the projector was fitted for (reads config.json)")
    parser.add_argument("--outfile", type=Path, help="output file (default: projector name with .gguf, plus the type)")
    parser.add_argument("--outtype", choices=list(OUTTYPES), default="bf16", help="type of the maps (default: bf16, the type they are stored in)")
    parser.add_argument("--verbose", action="store_true", help="increase output verbosity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    config = json.loads((args.base / "config.json").read_text())
    config = config.get("text_config", config)

    model_type = config.get("model_type")
    if model_type not in ARCHS:
        raise ValueError(f"model_type {model_type!r} is not supported (supported: {', '.join(ARCHS)})")
    arch = ARCHS[model_type]
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
