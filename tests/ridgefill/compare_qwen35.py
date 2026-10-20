#!/usr/bin/env python3
"""G1 oracle and hybrid-state parser, adapted from the Oct-8 compare.py.

Traps: decode can erase an alpha correction, so compare recurrent state at the prefix boundary too;
M-RoPE adds cell metadata; only attention layers have KV and only GDN layers have conv/state;
llama.cpp tiles value heads and stores state [H,V,K], while the oracle uses grouped [H,K,V].
Unsupported serialization, absent arrays and non-finite data fail instead of producing an empty pass.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import convert_ridgefill_to_gguf as converter


def read_state(path: Path, config: dict, length: int) -> dict[str, np.ndarray]:
    data = path.read_bytes()
    offset = 0

    def read(fmt):
        nonlocal offset
        result = struct.unpack_from("<" + fmt, data, offset)
        offset += struct.calcsize("<" + fmt)
        return result[0] if len(result) == 1 else result

    def floats(count):
        nonlocal offset
        if offset + count * 4 > len(data):
            raise ValueError("truncated float state")
        result = np.frombuffer(data, dtype="<f4", count=count, offset=offset)
        offset += count * 4
        return result

    def row(expected):
        ty, size = read("iQ")
        if ty != 0 or size != expected * 4:
            raise ValueError(f"state: expected F32 row of {expected}, got type={ty} bytes={size}")
        return floats(expected)

    if read("Ii") != (0xaf143cd8, 0) or read("I") != 1:
        raise ValueError("state: expected sequence 0 and one KV stream")
    cells = read("I")
    if cells != length:
        raise ValueError(f"state: {cells} KV cells != {length}")
    for pos in range(cells):
        saved_pos, nseq = read("iI")
        x, y, token = read("iii")
        if saved_pos != pos or x != pos or y != pos or nseq != 1 or read("i") != 0:
            raise ValueError(f"state: invalid M-RoPE metadata at row {pos}, token {token}")
    trans, nl, rot_k, rot_v = read("IIII")
    kinds = config["layer_types"]
    attention = [i for i, kind in enumerate(kinds) if kind == "full_attention"]
    recurrent = [i for i, kind in enumerate(kinds) if kind == "linear_attention"]
    if nl != len(attention) or rot_k != 0 or rot_v != 0:
        raise ValueError("state: unsupported attention layer count or KV rotation")
    nkv, hd = config["num_key_value_heads"], config["head_dim"]
    width = nkv * hd
    result = {}
    for il in attention:
        ty, size = read("iQ")
        if ty != 0 or size != width * 4:
            raise ValueError("state: expected F32 K rows")
        result[f"{il}.k"] = floats(cells * width).reshape(cells, nkv, hd).transpose(1, 0, 2)
    for il in attention:
        if trans:
            ty, element, columns = read("iII")
            if (ty, element, columns) != (0, 4, width):
                raise ValueError("state: unsupported transposed V")
            result[f"{il}.v"] = floats(cells * width).reshape(nkv, hd, cells).transpose(0, 2, 1)
        else:
            ty, size = read("iQ")
            if ty != 0 or size != width * 4:
                raise ValueError("state: expected F32 V rows")
            result[f"{il}.v"] = floats(cells * width).reshape(cells, nkv, hd).transpose(1, 0, 2)
    if read("I") != 1:
        raise ValueError("state: expected one recurrent cell")
    pos, nseq = read("iI")
    if pos != length - 1 or nseq != 0:
        raise ValueError("state: invalid recurrent metadata")
    if read("II") != (0, len(kinds)):
        raise ValueError("state: unsupported recurrent layout")
    nk, nv = config["linear_num_key_heads"], config["linear_num_value_heads"]
    dk, dv = config["linear_key_head_dim"], config["linear_value_head_dim"]
    kernel = config["linear_conv_kernel_dim"]
    qk = 2 * nk * dk
    channels = qk + nv * dv
    tiled_order = np.array([(head % nk) * (nv // nk) + head // nk for head in range(nv)])
    inverse = np.argsort(tiled_order)
    for il in recurrent:
        raw = row(channels * (kernel - 1)).reshape(channels, kernel - 1).T
        values = raw[:, qk:].reshape(kernel - 1, nv, dv)[:, inverse].reshape(kernel - 1, nv * dv)
        result[f"{il}.conv"] = np.concatenate((raw[:, :qk], values), axis=1)
    for il in recurrent:
        raw = row(nv * dk * dv).reshape(nv, dv, dk)
        result[f"{il}.rec"] = raw[inverse].swapaxes(-1, -2)
    if offset != len(data):
        raise ValueError(f"state: {len(data) - offset} unread bytes")
    return result


def reference(kva: Path, work: Path, split: int, mode: str, alpha: float, rows: list[int]) -> dict:
    import torch
    sys.path.insert(0, str(kva.resolve()))
    from kva.model import Qwen35
    from kva.kva import Arm, Projector, finish, run_lower
    torch.set_num_threads(4)
    model = Qwen35(work / "tiny-qwen35-hf", device="cpu", dtype=torch.float32)
    ids = torch.from_numpy(np.fromfile(work / "tokens.bin", dtype=np.int32).astype(np.int64))
    n, keep = len(ids), 200
    result = {}
    with torch.no_grad():
        if mode == "exact":
            logits, cache = model.forward_exact(ids, keep=torch.arange(n - keep, n))
            _, before = model.forward_exact(ids[:-keep], keep=torch.tensor([n - keep - 1]))
        else:
            folder = work / f"projector-s{split}"
            manifest, _ = converter.read_package(folder)
            weights, biases = {}, {}
            for il in range(split, 8):
                w, b = converter.package_map(folder, manifest, il)
                weights[il], biases[il] = torch.from_numpy(w), torch.from_numpy(b)
            _, raw_st, _ = converter.read_safetensors(folder / "correction.safetensors")
            st = {int(name.split(".")[1]): alpha * torch.from_numpy(np.ascontiguousarray(value.swapaxes(-1, -2))) for name, value in raw_st.items()}
            projector = Projector(split, weights, biases)
            hs, lower = run_lower(model, ids, [split])
            arm = Arm(name="ridgefill", split=split, tail=keep, fill="proj", projector=projector, st=st if alpha else None)
            before = {}
            logits, cache, prefix = finish(model, ids, hs[split], lower, arm, keep=keep, fill_store=before,
                                           exact_rows=rows if mode == "quality" else None)
            if prefix != n - keep:
                raise AssertionError("oracle prefix differs")
            if mode == "quality" and alpha:
                raise ValueError("quality alpha correction requires the rho oracle; unsupported in this harness")
            for il, correction in st.items():
                if alpha:
                    before[il]["rec"] = before[il]["rec"] + correction
        result["logits"] = logits.numpy()
        for il, layer in enumerate(cache):
            for name in ("k", "v", "conv", "rec"):
                if name in layer:
                    result[f"final.{il}.{name}"] = layer[name].numpy().copy()
        items = enumerate(before) if isinstance(before, list) else before.items()
        for il, layer in items:
            if il < split and mode != "exact":
                continue
            for name in ("k", "v", "conv", "rec"):
                if name in layer:
                    result[f"prefix.{il}.{name}"] = layer[name].numpy().copy()
    return result


def compare(work: Path, prefix: Path, expected: dict, split: int, exact: bool, tolerance: float = 1e-4) -> dict:
    config = json.loads((work / "tiny-qwen35-hf/config.json").read_text())
    actual = {"logits": np.fromfile(str(prefix) + ".logits", dtype=np.float32).reshape(200, 300)}
    for stage, length in (("prefix", 500), ("final", 700)):
        path = Path(str(prefix) + (".prefix.kv" if stage == "prefix" else ".kv"))
        for key, value in read_state(path, config, length).items():
            if exact or int(key.split(".")[0]) >= split or stage == "final":
                actual[stage + "." + key] = value
    if set(actual) != set(expected):
        raise ValueError(f"arrays differ: {set(actual) ^ set(expected)}")
    errors = {}
    for name, target in expected.items():
        value = actual[name]
        if value.shape != target.shape or not np.isfinite(value).all() or not np.isfinite(target).all():
            raise ValueError(f"{name}: shape mismatch or non-finite values")
        errors[name] = float(np.max(np.abs(value - target)) / max(float(np.max(np.abs(target))), 1e-30))
    return dict(result="PASS" if all(error <= tolerance for error in errors.values()) else "FAIL",
                arrays=len(errors), tolerance=tolerance, relative_errors=errors, max_relative_error=max(errors.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kva", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--split", type=int, choices=(3, 4), default=3)
    parser.add_argument("--mode", choices=("exact", "speed", "quality"), default="speed")
    parser.add_argument("--alpha", type=float, choices=(0, 1), default=0)
    parser.add_argument("--rows", default="0,7,31,127,255,499")
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()
    rows = [int(x) for x in args.rows.split(',') if x]
    expected = reference(args.kva, args.work, args.split, args.mode, args.alpha, rows)
    if args.save:
        np.savez(args.save, **expected)
    if args.dump:
        result = compare(args.work, args.dump, expected, args.split, args.mode == "exact")
        print(json.dumps(result, sort_keys=True))
        sys.exit(0 if result["result"] == "PASS" else 1)
    elif args.save:
        print(json.dumps(dict(result="PREPARED", arrays=len(expected), path=str(args.save))))
    else:
        parser.error("--dump or --save is required; preparing no artifact is not a check")
