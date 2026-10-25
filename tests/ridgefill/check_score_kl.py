#!/usr/bin/env python3
"""Recompute native scorer KL statistics from two RFLP float32 files."""

import argparse
import json
import math
from pathlib import Path
import struct


def read_header(file):
    magic, version, dtype, vocab, scored = struct.unpack("<4s4I", file.read(20))
    assert (magic, version, dtype) == (b"RFLP", 1, 32)
    targets = struct.unpack(f"<{scored}i", file.read(4 * scored))
    return vocab, scored, targets


def compare(reference, current, record):
    kl, top1, dp = [], [], []
    with reference.open("rb") as ref, current.open("rb") as now:
        vocab, scored, targets = read_header(ref)
        assert read_header(now) == (vocab, scored, targets)
        for target in targets:
            p = struct.unpack(f"<{vocab}f", ref.read(4 * vocab))
            q = struct.unpack(f"<{vocab}f", now.read(4 * vocab))
            assert all(math.isfinite(x) and x <= 0 for x in (*p, *q))
            kl.append(math.fsum(math.exp(a) * (a - b) for a, b in zip(p, q)))
            top1.append(max(range(vocab), key=p.__getitem__) == max(range(vocab), key=q.__getitem__))
            dp.append(math.exp(q[target]) - math.exp(p[target]))
        assert not ref.read(1) and not now.read(1)
    ordered = sorted(kl)

    def percentile(fraction):
        position = fraction * (scored - 1)
        first = int(position)
        return ordered[first] + (position - first) * (ordered[min(first + 1, scored - 1)] - ordered[first])

    expected = dict(kl_mean=math.fsum(kl) / scored, kl_median=percentile(0.5), kl_p99=percentile(0.99),
                    kl_max=max(kl), top1_agree=sum(top1) / scored, dp_target_mean=math.fsum(dp) / scored)
    assert record["top1"] == top1
    assert len(record["kl"]) == scored
    errors = [abs(a - b) for a, b in zip(kl, record["kl"])]
    errors.extend(abs(value - record[key]) for key, value in expected.items())
    assert max(errors) <= 1e-9, max(errors)
    return dict(result="PASS", positions=scored, n_vocab=vocab, max_abs_error=max(errors), **expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.reference, args.current, json.loads(args.record.read_text())), sort_keys=True))


if __name__ == "__main__":
    main()
