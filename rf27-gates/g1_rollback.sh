#!/bin/bash
# Rollback (n_rs_seq > 0) vs the fill, tiny qwen35 (L-B's fixture), CPU.
#  1. L-B's G1 speed harness with n_rs_seq = 4 (all oracle checks and controls).
#  2. Boundary: batches cross P and end 1..4 exact tokens after it (< n_rs_seq + 1 = 5), so split_equal's keep-tail
#     rule meets the fill boundary inside one batch.
#     Batches cross P (--cross, as the server's), so only logits and final states are compared.
#     n_rs_seq 4 must equal n_rs_seq 0 within 1e-4 at 3 chunk configs; control: the fill differs from stock by > 1e-2.
# Trap: before the fix the keep-tail rule extended the filled ubatch with tail rows; the mixed ubatch then ran stock
# (it has outputs), so the "fill" silently computed the prefix exactly. The control catches a fill that did nothing.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
: "${ORACLE_PYTHON:?set ORACLE_PYTHON to the CPU-torch oracle python}" "${KVA:?set KVA to the kva repo (q27-projector)}"
PY=$ORACLE_PYTHON
W=$HERE/lb/out/gate1-qwen35
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
out=results-g1-rollback.txt; : > $out
say() { echo "$*" | tee -a $out; }
mkdir -p out/rb

r=$($PY lb/tests/ridgefill/run_gate1_qwen35.py --kva $KVA --work $W --dump ./rf_dump_rs4.sh --split 3 --mode speed 2>/dev/null | tail -1)
case "$r" in *'"result": "PASS"'*) s=PASS;; *) s=FAIL;; esac
say "$s harness-speed-n_rs_seq4: $r"
cp $W/g1-s3-speed/results.json results-g1-qwen35-speed-rs4.json 2>/dev/null

N=700; T=8; P=$((N - T))
# batch sizes chosen so that a batch ends 1..4 tokens after P (694, 695: < n_rs_seq + 1 tail tokens in the batch)
for cfg in 700/512 694/512 695/128 128/128 50/50; do
  b=${cfg%/*}; ub=${cfg#*/}; tag=b$b-ub$ub
  rm -f out/rb/*-$tag.*
  for rs in 0 4; do
    ./rf_dump_rs --model $W/tiny-qwen35-f32.gguf --tokens $W/tokens.bin --output out/rb/fill-rs$rs-$tag --mode speed --cross \
      --tail $T --keep $T --batch $b --ubatch $ub --projector $W/projector-s3-f32.gguf --n-rs-seq $rs > out/rb/fill-rs$rs-$tag.log 2>&1
  done
  ./rf_dump_rs --model $W/tiny-qwen35-f32.gguf --tokens $W/tokens.bin --output out/rb/stock-$tag --mode off --tail $T --keep $T --cross \
    --batch $b --ubatch $ub --n-rs-seq 4 > out/rb/stock-$tag.log 2>&1
  r=$($PY cmp_states.py $W out/rb/fill-rs4-$tag out/rb/fill-rs0-$tag $N 0 $T 2>&1 | tail -1)
  s=$(echo "$r" | $PY -c 'import json,sys; print("PASS" if json.loads(sys.stdin.read())["max_rel"] <= 1e-4 else "FAIL")' 2>/dev/null || echo FAIL)
  say "$s boundary rs4==rs0  $tag T=$T: $r"
  r=$($PY cmp_states.py $W out/rb/fill-rs4-$tag out/rb/stock-$tag $N 0 $T 2>&1 | tail -1)
  s=$(echo "$r" | $PY -c 'import json,sys; print("PASS" if json.loads(sys.stdin.read())["max_rel"] > 1e-2 else "FAIL")' 2>/dev/null || echo FAIL)
  say "$s control fill!=stock $tag T=$T: $r (must exceed 1e-2)"
done
say "TOTAL PASS $(grep -c '^PASS' $out) FAIL $(grep -c '^FAIL' $out)"
