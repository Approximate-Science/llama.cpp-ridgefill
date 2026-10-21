#!/bin/bash
# G0 (off is stock) + the dense Llama/Qwen3 regression gate, CPU, on the Oct-8 tiny F32 fixtures.
#   G0:  this branch with no projector, and with a projector loaded but nothing filled (T >= N), must be
#        byte-identical to upstream (the commit this branch is rebased on) in logits and KV/state bytes;
#        llama-perplexity per-chunk output identical (no projector, and projector loaded but min-prompt > n_ctx).
#   dense regression: fill vs kva.kva.finish (1e-4 FA off, 1e-2 FA on), fill != exact, wrong-map positive control.
# Traps: rf_dump_stock must be built from the SAME upstream commit as the merge base (checked below), otherwise
# "identical" compares two different upstreams. A run that produced no file is FAIL, never PASS.
# Trap (upstream, measured 2026-10-20): llama.cpp's async logger can drop its last lines at exit with rc 0 under CPU
# load (upstream llama-perplexity 6/40, this branch 4/40, 4 parallel runs). A perplexity log without its
# "Final estimate" line is rerun (at most 3 attempts, the count is printed); the comparison still requires all chunks.
# Usage: FIX=... PY=... KVA=... rf27-gates/g0_dense.sh   (from anywhere; needs build-cpu/ of this repo built)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(dirname "$HERE")"
cd "$HERE"
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
# machine-local inputs are required, never defaulted (G5): the Oct-8 dense fixture dir (gate1/ with work/ and
# compare.py), the kva repo of the dense oracle and its python
: "${FIX:?set FIX to the dense gate1 fixture directory}" "${PY:?set PY to the dense oracle python}" "${KVA:?set KVA to the dense kva repo}"
BASE=$(git -C "$REPO" merge-base HEAD origin/master)
JOBS=${JOBS:-4}
mkdir -p out
out=results-g0${ONLY:+-$ONLY}.txt; : > $out
say() { echo "$*" | tee -a $out; }

# upstream at the merge base
if [ "$(cat .upstream/.commit 2>/dev/null)" != "$BASE" ]; then
  rm -rf .upstream && mkdir -p .upstream && git -C "$REPO" archive "$BASE" | tar -x -C .upstream && echo "$BASE" > .upstream/.commit
fi
cmake -S .upstream -B build-upstream -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF >/dev/null || exit 1
cmake --build build-upstream -j$JOBS --target llama llama-perplexity >/dev/null || exit 1
cmake --build "$REPO/build-cpu" -j$JOBS --target llama llama-perplexity >/dev/null || exit 1

cc() { # out-binary build-dir src-dir extra-flags
  g++ -O2 -std=c++17 $4 -I"$3/include" -I"$3/ggml/include" -I"$3/src" rf_dump.cpp -o "$1" \
      -L"$2/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$2/bin" || exit 1
}
cc rf_dump_stock build-upstream .upstream -DSTOCK
cc rf_dump "$REPO/build-cpu" "$REPO" ""
say "INFO upstream base $BASE, branch $(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet HEAD || echo '+dirty')"

same() { [ -s "$1.logits" ] && [ -s "$1.kv" ] && cmp -s "$1.logits" "$2.logits" && cmp -s "$1.kv" "$2.kv"; }
summ='import json,sys; d=json.load(sys.stdin); print("logits rel", "%.2e" % d["logits"], "kv rel max", "%.2e" % max(max(v["k"], v["v"]) for v in d["kv"].values()))'

for tiny in llama qwen3; do
  if [ $tiny = llama ]; then M=$FIX/work/tiny-llama-f32.gguf; PJ=proj-s3; cfgs="128/64 700/512 50/37 256/256"
  else                       M=$FIX/work/tiny-qwen3-f32.gguf; PJ=qproj-s2; cfgs="128/64 700/512"; fi
  export TINY=$tiny
  T=$FIX/work/tokens.bin; TAIL=200
  for fa in 0 1; do
    [ "${ONLY:-}" = ppl ] && break
    for cfg in $cfgs; do
      b=${cfg%/*}; ub=${cfg#*/}; tag="$tiny-fa$fa-b$b-ub$ub"
      if [ $fa = 1 ]; then export TOL=1e-2; else export TOL=1e-4; fi
      rm -f out/*-$tag.*
      ./rf_dump_stock -m $M -t $T -o out/stock-$tag --keep $TAIL -b $b -ub $ub --fa $fa 2>/dev/null
      ./rf_dump       -m $M -t $T -o out/off-$tag   --keep $TAIL -b $b -ub $ub --fa $fa 2>/dev/null
      same out/off-$tag out/stock-$tag && say "PASS G0 off        $tag: no projector, byte-identical to upstream" \
                                       || say "FAIL G0 off        $tag"
      ./rf_dump -m $M -t $T -o out/tgen-$tag --proj $FIX/work/$PJ-f32.gguf --tail 700 --keep $TAIL -b $b -ub $ub --fa $fa 2>/dev/null
      same out/tgen-$tag out/stock-$tag && say "PASS G0 T>=N       $tag: projector loaded, nothing filled, byte-identical" \
                                        || say "FAIL G0 T>=N       $tag"
      ./rf_dump -m $M -t $T -o out/fill-$tag --proj $FIX/work/$PJ-f32.gguf --tail $TAIL --keep $TAIL -b $b -ub $ub --fa $fa 2>/dev/null
      r=$($PY $FIX/compare.py $KVA $FIX/work out/fill-$tag fill $FIX/work/$PJ.safetensors $TAIL) && s=PASS || s=FAIL
      say "$s dense fill      $tag (tol $TOL): $(echo "$r" | $PY -c "$summ" 2>&1 | tail -1)"
      r=$($PY $FIX/compare.py $KVA $FIX/work out/fill-$tag exact) && s=FAIL || s=PASS
      say "$s dense fill!=ex  $tag: $(echo "$r" | $PY -c "$summ" 2>&1 | tail -1)"
      ./rf_dump -m $M -t $T -o out/wrong-$tag --proj $FIX/work/$PJ-wrong-f32.gguf --tail $TAIL --keep $TAIL -b $b -ub $ub --fa $fa 2>/dev/null
      r=$($PY $FIX/compare.py $KVA $FIX/work out/wrong-$tag fill $FIX/work/$PJ.safetensors $TAIL) && s=FAIL || s=PASS
      [ -s out/wrong-$tag.logits ] || s=FAIL
      say "$s dense wrongmap  $tag: $(echo "$r" | $PY -c "$summ" 2>&1 | tail -1) (must exceed tolerance)"
    done
  done
  if [ "${ONLY:-}" != ppl ]; then
    ./rf_dump -m $M -t $T -o out/err-$tiny --proj $FIX/work/$PJ-f32.gguf --tail $TAIL --keep $((TAIL + 1)) -b 128 -ub 64 2>/dev/null; rc=$?
    [ $rc -eq 2 ] && say "PASS refuse-output $tiny: decode refuses a filled token that is an output (rc $rc)" || say "FAIL refuse-output $tiny (rc $rc)"
  fi

  # llama-perplexity: per-chunk lines identical (timings stripped)
  # -o: the async log interleaves lines; the exit code goes to <out>.rc (a run cut short is visible as such)
  ppl() { local o=$1; shift; local a
          for a in 1 2 3; do
            "$1" -m $M -f $FIX/work/text.txt -c 256 -b 256 -t 4 "${@:2}" > $o.log 2>&1; echo $? > $o.rc; echo $a > $o.attempts
            grep -q 'Final estimate' $o.log && break
          done
          grep -oE '\[[0-9]+\][0-9.]+|Final estimate.*' $o.log > $o; }
  ppl out/ppl-stock-$tiny.txt build-upstream/bin/llama-perplexity
  ppl out/ppl-off-$tiny.txt  "$REPO/build-cpu/bin/llama-perplexity"
  ppl out/ppl-tgen-$tiny.txt "$REPO/build-cpu/bin/llama-perplexity" --ridgefill $FIX/work/$PJ-f32.gguf --ridgefill-min-prompt 100000
  for v in off tgen; do
    [ -s out/ppl-stock-$tiny.txt ] && grep -q '^\[33\]' out/ppl-stock-$tiny.txt && cmp -s out/ppl-stock-$tiny.txt out/ppl-$v-$tiny.txt \
      && say "PASS G0 ppl-$v    $tiny: $(grep -c . out/ppl-stock-$tiny.txt) lines identical ($(tail -1 out/ppl-stock-$tiny.txt | tr -s ' ' | cut -c1-60)), attempts $(cat out/ppl-stock-$tiny.txt.attempts)/$(cat out/ppl-$v-$tiny.txt.attempts)" \
      || say "FAIL G0 ppl-$v    $tiny (rc stock $(cat out/ppl-stock-$tiny.txt.rc) $v $(cat out/ppl-$v-$tiny.txt.rc), lines $(grep -c . out/ppl-stock-$tiny.txt)/$(grep -c . out/ppl-$v-$tiny.txt))"
  done
done
# qwen35 (hybrid GDN + M-RoPE), L-B's tiny fixture: G0 rows only (oracle agreement is L-B's run_gate1_qwen35)
Q35=${Q35:-$HERE/lb/out/gate1-qwen35}
if [ "${ONLY:-}" = ppl ]; then
  :   # perplexity rows only
elif [ -s "$Q35/tiny-qwen35-f32.gguf" ]; then
  M=$Q35/tiny-qwen35-f32.gguf; T=$Q35/tokens.bin
  for cfg in 700/512 128/128 50/50; do
    b=${cfg%/*}; ub=${cfg#*/}; tag="qwen35-b$b-ub$ub"
    rm -f out/*-$tag.*
    ./rf_dump_stock -m $M -t $T -o out/stock-$tag --keep 200 -b $b -ub $ub --fa 0 2>/dev/null
    ./rf_dump       -m $M -t $T -o out/off-$tag   --keep 200 -b $b -ub $ub --fa 0 2>/dev/null
    same out/off-$tag out/stock-$tag && say "PASS G0 off        $tag: no projector, byte-identical to upstream" || say "FAIL G0 off        $tag"
    ./rf_dump -m $M -t $T -o out/tgen-$tag --proj $Q35/projector-s3-f32.gguf --tail 700 --keep 200 -b $b -ub $ub --fa 0 2>/dev/null
    same out/tgen-$tag out/stock-$tag && say "PASS G0 T>=N       $tag: projector loaded, nothing filled, byte-identical" || say "FAIL G0 T>=N       $tag"
  done
else
  say "SKIPPED G0 qwen35: no fixture at $Q35"
fi
say "TOTAL PASS $(grep -c '^PASS' $out) FAIL $(grep -c '^FAIL' $out) SKIPPED $(grep -c '^SKIPPED' $out)"
