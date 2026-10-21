# Native RidgeFill NLL scorer

`llama-ridgefill-score` reads token IDs and runs one whole-document prefill per document. It does not tokenize, sample, or send HTTP requests. `off`, `speed`, and `quality` use normal decode batches; `quality+qattn` uses the same LOWER / question mass / selected-row LATE prefix / exact LATE tail driver as the server. The model and context stay loaded across documents, with memory cleared between them.

Build in the configured CPU, HIP, Vulkan, or CUDA build directory:

```sh
cmake --build build --target llama-ridgefill-score -j4
```

Input may be a JSON token-ID array, a `{ "name": "doc", "ids": [...] }` object, an array of document objects, or JSONL with one document object per line. Output is JSONL with one record per document: target positions, raw full-vocabulary next-token log-probabilities, full-vocabulary logsumexp, all-token mean NLL, selected exact rows, the selector window, and build/settings metadata. Only the K prediction rows have logits enabled. Full logits are reduced in float64 on the host and are not retained. Invalid tokens, incomplete output, non-finite logits, and unsupported settings fail with a nonzero exit code; partial output is not a completed run.

Defaults match the server recipe: split from the projector, tail 4096, min prompt 8192, class share 0.5, digits on, qattn share 0.2, window 64, alpha 0, F16 KV and h_S. The scorer supports `--cache-type` and `--h-type` f32/f16/bf16. CPU is the default; pass `--gpu-layers 99` in lane Q's assigned GPU environment. `--help` lists runtime controls. JSON/JSONL is the supported interchange format; no NPY dependency is needed.

## The NLL selector window excludes targets

`q27/rf27_quality.py` sets `known = N - scored` and captures `[known-window, known)`. Its `finish(..., keep=K+1)` result is passed to `score_tokens()`, which drops the final logit row and scores IDs `[N-K,N)`. This is also the target exclusion documented in `kva/select.py`; `q27/drive.py` delegates quality work to the quality steps rather than defining a different qattn NLL rule.

For N=16384 and K=512, the exact fill boundary is P=12288, the selector reads **[15808,15872)**, prediction logits come from **[15871,16383)**, and their target IDs are **[15872,16384)**. Intervals are zero-based and half-open. The selection is frozen once before scoring all 512 targets. Using the document's final 64 rows would expose scored targets and does not match the reference. `--qattn-end document` is available only for that explicit diagnostic; its output marks `selector_excludes_targets=false`. The paired wrapper always uses `known` and rejects target exposure.

The reference's primary document NLL averages only the novel tokens selected by `kva.data.novel_mask(ids, K)` (no repeated 8-token span). The existing HTTP `run_g3.py` averages all K targets and changes the qattn window for each truncation. Those are different estimators. The wrapper reports reference-compatible novel-token NLL as `metric=nll`, and retains `metric=nll_all` as an all-token diagnostic. Both summaries use the same `paired_summary` helper and `finals/stats.py` BCa implementation as `run_g3.py`: document weighting, 5000 resamples, seed 20261004. It makes no noninferiority verdict; unmeasured metrics are SKIPPED.

## Lane Q command

From the merged checkout, with its configured GPU build named `build` and its GPU assignment already applied:

```sh
python3 tools/ridgefill/run_nll.py \
  --binary build/bin/llama-ridgefill-score \
  --model /var/home/dylan/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --projector /var/home/dylan/models/qwen3.8-27b/projector/ridgefill-projector-qwen3.8-27b-i8-S16.gguf \
  --docs /var/mnt/qwen-storage/dylan/AI-Work/ridgefill-q27-projector-20261008/evaldata/nll.jsonl \
  --kva /var/home/dylan/projects/research/kva-wt-q27 \
  --out out/g3-native-nll \
  --length 16384 --scored 512 --tail 4096 --limit 8 \
  --gpu-layers 99 --threads 4 --batch 512 --ubatch 512 \
  --cache-type f16 --h-type f16 --flash-attn auto
```

The output directory must be new. The wrapper invokes the scorer twice (off, then quality+qattn); each invocation loads the model once and scores every document. `summary.json` has the same COMPLETE/MEASURED/SKIPPED structure as `run_g3.py`; `nll.jsonl` contains paired document records, `off.jsonl` and `quality+qattn.jsonl` retain the native per-position results, and `provenance.json` records commands and input/statistics hashes. `--timeout` defaults to 7200 seconds per arm. `wall_s` excludes model loading and is not a TTFT measurement.

For a single native arm on an already prepared token file:

```sh
build/bin/llama-ridgefill-score \
  --model /var/home/dylan/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --projector /var/home/dylan/models/qwen3.8-27b/projector/ridgefill-projector-qwen3.8-27b-i8-S16.gguf \
  --input tokens.jsonl --output scores.jsonl --mode quality+qattn \
  --scored 512 --tail 4096 --qattn-end known --gpu-layers 99
```

## Tiny CPU gate

Use L-B's split-4 fixture and a new work directory. `rf_dump` is the existing stock decode harness, compiled against the same build:

```sh
c++ -std=c++17 -O2 -I include -I src -I ggml/include \
  tests/ridgefill/rf_dump.cpp -L build/bin -Wl,-rpath,'$ORIGIN/bin' \
  -lllama -lggml -lggml-base -o build/score-stock-dump
python3 tools/ridgefill/check_score.py \
  --binary build/bin/llama-ridgefill-score \
  --fixture ../llama.cpp-ridgefill-tools/out/gate1-qwen35 \
  --kva /var/home/dylan/projects/research/kva-wt-q27 \
  --stock-dump build/score-stock-dump --work build/score-gate
```

The gate covers all four modes against FP32 `finish()` logits at K=64/T=200 and K=512/T=600 with ubatches 50/128/512. It also compares selector mass and row sets, zero-share versus quality, off versus stock next-token scoring, and changes all targets to prove they do not affect selection or the first predictive distribution. The paired wrapper should also be checked on at least two documents, including repeated spans, before a full-model run. Tiny CPU agreement does not establish GPU or full-model quality.
