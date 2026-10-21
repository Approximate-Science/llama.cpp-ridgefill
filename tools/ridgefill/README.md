# Native RidgeFill NLL scorer

`llama-ridgefill-score` reads token IDs and runs one whole-document prefill per document. It does not tokenize, sample, or send HTTP requests. `off`, `speed`, and `quality` use normal decode batches; `quality+qattn` uses the same LOWER / question mass / selected-row LATE prefix / exact LATE tail driver as the server. The model and context stay loaded across documents, with memory cleared between them.

Build in the configured CPU, HIP, Vulkan, or CUDA build directory:

```sh
cmake --build build --target llama-ridgefill-score -j4
```

Input may be a JSON token-ID array, a `{ "name": "doc", "ids": [...] }` object, an array of document objects, or JSONL with one document object per line. Output is JSONL with one record per document: target positions, raw full-vocabulary next-token log-probabilities, full-vocabulary logsumexp, all-token mean NLL, selected exact rows, the selector window, and build/settings metadata. Only the K prediction rows have logits enabled. Full logits are reduced in float64 on the host and are not retained. Invalid tokens, incomplete output, non-finite logits, and unsupported settings fail with a nonzero exit code; partial output is not a completed run.

Defaults match the server recipe: split from the projector, tail 4096, min prompt 8192, class share 0.5, digits on, qattn share 0.2, window 64 ending at N, alpha 0, F16 KV and h_S. The scorer supports `--cache-type` and `--h-type` f32/f16/bf16. CPU is the default; pass `--gpu-layers 99` in lane Q's assigned GPU environment. `--help` lists runtime controls. JSON/JSONL is the supported interchange format; no NPY dependency is needed.

## The NLL selector window excludes targets

`q27/rf27_quality.py` sets `known = N - scored` and captures `[known-window, known)`. Its `finish(..., keep=K+1)` result is passed to `score_tokens()`, which drops the final logit row and scores IDs `[N-K,N)`. This is also the target exclusion documented in `kva/select.py`; `q27/drive.py` delegates quality work to the quality steps rather than defining a different qattn NLL rule.

`--qattn-window-end END` sets the exclusive endpoint of `[END-window,END)`. It accepts a positive row index, `document` (N, the default), or `known` (N-K for each document). `--qattn-end` remains an alias. The default matches the server's prompt-end selector; G3 must explicitly choose N-K. The paired wrapper passes that numeric endpoint and rejects target exposure.

For N=16384 and K=512, use `--qattn-window-end 15872`: the exact fill boundary is P=12288, the selector reads **[15808,15872)**, prediction logits come from **[15871,16383)**, and their target IDs are **[15872,16384)**. Intervals are zero-based and half-open. The selection is frozen once before scoring all 512 targets. Using the document's final 64 rows exposes scored targets, does not match the NLL reference, and marks `selector_excludes_targets=false` in the output.

These boundaries match `ridgefill-tune/REFERENCE-ARMS.md`, "For the engine lanes": G2 uses the prompt's last 64 rows, and teacher-forced needles use the 64 rows before the appended answer (`END=N-answer_length`). The S=16 projector selects attention layers 3, 7, 11 and 15; the split-4 tiny gate uses layer 3.

The reference's primary document NLL averages only the novel tokens selected by `kva.data.novel_mask(ids, K)` (no repeated 8-token span). The existing HTTP `run_g3.py` averages all K targets and changes the qattn window for each truncation. Those are different estimators. The wrapper reports reference-compatible novel-token NLL as `metric=nll`, and retains `metric=nll_all` as an all-token diagnostic. Both summaries use the same `paired_summary` helper and `finals/stats.py` BCa implementation as `run_g3.py`: document weighting, 5000 resamples, seed 20261004. It makes no noninferiority verdict; unmeasured metrics are SKIPPED.

## Skipped-work metadata

Every native document has `ridgefill.n_prompt`, `n_filled`, `n_exact_prefix`, `n_exact_prefix_by_source`, `n_cached_exact_prefix`, `tokens_skipped` and `mac_share_skipped`, plus prefix/tail counts and both analytic MAC totals. `n_filled` excludes exact prefix rows; the existing `filled` field remains the prefix boundary for compatibility. Source counts are disjoint in class/digit/special/header/qattn order, with additional cached exact rows counted separately. The paired wrapper validates every native report and adds per-arm aggregate denominators to each BCa summary; these are sums of prompt coverage and MAC budgets, not averages of percentages.

MACs follow `kva/compute.py` efficient bypass with the measured row positions and actual query window. All prefix rows selected for exact computation pay full layer cost. The model includes attention/GDN/MLP/projector work and one LM head; it excludes other elementwise work, bandwidth, cache savings and extra scoring heads. Query/output gating is read from the actual model architecture; the tiny HF config omits the optional `attn_output_gate` flag, so the independent test derives it from the checkpoint projection shape. `mac_share_skipped` can be negative when selection costs more than it saves. These fields describe compute coverage, not numerical equivalence or measured speedup.

## Chat selector controls

Both the native scorer and `run_nll.py` accept `--special-exact on|off` (default off), `--header-rows K` (default 0), `--qattn-window-mode last|turn` (default last), and `--qattn-turn-cap N` (default 256). The native scorer also accepts `--ridgefill-special-exact` and `--ridgefill-header-rows` aliases. These are experimental controls; no chat quality verdict or release-default change follows from the tiny correctness gate.

Special rows and the K rows following each special row are added to the prefix exact rows through `llama_ridgefill_select_rows_ext`. Header rows are independent of `special-exact`, stop at P, and can overlap. Speed mode refuses either rule when enabled. Off mode stays exact and ignores these controls, allowing paired comparisons with identical arguments.

In `last` mode the window remains `[END-window,END)`. In `turn` mode the driver tokenizes the literal `<|im_start|>user\n` header with special parsing and no BOS, finds the last complete header entirely inside `[0,END)`, and sets `turn_start` to the row after that header. It captures `[max(turn_start,END-cap),END)`. The cap is independent of `--qattn-window`; header rows are excluded. Missing or empty turns are refused by name, and turn mode does not support blind selection. For NLL, END remains N-K, so teacher-forced headers after that endpoint cannot change selection. Windows are sized once per document and remain fixed while scoring all K targets.

To measure a chat arm, append `--special-exact on --header-rows 3 --qattn-window-mode turn --qattn-turn-cap 256` to the paired command below and use its frozen chat document input. Each output records the resolved window and selector options.

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
  --scored 512 --tail 4096 --qattn-window-end known --gpu-layers 99
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

The gate covers all four modes against FP32 `finish()` logits at K=64/T=200 and K=512/T=600 with ubatches 50/128/512. It also compares selector mass and row sets, zero-share versus quality, off versus stock next-token scoring, and changes all targets to prove they do not affect selection or the first predictive distribution. Endpoint controls compare the default with explicit N, compare N and N-32 with the oracle, verify the known/legacy aliases, and reject invalid endpoints. The paired wrapper runs on two documents, including repeated spans, and verifies its explicit N-K endpoint and real BCa summary. Tiny CPU agreement does not establish GPU or full-model quality.

The chat gate needs a pre-chat scorer binary built at d619c13d6 (save it before rebuilding), plus the same fixture. It writes a local copy with ChatML control-token names and keeps the tiny weights and row scores unchanged:

```sh
python3 tools/ridgefill/check_chat_score.py \
  --binary build/bin/llama-ridgefill-score \
  --baseline-binary build/bin/llama-ridgefill-score-before-chat \
  --fixture ../llama.cpp-ridgefill-tools/out/gate1-qwen35 \
  --kva /var/home/dylan/projects/research/kva-wt-q27 \
  --work build/score-chat-gate
```

This gate compares default log-probabilities, logsumexp, mass and rows byte-for-byte on raw and templated inputs; compares chat selection, mass and `finish()` logits across three chunk sizes; covers capped, uncapped and short turns; checks future/partial/missing/empty headers and zero-share equality; and runs the paired wrapper with per-document turn bounds and BCa.

## Experimental chat-row flags

`--special-exact on|off` (default off) keeps added/special token positions exact; `--header-rows K` (default 0) adds up to K following prefix rows after each such token, independently of special-exact. Unknown, control and user-defined tokens are special; unused padding is excluded. The scorer also accepts the common CLI aliases `--ridgefill-special-exact` and `--ridgefill-header-rows`. Both quality modes apply these through the engine's extended selector; speed with either option active refuses by name. Off remains a stock control. Native records include these choices in parameters.

The paired wrapper accepts the same new controls and `--mode off|speed|quality|quality+qattn` (default quality+qattn). It always pairs the selected mode against off; mode off is an identity control. Input remains the existing named-token-ID format. Lane T's frozen CHAT-TUNE/CHAT-HELDOUT schema is pending; no schema is guessed here.

`--qattn-window-mode last|turn` and `--qattn-turn-cap N` accept the corresponding `--ridgefill-*` aliases in both the scorer and wrapper. Turn mode uses the last complete user header before the fixed N-K endpoint, excludes that header, and caps query rows independently of --qattn-window. It requires quality+qattn and refuses missing/empty turns by name. The engine and HTTP caller use the same sweep driver.

Class `--share` accepts [0,1]. Zero disables the class source only. For qattn-only use `--mode quality+qattn --share 0 --digits off`; add `--special-exact on` for qattn+special. Special-only uses `--mode quality --share 0 --digits off --special-exact on`. With no other source and alpha 0, quality selects no exact prefix rows and matches speed mode byte-for-byte on the tiny CPU fixture. Projector metadata may also set `share_default` to zero.

## S1 and vtail orchestration (requires libllama C API v2)

The standalone scorer accepts `--s1 on|off`, `--s1-window W` (256), `--s1-factor F` (1), `--vtail on|off`, and `--vtail-rows C` (512). Both features default off. Use `--s1 on --vtail on --special-exact on` with quality+qattn for the S1+special/vtail arm. S1 and vtail are independently selectable for controls. `--ridgefill-api-version` prints the linked engine API version. A build with pre-v2 headers or a linked API version below 2 refuses either feature by name; it does not substitute Tier 3.

The driver runs LOWER, computes the Tier-3 base count, runs S1_FILL over [0,P), runs S1_PROBE over [END-W,END), selects max(1,round-half-even(F*n_base)) prefix rows by mass (capped at P, ties by position), and unites them with special rows when enabled. Class/digit/qattn rows are not retained. It then sets rows, enables optional vtail, replays LATE over the prefix and tail, disables vtail, and restores FULL. S1 currently requires a positive qattn share and header_rows=0 in the scorer; these match the registered base arm. The driver accepts the caller's base rows and uses `projector` plus `special_exact` for the final special-row union.

For S1/vtail the standalone scorer defaults END to N-K unless explicitly overridden. Serving keeps the driver's prompt-end default. S1 requires its full W-row window to lie in the exact tail. Vtail calibration uses the first min(C,N-P) tail rows, and the scorer refuses calibration that reaches scored NLL targets. It raises the context batch/ubatch capacity as needed and makes the first LATE tail decode large enough. A server caller must provide sufficient context capacity; calibration that would consume its deferred final token is refused. No calibration range is silently shortened.

Outputs include the base count, S1 window and mass (with diagnostics), vtail scales, and separate analytic costs for S1_FILL, S1_PROBE and vtail. S1_FILL charges P rows of late-attention projector plus K/V work even though LATE recomputes them; `prefix_kv_reused` is false. S1_PROBE charges W rows through every late layer against the visible prefix/window. These costs are added to the existing cold-prefill MAC estimate. Final row-source reporting uses `s1`, with class/digit/header/qattn counts zero and special rows counted separately. The paired wrapper and HTTP request/reporting integration remain with L-B; their legacy report validator needs the new S1 source key.

The gate verifies lane T's frozen reference hashes, default equality against a saved pre-S1 binary, mass, row sets, logits, vtail scales, and total MACs including recomputed prefix K/V:

```sh
python3 tools/ridgefill/check_s1_score.py \
  --binary build/bin/llama-ridgefill-score \
  --baseline-binary build/bin/llama-ridgefill-score-before-s1 \
  --fixture ../llama.cpp-ridgefill-tools/out/gate1-qwen35 \
  --reference /var/home/dylan/projects/research/kva-wt-q27/ridgefill-tune/engine-ref/s1-vtail \
  --kva /var/home/dylan/projects/research/kva-wt-q27 \
  --work build/score-s1-gate
```

Without C API v2 the gate checks defaults and named refusals, reports the feature and cost checks as SKIPPED, and exits 77. This is not a passing S1 gate. The release remains on hold.
