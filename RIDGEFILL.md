# RidgeFill in llama.cpp (experimental fork)

RidgeFill makes prefill of long prompts faster by computing most of the prompt only through the first layers of the
model. It is training-free: a small file (the *projector*) holds one linear map per later layer, fitted once by ridge
regression on the model's own hidden states.

How a prompt of N tokens is processed with split layer S and tail T:

- the last T tokens (the *tail*) run through every layer, exactly as without RidgeFill;
- the other N - T tokens run layers 0 .. S-1 only. For every later layer, the projector predicts that layer's input
  from the hidden state at layer S, and only the K/V cache of that layer is computed from the prediction (through the
  layer's own input norm, K/V projections and RoPE). Attention and MLP of those layers are skipped for these tokens;
- generation after the prompt is exact and attends to the filled cache.

The filled part of the cache is approximate. The output is not identical to normal llama.cpp: expect some quality
loss, more so when the tail is short relative to the prompt (numbers below). With RidgeFill off, this fork is
bit-identical to upstream llama.cpp (checked, see "Tests").

RidgeFill is by Dylan Johnston and Robert P Smith. Technical note: [doi:10.5281/zenodo.23179168](https://doi.org/10.5281/zenodo.23179168). The projector is licensed under Apache-2.0; the code follows [this repository's license](LICENSE).

## Qwen3.8-27B (`qwen35`)

The Qwen3.8-27B port supports speed, quality and quality+qattn prefill. The recommended E3 configuration uses quality+qattn with S1 attention-based row selection, V rescaling and the lexical net. S1/vtail supports the `qwen35` architecture only. The later Llama/Qwen3 tables are earlier measurements on different models.

### Quick start (Qwen3.8-27B)

Run these commands from a checkout of this fork. With the Hugging Face `hf` CLI installed, download the model and projector; each command prints the downloaded file path:

```sh
MODEL=$(hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502)
PROJECTOR=$(hf download TheCacheIsALie/RidgeFill-Projector-Qwen3.8-27B-i8 ridgefill-projector-qwen3.8-27b-i8-S16.gguf)
printf '%s  %s\n' 8646d41a5a28d60ef53cc5e1fee4b2bfb20d0775db88d3f89d0731ad9f6b629d "$PROJECTOR" | sha256sum -c -
```

Build with CMake using CUDA (requires the CUDA toolkit) or Vulkan (requires the Vulkan SDK); see [build dependencies](docs/build.md):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
# For Vulkan, use this configure command instead:
# cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build --config Release -j --target llama-server
```

Start the server with E3 settings (thinking off, F16 KV, one slot):

```sh
./build/bin/llama-server -m "$MODEL" -c 131072 -ngl 99 -fa on \
    -ctk f16 -ctv f16 -b 2048 -ub 512 --parallel 1 \
    --cache-reuse 0 --cache-ram 0 --port 8080 \
    --chat-template-kwargs '{"enable_thinking":false}' \
    --ridgefill "$PROJECTOR" \
    --ridgefill-mode quality+qattn \
    --ridgefill-tail 4096 \
    --ridgefill-min-prompt 8192 \
    --ridgefill-share 0.5 \
    --ridgefill-digits on \
    --ridgefill-special-exact on \
    --ridgefill-alpha 0 \
    --ridgefill-qattn-share 0.2 \
    --ridgefill-qattn-window 64 \
    --ridgefill-header-rows 0 \
    --ridgefill-qattn-window-mode last \
    --ridgefill-qattn-turn-cap 256 \
    --ridgefill-s1-window 256 \
    --ridgefill-s1-factor 1.0 \
    --ridgefill-vtail on \
    --ridgefill-vtail-rows 512 \
    --ridgefill-lexical on \
    --ridgefill-lexical-window 64 \
    --ridgefill-lexical-min-words 4 \
    --ridgefill-lexical-span-rows 80 \
    --ridgefill-lexical-max-occurrences 2 \
    --ridgefill-trim-kv on
```

For lower VRAM use, add `--ridgefill-stream-projector on` to the server command. The server listens at `http://127.0.0.1:8080`; use `/v1/chat/completions` with an OpenAI-compatible client. Prompts shorter than 8192 tokens run exact. See [measured TTFT, VRAM and quality](#measured-qwen38-27b-results) before choosing a context size.

### Modes and row selection

The source projector uses split 16 of 64 layers, exact tail 4096, minimum tail 512, and minimum prompt 8192. Defaults come from the projector: alpha 0, class share 0.5, digits on, question-attention share 0.2. A format-2 GGUF checks every model fingerprint field and the vocabulary token hash; the tokenizer JSON hash and source file hashes are retained as provenance. A different quantization of the same base model is allowed.

| mode | exact prefix rows R | availability |
|---|---|---|
| `off` | all prompt rows run stock | no approximate prefill |
| `speed` | empty | row-selector flags do not add exact rows |
| `quality` | class and digit rows, plus enabled special/header rows | alpha 0 |
| `quality+qattn` | quality rows plus question-attention rows; S1 and lexical selection when enabled | server/scorer; alpha 0; bench/perplexity refuse this mode |

For a prompt of N tokens and exact tail T, P=max(0,N-T) is the prefix length. Selection is per prompt, not per prefill chunk:

- Class rows qualify when the token's vocabulary score is finite. Keep `rint(share * qualifying_rows)` positions with highest scores, ties by lower position. Rounding is half to even; the default class share is 0.5. Class share 0 disables only class rows; digit, special, header and question-attention rows still apply. With all row sources disabled and alpha 0, empty quality selection is byte-identical to speed mode on the tiny CPU fixture.
- Digit rows contain an ASCII digit in their vocabulary token text. Unknown, control, user-defined and unused tokens are excluded. Digits are on by default. Quality uses the sorted union of class and digit positions below P.
- Question-attention adds `round(qattn_share * P)` prefix positions with highest causal attention mass, ties by lower position and half-even budget rounding. The default share is 20%, using the last 64 prompt rows as queries at attention layers 3, 7, 11 and 15 below split 16. Mass sums causal `softmax(Q K^T / sqrt(head_dim))` over these layers, heads and queries. The union with class/digit rows can exceed 20% of P. `--ridgefill-qattn-blind` places the query window immediately before P instead of at the prompt end. Share 0 runs ordinary quality with identical rows and no sweep allocation.

Positive-share question-attention runs lower layers over the whole prompt first, saves h_S on the backend, selects rows, then replays the late prefix and exact tail. It never transfers the full h_S through host memory. The server yields between sweep steps; slot cancellation and shutdown release the buffers while the context is alive. Question-dependent requests recompute their prompt rather than reuse a stale question-selected prefix. L-C's tiny split-4 native G1 compared 25 cache/state/logit arrays at three chunk sizes (worst relative error 6.64e-7, threshold 1e-4); its HTTP gate passed 17 comparisons, cancellation/reuse and shutdown controls. These are CPU fixture checks, not 27B measurements.

The server stores h_S as F16, with a checked 2 GiB maximum per sweep buffer: `N * n_embd * sizeof(element) <= 2147483648`. At width 5120, F16 admits at most 209715 rows; 209716 refuses before allocating h_S. A 32768-row h_S is 320 MiB, and a 131072-row buffer is 1280 MiB. F32/BF16 buffers are available through the engine API, with the same byte bound (F32 admits 104857 rows). Callers may impose a smaller byte limit. Window captures, attention scratch space, projector and model/KV buffers add memory outside that h_S bound. The host attention-mask cache has its own 1 GiB bound.

Speed mode writes attention KV and GDN convolution/recurrent state from each predicted layer input. Quality mode mixes predicted bulk inputs with the carried exact-row residuals, then restricts the remaining late-layer work to exact rows. The terminal GDN correction is applied once at the prefix boundary in speed mode. Quality needs scaling by the gate-decayed bulk share; this rho-scaled correction is pending, so nonzero quality alpha is refused. The source package already stores corrections as `[heads, V, K]`; GGUF conversion reorders value heads into llama.cpp order without transposing K/V again.

### Measured Qwen3.8-27B results

| Setup | Value |
|---|---|
| Source | `ridgefill-results-compendium-20261026/e3-27b/RESULTS-E3.md`, 2026-10-26 |
| Hardware | 4x NVIDIA A100 PCIe 40 GB; LongBench on a separate machine with the same GPU model |
| Backend | CUDA, sm_80 |
| Stock | upstream llama.cpp `1242be66f` |
| RidgeFill | E3, final build; runtime `b0439f7bb`, scorer `b70b0c5b4` |
| Model | `Qwen3.8-27B-UD-Q4_K_XL.gguf`, revision `4ca720788d1e01f1bff70c033e0d0028fd02e502` |
| Runtime | `-b 2048 -ub 512`, F16 KV, flash attention on, cache reuse off |
| Confidence intervals | 95% bootstrap, 10,000 resamples, seed 1; paired items for RULER/LongBench, documents for KL/dNLL |

#### TTFT and VRAM (warm, median of 3, same GPU)

| Context | Stock s | E3 GPU s | E3 GPU speedup | E3 streamed s | E3 streamed speedup | VRAM over stock, GPU / streamed (MiB) |
|---|---|---|---|---|---|---|
| 16,384 | 12.97 | 10.38 | 1.249 | 10.88 | 1.192 | 1592 / 268 |
| 32,768 | 27.49 | 21.11 | 1.302 | 21.69 | 1.267 | 1802 / 478 |
| 65,536 | 60.50 | 46.28 | 1.307 | 46.55 | 1.300 | 2270 / 944 |
| 120,000 | 128.10 | 100.01 | 1.281 | 99.06 | 1.293 | 3014 / 1732 |

#### RULER (official budgets)

| Context | n | Stock % [CI] | E3 % [CI] | Delta points [CI] |
|---|---|---|---|---|
| 16,384 | 260 | 85.69 [83.15, 88.04] | 84.51 [81.87, 87.00] | -1.19 [-3.13, +0.79] |
| 32,768 | 260 | 87.85 [85.46, 90.04] | 86.42 [83.88, 88.81] | -1.42 [-3.21, +0.35] |

#### LongBench v2 (official CoT, full length)

| Group | n | Stock % | E3 % | Delta points |
|---|---|---|---|---|
| overall | 503 | 38.57 | 37.18 | -1.39 [-5.17, +2.19] |
| easy | 192 | 42.71 | 40.62 | -2.08 |
| hard | 311 | 36.01 | 35.05 | -0.96 |
| short | 180 | 46.11 | 44.44 | -1.67 |
| medium | 215 | 38.14 | 36.74 | -1.40 |
| long | 108 | 26.85 | 25.93 | -0.93 |
| Single-Document QA | 175 | 44.00 | 42.29 | -1.71 |
| Multi-Document QA | 125 | 30.40 | 34.40 | +4.00 |
| Long In-context Learning | 81 | 38.27 | 30.86 | -7.41 |
| Long-dialogue History | 39 | 48.72 | 41.03 | -7.69 |
| Code Repository | 50 | 34.00 | 34.00 | 0.00 |
| Long Structured Data | 33 | 36.36 | 36.36 | 0.00 |

#### KL and dNLL on post-prompt tokens (100 documents per length, 51,200 positions)

| Context | Mean KL [CI] | Median KL | p99 KL | Max KL | Top-1 agreement % | Mean dNLL [CI] | Median dNLL |
|---|---|---|---|---|---|---|---|
| 16,384 | 0.0894 [0.0399, 0.1556] | 0.0040 | 0.965 | 35.2 | 94.64 | -0.0330 [-0.0875, +0.0059] | +0.0044 |
| 32,768 | 0.2656 [0.0599, 0.6017] | 0.0042 | 12.77 | 33.0 | 93.60 | -0.2142 [-0.5629, -0.0025] | +0.0071 |

#### Multi-user (16K new request among C users; E3 streamed; 2 repetitions; one GPU)

| Schedule | New request TTFT s, stock / E3 | Other users tok/s, stock / E3 | Other users ITL mean ms | Other users ITL p99 ms | Other users ITL max ms |
|---|---|---|---|---|---|
| unified C1 | 13.06 / 11.11 | - | - | - | - |
| unified C2 | 13.53 / 11.70 | 19.92 / 21.60 | 49.33 / 45.50 | 1605.7 / 863.0 | 1707.8 / 1317.1 |
| unified C4 | 14.18 / 12.22 | 60.06 / 62.96 | 46.47 / 44.20 | 148.0 / 669.0 | 1785.0 / 1323.7 |
| unified C8 | 15.32 / 13.53 | 86.90 / 89.48 | 71.58 / 69.35 | 621.7 / 774.9 | 1919.6 / 1876.1 |
| separate KV C2 | 13.48 / 11.34 | 19.91 / 21.87 | 49.38 / 44.90 | 1592.2 / 858.3 | 1703.4 / 1310.7 |
| separate KV C8 | 13.93 / 11.86 | 88.88 / 91.98 | 69.84 / 67.46 | 584.7 / 654.3 | 1821.1 / 1809.6 |

### Projector conversion

Download the projector from [TheCacheIsALie/RidgeFill-Projector-Qwen3.8-27B-i8](https://huggingface.co/TheCacheIsALie/RidgeFill-Projector-Qwen3.8-27B-i8). Projector: Dylan Johnston and Robert P Smith, Apache-2.0, DOI `10.5281/zenodo.23179168`. The converted artifact is `ridgefill-projector-qwen3.8-27b-i8-S16.gguf`: Q8_0 maps, F32 biases/corrections/row scores, 1452188032 bytes, SHA256 `8646d41a5a28d60ef53cc5e1fee4b2bfb20d0775db88d3f89d0731ad9f6b629d`.

```sh
hf download TheCacheIsALie/RidgeFill-Projector-Qwen3.8-27B-i8 ridgefill-projector-qwen3.8-27b-i8-S16.gguf --local-dir projector

# Or convert the verified int8 source folder, with NumPy and safetensors installed:
python3 convert_ridgefill_to_gguf.py projector-source/ --model-gguf model.gguf --outfile projector.gguf
```

Folder conversion needs no torch and defaults to Q8_0 maps. `--outtype f32|f16|bf16|q8_0` changes the maps; biases, row scores and corrections remain F32. The converter verifies SHA256SUMS and manifest file checksums before writing, and computes `adapter.ridgefill.vocab_sha256` from the base model GGUF's token strings, each followed by one NUL byte. A different tokenizer or model fingerprint is refused by name.

### Skipped-token and analytic MAC reporting

Native scorer records carry a `ridgefill` object. The server puts the same object in per-request `timings.ridgefill` and completion/chat/Responses usage when a projector is loaded, including mode off. Needle, NLL and TTFT records retain it; paired quality/performance summaries include both arms' aggregate counts and fractions. Bench prints one `RidgeFill: report {JSON}` line per measured repetition after stopping its timer, plus separately labelled warmup reports. Perplexity logs it for each scored chunk.

The fields are `n_prompt`, `n_prefix`, `n_filled`, `n_exact_prefix`, `n_exact_tail`, `n_exact_prefix_by_source`, `n_cached_exact_prefix`, `tokens_skipped`, `macs_exact`, `macs_ridgefill` and `mac_share_skipped`. Filled rows exclude every prefix row computed exactly: `n_filled = n_prefix - n_exact_prefix`, and `tokens_skipped = n_filled / n_prompt` is a fraction. The source counts give disjoint contributions in class, digit, special, header, qattn order. Cached rows that are exact without belonging to one of those selector sets contribute to `n_cached_exact_prefix`; source counts plus that count equal `n_exact_prefix`. Ordinary cached approximate rows still count as filled prompt coverage, while a fully exact cache reports zero filled rows even if the current request enables speed/quality. The server's existing processed/cached timing counters describe work performed on the current request. The legacy native `filled` and server `prompt_filled_n` fields retain their prefix-span meanings; use the new object for skipped-token accounting.

The MAC fraction is `1 - macs_ridgefill / macs_exact`, using `kva/compute.py`'s efficient bypass model: linear projections, MLPs, causal attention, GDN convolution/state work, a full projector only for filled rows, and one LM head. Exact rows pay full late-layer costs at their actual positions. Qattn pays its query projections and QK scores at the actual window/layers, including a capped user turn. Qwen35's doubled query/output-gate projection is included. This is an analytic cold-prefill model, excluding cache savings, norms, RoPE, softmax, bandwidth, extra scoring heads and other elementwise work; it is not a measured speedup. Selector overhead can make MAC savings negative even when no rows are filled. Empty prefills report zero fractions. Aggregates divide summed filled rows by summed prompt rows and summed MAC savings by summed baseline MACs. Statistical quality estimates and their BCa weighting are unchanged.

A standalone stock server does not expose model MAC totals. The client labels that exact off control, records both skipped fractions as zero and leaves its MAC totals null. Active RidgeFill modes require a matching reporting-capable server; missing/inconsistent reports fail by name. The reporting gates use CPU tiny fixtures and establish no new full-model quality or speed result.

### Options and evaluation

Unsupported LoRA, control vectors, embeddings, nonzero --cache-reuse and --split-mode row combinations are rejected during argument validation before model or backend loading. Library-side guards remain active.

With no projector, processing is stock. `--ridgefill FILE` defaults to speed mode; `--ridgefill-mode off` keeps the projector loaded and requests stock processing. Mode availability is listed above; unsupported combinations below refuse by name.

| option | default | meaning |
|---|---|---|
| `--ridgefill FILE` | off | projector GGUF |
| `--ridgefill-tail N` | projector (27B: 4096) | last prompt tokens computed exactly; at least projector tail_min (27B: 512) |
| `--ridgefill-min-prompt N` | 8192 | shorter prompts are not filled; CLI/server/perplexity only |
| `--ridgefill-mode MODE` | speed with a projector | off, speed, quality, quality+qattn; availability above |
| `--ridgefill-stream-projector on\|off` | off | quality+qattn server/scorer: host maps, two staging maps per device, layer-major late prefix; alpha 0 only |
| `--ridgefill-trim-kv on\|off` | on | reuse S1 bulk K/V in the late sweep |
| `--ridgefill-share N` | projector (27B: 0.5) | per-prompt class-row share in [0,1] |
| `--ridgefill-digits on\|off` | projector (27B: on) | add digit rows in quality |
| `--ridgefill-lexical on\|off` | off | quality+qattn server/scorer: reserve rare question-matching word phrases and their containing units, within a 5% prefix cap |
| `--ridgefill-lexical-window N` | 64 | Last content-token window, clipped to the exact tail and selector endpoint (1..4096) |
| `--ridgefill-lexical-min-words N` | 4 | Minimum consecutive normalized words in one unit (2..4096) |
| `--ridgefill-lexical-span-rows N` | 80 | Maximum rows from the containing unit's start (1..4096) |
| `--ridgefill-lexical-max-occurrences N` | 2 | Maximum prefix occurrences of a rare unigram anchor (1..4096) |
| `--ridgefill-special-exact on\|off` | off | keep unknown/control/user-defined token rows exact in quality modes |
| `--ridgefill-header-rows K` | 0 | keep the next K prefix rows after each special token exact, independently of special-exact |
| `--ridgefill-qattn-window-mode last\|turn` | last | last uses the fixed query window; turn uses rows after the last complete user header |
| `--qattn-turn-cap N` (alias `--ridgefill-qattn-turn-cap`) | 256 | maximum query rows in turn mode; independent of --ridgefill-qattn-window |
| `--ridgefill-s1-window W` | 0 (off) | experimental late-layer attention-mass probe; use 256 when enabled; requires API v2 and the two-sweep driver |
| `--ridgefill-s1-factor F` | 1.0 | finite nonnegative multiplier for the S1 base-row budget |
| `--ridgefill-vtail on\|off` | off | experimental bulk-prefix V rescale from the first exact tail rows; requires two sweeps |
| `--ridgefill-vtail-rows C` | 512 | positive sample count, capped to the exact tail; first tail ubatch must hold all sampled rows |
| `--ridgefill-alpha N` | projector (27B: 0) | state correction scale in [0,1]; quality currently requires 0 |
| `--ridgefill-qattn-share N` | projector (27B: 0.2) | question-attention prefix share in [0,1]; server only |
| `--ridgefill-qattn-window N` | 64 | positive query-window size in last mode only; server only |
| `--ridgefill-qattn-blind` | off | use queries immediately before the prefix end; server only |

```sh
llama-server -m model.gguf -c 32768 -ngl 99 --ridgefill projector.gguf --ridgefill-mode speed --ridgefill-tail 4096
```

In experimental turn mode (`--qattn-window-mode turn`, alias `--ridgefill-qattn-window-mode`), the driver parses the literal `<|im_start|>user\n` with special tokens enabled and no BOS. It finds the last complete header entirely before END, excludes the header, and captures `[max(turn_start, END-cap), END)`. END is the prompt end for HTTP and N-K for native NLL, which excludes scored targets from selection. Missing/empty user turns and turn plus blind selection refuse by name. The E3 measurements above use last-window mode; turn-mode quality is not measured there. Defaults remain last/64 with special-exact off and header-rows 0.

Quality settings are `--ridgefill-mode quality --ridgefill-share 0.5 --ridgefill-digits on --ridgefill-alpha 0`. For question-attention, use `--ridgefill-mode quality+qattn --ridgefill-qattn-share 0.2 --ridgefill-qattn-window 64 --ridgefill-alpha 0` on llama-server. For the recommended E3 configuration, use the complete [quick-start command](#quick-start-qwen38-27b), including S1, vtail and lexical settings.

| per-request field | effect |
|---|---|
| `ridgefill` | boolean; false selects stock processing regardless of mode |
| `ridgefill_mode` | off, speed, quality or quality+qattn; inherits server mode when omitted |
| `ridgefill_share` | finite number in [0,1]; inherits the startup class share |
| `ridgefill_tail` | integer; 0 inherits the server tail, positive values must meet projector tail_min |
| `ridgefill_s1_window` | nonnegative integer; 0 disables S1; omitted/null inherits startup setting |
| `ridgefill_s1_factor` | finite nonnegative number; omitted/null inherits startup factor |
| `ridgefill_vtail` | boolean; omitted/null inherits startup setting |
| `ridgefill_vtail_rows` | positive integer; omitted/null inherits startup sample count |
| `ridgefill_special_exact` | boolean; inherits startup setting; quality modes only |
| `ridgefill_header_rows` | nonnegative integer; inherits startup setting; quality modes only |
| `ridgefill_qattn_window_mode` | last or turn; inherits startup setting |
| `ridgefill_qattn_turn_cap` | positive integer; inherits startup turn cap (256) |

Digit/alpha settings and question share/window-size/blind are server startup settings. Class share can also be overridden with `ridgefill_share`. The chat-row and question-window mode/cap fields above can be overridden per request. `ridgefill_mode: "off"` selects stock. Unknown modes, wrong field types, negative tails and tails below tail_min return a named request error.

The RAM prompt cache carries the mode, exact-row set and fill marker. The unchanged fill boundary can reuse its exact cached tail. Changing mode or selected rows in a retained approximate region requires recomputation; a changed boundary restarts a recurrent/hybrid context when it cannot rewind safely. Ordinary cache_prompt reuse and --cache-ram remain available. KV shifting through nonzero --cache-reuse or request n_cache_reuse is refused by name. Slot save of filled rows is refused; an exact slot can be saved.

The following combinations and invalid inputs refuse by name; there is no fallback to a different RidgeFill mode:

| request or setting | refusal / supported alternative |
|---|---|
| LoRA (`--lora`), control vectors (`--control-vector`), embeddings/reranking (`--embeddings`) | `RidgeFill does not support` the named setting; model-enabled embeddings also refuse |
| nonzero `--cache-reuse` or request `n_cache_reuse` | names the cache-shift setting; set it to 0; ordinary `cache_prompt` and RAM prompt cache remain available |
| `-sm row` | names `--split-mode row`; use layer split |
| save a slot containing filled rows | names `save slot`; an exact slot can be saved |
| speed with special-exact on or header-rows > 0 | names the chat flags; use a quality mode |
| question-window-mode turn | requires quality+qattn; refuses blind selection and missing or empty user turns by name |
| alpha != 0 in either quality mode | names the missing rho-scaled correction; use `--ridgefill-alpha 0` |
| single-pass parallel quality with non-unified KV | names `--kv-unified`; the server/scorer two-sweep flow supports separate KV streams |
| bench/perplexity `quality+qattn` | names the server question-attention sweep; use the server or ordinary quality |
| positive-share server qattn with shared/parent/child prompts, speculative decoding, LoRA or a non-completion task | names the requirement for an independent text completion without speculative decoding or LoRA |
| qattn replay with recurrent rollback (`n_rs_seq > 0`) | names recurrent rollback snapshots; disable that combination |
| qattn split without lower attention layers, or a requested layer that is recurrent | names the split/layer; the 27B split-16 projector has four usable layers |
| h_S exceeding 2 GiB, invalid shape/type, or failed backend allocation | names the sweep buffer dimensions/type/byte limit/allocation; no oversized allocation or host fallback |
| filled prefix token requesting logits | names the filled token/position; logits must come from the exact tail |
| invalid mode, nonfinite/out-of-range shares or alpha, nonpositive window/tail, tail below projector minimum | names the invalid mode/flag/default; class share is [0,1], qattn share/alpha are [0,1] |
| wrong projector architecture/format/dimensions/layer kinds, model metadata or vocabulary hash | names the mismatched field; qwen35 requires format 2 and the matching model/tokenizer |
| missing/malformed projector metadata, unknown `model_meta` keys, unknown/malformed tensors | names the key/tensor/shape/type; source checksum failures also stop conversion |
| malformed replay API calls | names the missing projector, qwen35-only/non-causal context, invalid capture layer/buffer, sequence/order/batch mismatch or incomplete LOWER/LATE transition |

Startup combination guards apply even when a loaded projector is in mode off. Short prompts (`N < min-prompt`) and `T >= N` run stock. Decode remains stock. Multimodal chunks do not fill; this text-only port has no multimodal RidgeFill validation.

Perplexity and bench support off, speed and quality at alpha 0. Both still refuse quality+qattn by name. Perplexity fills at most the unscored first half of each chunk, so its effective tail is `max(T, n_ctx/2)`; it selects quality rows separately for each chunk, including its actual BOS token. Parallel quality chunks require --kv-unified (-kvu). Failed decoding and a missing/nonfinite perplexity score exit nonzero. Bench selects rows from its full generated prompt and logs the exact-prefix row count. Startup and per-request quality with nonzero alpha refuse by name until the rho-scaled correction is implemented. Tiny CPU integration checks cover actual quality runs and these refusals, without establishing real-model quality or performance.

For reference-compatible NLL, use [llama-ridgefill-score and its paired wrapper](tools/ridgefill/README.md). The HTTP client below scores all targets and is a different estimator; repeated question sweeps make it unsuitable for the measured Tier 3 NLL run. The paired client and CPU gates are documented in [tests/ridgefill/README.md](tests/ridgefill/README.md). `run_g3.py --mode ... --docs ... --needles ... --out ...` runs NLL, needles, TTFT and native bench against one build's mode-off arm. NLL uses only the last 512 teacher-forced targets and keeps the approximate prefix fixed; it requires T-512 >= tail_min and N-512 >= min-prompt. Needle evaluation removes appended answer IDs, applies the loaded Qwen chat template with `enable_thinking=false`, generates greedily, and checks seven digits at digit boundaries. Timing runs alternate off and candidate for at least three repeats. Paired 95% BCa intervals use the oracle's `finals/stats.py`; an inactive eligible fill or missing probability fails the run. Tiny instrumentation checks are not real-model quality/speed measurements.

`run_g5.py --ref COMMIT --backend cpu|vulkan|hip --fixture ... --out ...` audits and builds a clean git archive, then tests named refusals and exact/filled slot-save controls. It uses the caller's environment and does not provide container isolation. The host wrappers `run_g5_gpu.sh` and `run_g5_two_card.sh` add single-GPU archive builds/27B speed smokes and the separate two-GPU layer-split smoke. See [GPU G5 commands](tests/ridgefill/README.md#gpu-g5-commands). They require real device enumeration, model tensors on every selected GPU and a completed filled prefix. Container isolation and numerical/performance validation remain separate; the CPU archive result does not establish them.

## Earlier measured results: dense Llama and Qwen3

Llama-3.1-8B-Instruct (unsloth, revision 4699cc75), projector split 12 of 32 (bf16 maps), one NVIDIA RTX 4090 24 GB,
CUDA 12.4, flash attention on, default batch sizes (`-b 2048 -ub 512`), F16 KV cache. Branch commit: see
`git log` (measured on `fa8061c4b`, upstream base `de7fa0a3c`). Text: wikitext-2 test set (public, 289k tokens).
These are development numbers, not paper numbers.

### Speed (tail T = 4096)

Prompt processing, `llama-bench -p N -n 0 -r 3`, tokens/s (stock -> RidgeFill):

| model | 8K | 16K | 32K |
|---|---|---|---|
| F16    | 10219 -> 12861 (**1.26x**) | 9080 -> 13743 (**1.51x**) | 7364 -> 12794 (**1.74x**) |
| Q8_0   | 10928 -> 13794 (1.26x) | 9632 -> 14610 (1.52x) | 7748 -> 13474 (1.74x) |
| Q4_K_M | 10744 -> 13571 (1.26x) | 9459 -> 14358 (1.52x) | 7666 -> 13293 (1.73x) |
| analytic ceiling (share of prefill multiply-adds skipped) | 1.35x (26%) | 1.67x (40%) | 1.93x (48%) |

Time to first token through `llama-server` (prompt + 1 token, no prompt cache, mean of 3 prompts), seconds:

| model | 8K | 16K | 32K |
|---|---|---|---|
| F16    | 0.81 -> 0.64 | 1.83 -> 1.19 | 4.50 -> 2.57 |
| Q8_0   | 0.75 -> 0.60 | 1.70 -> 1.13 | 4.24 -> 2.44 |
| Q4_K_M | 0.77 -> 0.60 | 1.73 -> 1.14 | 4.30 -> 2.47 |

Generation speed (tg128) is unchanged: F16 61.7, Q8_0 108.3, Q4_K_M 172.1-172.4 tokens/s with and without RidgeFill.

### Quality

**Tail NLL** (T = 4096): for every window of N tokens, mean negative log-likelihood of its last 4095 tokens (all of
them in the exact tail), stock -> RidgeFill. Lower is better; perplexity = exp(NLL).

| model | 8K (35 windows) | 16K (17) | 32K (8) |
|---|---|---|---|
| F16    | 1.821 -> 2.022 (ppl 6.18 -> 7.55) | 1.786 -> 2.050 (5.96 -> 7.77) | 1.817 -> 2.725 (6.16 -> 15.25) |
| Q8_0   | 1.821 -> 2.023 (6.18 -> 7.56) | 1.786 -> 2.054 (5.97 -> 7.80) | 1.818 -> 2.714 (6.16 -> 15.10) |
| Q4_K_M | 1.846 -> 2.031 (6.33 -> 7.62) | 1.811 -> 2.047 (6.12 -> 7.75) | 1.843 -> 2.681 (6.32 -> 14.60) |

**llama-perplexity** (`-c N`, standard scoring of the second half of each chunk). RidgeFill never fills a scored
token here, so the effective tail is max(T, N/2): 4096 at 8K, 8192 at 16K, 16384 at 32K.

| model | -c 8192 | -c 16384 | -c 32768 |
|---|---|---|---|
| F16    | 6.1775 -> 7.5485 | 5.8588 -> 7.0083 | 6.7350 -> 7.8851 |
| Q8_0   | 6.1777 -> 7.5617 | 5.8605 -> 7.0212 | 6.7374 -> 7.9055 |
| Q4_K_M | 6.3350 -> 7.6229 | 6.0108 -> 7.0517 | 6.9105 -> 7.9416 |

**Quantized models still benefit.** The maps were fitted on bf16 activations, but the loss from RidgeFill is about the
same on Q8_0 and Q4_K_M as on F16 (relative perplexity increase at 8K: F16 +22.2 %, Q8_0 +22.4 %, Q4_K_M +20.3 %),
and the speedup is the same.

**Q8_0 projector** (half the size of bf16): matches the bf16 projector within noise (perplexity F16 7.5446 vs 7.5485 at
8K, 7.0046 vs 7.0083 at 16K, 7.8886 vs 7.8851 at 32K; tail NLL within 0.01 nats).

**Needle in a haystack** (built from public text: wikitext-2 test as the haystack, a synthetic "special magic number
for <city>" sentence as the needle, depths 10-90 %, 4 needles per depth and length, Llama 3.1 chat format, greedy;
`llama-server`, T = 4096), needles found:

| model | 8K | 16K | 32K | needles inside the filled region |
|---|---|---|---|---|
| F16 stock      | 20/20 | 20/20 | 20/20 | - |
| F16 RidgeFill  | 19/20 | 15/20 | 15/20 | 35/44 |
| Q4_K_M stock     | 20/20 | 20/20 | 20/20 | - |
| Q4_K_M RidgeFill | 18/20 | 17/20 | 14/20 | 34/44 |

Retrieval of facts from the filled part of a long prompt is where RidgeFill loses most: about 1 in 5 needles there
is missed.

**Longer tails** trade speed for quality. Tail NLL of the same 4095 scored tokens, and prompt-processing speed
(`llama-bench`), stock and T = 4096 / 8192 / 16384:

| model, prompt | stock | T = 4096 | T = 8192 | T = 16384 |
|---|---|---|---|---|
| F16, 16K: NLL / tok/s    | 1.786 / 9080 | 2.050 / 13743 (1.51x) | 1.919 / 11359 (1.25x) | (nothing filled) |
| F16, 32K: NLL / tok/s    | 1.817 / 7364 | 2.725 / 12794 (1.74x) | 2.063 / 11134 (1.51x) | 1.928 / 9036 (1.23x) |
| Q4_K_M, 16K: NLL / tok/s | 1.811 / 9459 | 2.047 / 14358 (1.52x) | 1.927 / 12033 (1.27x) | (nothing filled) |
| Q4_K_M, 32K: NLL / tok/s | 1.843 / 7666 | 2.681 / 13293 (1.73x) | 2.050 / 11657 (1.52x) | 1.938 / 9453 (1.23x) |

A tail of about a quarter of the prompt gives a similar loss (+0.2-0.26 nats) at 8K, 16K and 32K, with a 1.25-1.5x
faster prefill.

### Qwen3-8B (partial run)

Qwen3-8B (Qwen, revision b968826d), projector split 13 of 36, same GPU and settings, T = 4096. The run was stopped
before the Qwen3 speed and most tail-NLL measurements.

- Agreement with the PyTorch reference (BF16 GGUF, last 256 positions): KL 0.0016 at 8K (floor 0.0008), 0.0018 at 16K
  (floor 0.0010), passes; the wrong-map control fails as required.
- llama-perplexity (effective tail max(T, N/2), as above): F16 8.1460 -> 9.2040 (-c 8192), 7.8723 -> 8.5907 (-c 16384);
  Q4_K_M 8.2910 -> 9.3850, 8.0030 -> 8.8144, 8.6749 -> 9.2728 (-c 32768). The Q8_0 projector matches bf16
  (9.1869, 8.5772; Q4_K_M 9.3602, 8.8038, 9.2676).
- Tail NLL, Q4_K_M, 8K (35 windows): 2.112 -> 2.236 (ppl 8.27 -> 9.36).

The loss is smaller than on Llama-3.1-8B (perplexity +13 % / +9 % at 8K / 16K vs +22 % / +20 %).

**What this means in practice.** RidgeFill trades quality for prefill speed, and the trade gets worse as the filled
share of the prompt grows. At 8K-16K with T = 4096 the loss is moderate (about +0.2-0.26 nats on the tail); at 32K
with T = 4096 it is large. For long prompts use a longer tail, about a quarter of the prompt (for 32K prompts
`--ridgefill-tail 8192`); see "Longer tails". This is a property of the method
and of the fitted maps, not of this port: the PyTorch reference implementation shows the same behaviour on the same
prompts (see "Tests").

## Build

Exactly as upstream llama.cpp ([docs/build.md](docs/build.md)), for example:

```sh
cmake -B build -DGGML_CUDA=ON        # or no flag for CPU only
cmake --build build --config Release -j
```

RidgeFill uses only standard ggml ops (matrix multiplication, add, RMS norm, RoPE, KV-cache writes), so it should run
on every backend. **Tested: CPU, CUDA and Vulkan.** The final Qwen3.8-27B tables above use CUDA; earlier HIP native speed-mode measurements also exist. Metal and SYCL are untested.

## Get a projector

A projector belongs to one model (architecture, layer count, hidden size are checked when it is loaded) and one split.

Convert a projector from the kva safetensors format (`layer.<i>` tensors, `[W | bias]`, plus `split` in the metadata):

```sh
python convert_ridgefill_to_gguf.py llama-3.1-8b-instruct-sweep-s12.safetensors \
    --base /path/to/Llama-3.1-8B-Instruct --outtype bf16     # or q8_0 (half the size, same quality), f16, f32
```

`--base` is the Hugging Face model directory the projector was fitted for (only its `config.json` is read). The GGUF
records the split, the layer count, the hidden size and the sha256 of the source file.

Projectors:

| model | split | file | size | availability |
|---|---|---|---|---|
| Qwen3.8-27B | 16 of 64 | `ridgefill-projector-qwen3.8-27b-i8-S16.gguf` | 1452188032 bytes | [TheCacheIsALie/RidgeFill-Projector-Qwen3.8-27B-i8](https://huggingface.co/TheCacheIsALie/RidgeFill-Projector-Qwen3.8-27B-i8) |
| Llama-3.1-8B-Instruct | 12 of 32 | `llama-3.1-8b-instruct-s12-bf16.gguf` | 671 MB | [TheCacheIsALie/Llama-3.1-8B-Instruct-RidgeFill-Projector](https://huggingface.co/TheCacheIsALie/Llama-3.1-8B-Instruct-RidgeFill-Projector) |
| Llama-3.1-8B-Instruct | 12 of 32 | `llama-3.1-8b-instruct-s12-q8_0.gguf` | 357 MB | [TheCacheIsALie/Llama-3.1-8B-Instruct-RidgeFill-Projector](https://huggingface.co/TheCacheIsALie/Llama-3.1-8B-Instruct-RidgeFill-Projector) |
| Qwen3-8B | 13 of 36 | `qwen3-8b-s13-bf16.gguf` | 772 MB | [TheCacheIsALie/RidgeFill-Projector-Qwen3-8B](https://huggingface.co/TheCacheIsALie/RidgeFill-Projector-Qwen3-8B) |
| Qwen3-8B | 13 of 36 | `qwen3-8b-s13-q8_0.gguf` | 410 MB | [TheCacheIsALie/RidgeFill-Projector-Qwen3-8B](https://huggingface.co/TheCacheIsALie/RidgeFill-Projector-Qwen3-8B) |

The projector works with any quantization of its model (F16, Q8_0, Q4_K_M were measured).

## Use

### llama-server

```sh
llama-server -m Llama-3.1-8B-Instruct-Q4_K_M.gguf -c 32768 -ngl 99 -fa on \
    --ridgefill llama-3.1-8b-instruct-s12-q8_0.gguf --ridgefill-tail 4096
```

Every prompt of at least `--ridgefill-min-prompt` tokens is filled except its last `--ridgefill-tail` tokens. The log
shows `RidgeFill: filling prompt tokens [a, b) of N`, and the response timings gain `prompt_filled_n`.

Per request (`/completion`, `/v1/chat/completions`, ...):

- `"ridgefill": false` computes this prompt exactly;
- `"ridgefill_tail": N` overrides the tail for this prompt.

Prompt cache: the server marks approximate rows and carries their mode and exact-row set. An unchanged fill boundary can reuse its cached exact tail. A changed mode or incompatible fill region recomputes from a compatible boundary; recurrent/hybrid contexts restart if they cannot rewind safely. A request with `"ridgefill": false` or a short prompt never reuses approximate rows. The RAM prompt cache (`--cache-ram`) keeps the marker. Nonzero `--cache-reuse` (KV shifting) is refused by name. Saving filled rows (`/slots/N?action=save`) is refused; saving exact rows works.

### llama-cli

```sh
llama-cli -m Llama-3.1-8B-Instruct-Q4_K_M.gguf -c 32768 -ngl 99 -fa on \
    --ridgefill llama-3.1-8b-instruct-s12-q8_0.gguf -f long-document.txt
```

Same options and behaviour as the server (llama-cli runs on the server code).

### llama-perplexity, llama-bench

```sh
llama-perplexity -m model.gguf -f wiki.test.raw -c 8192 -ngl 99 -fa on --ridgefill proj.gguf
llama-bench -m model.gguf -ngl 99 -fa 1 -p 8192,16384,32768 -n 0 --ridgefill proj.gguf --ridgefill-tail 4096
```

llama-perplexity fills at most the unscored first half of each chunk (so the effective tail is at least N/2) and
supports the perplexity and KL-divergence modes only. llama-bench fills the pp tests without `-d`, and marks them
` rf-<mode>-t<T>` in the test name.

### llama-ridgefill-score

The [native NLL scorer](tools/ridgefill/README.md) reads pretokenized JSON/JSONL, performs one whole-document prefill, and writes full-vocabulary target logprobs for off, speed, quality or quality+qattn. For NLL, set `--qattn-window-end known` to exclude scored targets from selection. `tools/ridgefill/run_nll.py` pairs off and Tier 3 and reports reference-compatible novel-token NLL with BCa intervals.

### Options

See [Options and evaluation](#options-and-evaluation) for the shared flags and projector-derived defaults.

### API (libllama, `src/llama-ext.h`, experimental)

```c
struct llama_ridgefill * rf = llama_ridgefill_init(model, "proj.gguf");
llama_set_ridgefill(ctx, rf);                    // rf must outlive ctx
llama_ridgefill_seq_set(ctx, seq_id, n - tail);  // tokens of seq_id at positions < n - tail are filled
// ... llama_decode() the prompt as usual; a filled token cannot request logits ...
llama_ridgefill_seq_set(ctx, seq_id, 0);         // before positions can go below the fill end again (context shift)
llama_ridgefill_free(rf);
```

Batches and ubatches may be split anywhere: the fork never mixes filled and exact tokens in one ubatch.

## Limits

- S1/vtail quality mode supports architecture `qwen35` only, using the server or native scorer with `quality+qattn`. The earlier dense Llama (`llama`) and Qwen3 (`qwen3`) paths do not support S1/vtail. MoE, sliding-window attention, other hybrid architectures, multimodal RidgeFill inputs and Llama 4 remain unsupported.
- The output is not identical to stock. E3 changes RULER by -1.2/-1.4 percentage points at 16K/32K and LongBench v2 by -1.4 points. Top-1 token agreement on post-prompt tokens is about 94% (94.64% at 16K, 93.60% at 32K); see the [results and intervals](#measured-qwen38-27b-results).
- Prompts under `--ridgefill-min-prompt` run exact, including requests with `ridgefill: true`. Prompts no longer than the exact tail also run exact.
- Draft models (`--model-draft` / `-md`) and MTP (`--spec-type draft-mtp`) are refused at startup when a RidgeFill projector is loaded.
- LoRA adapters, control vectors, embeddings and reranking are refused. Filled prompt tokens cannot request logits; media inputs are unsupported.
- The approximate prompt cache rules above apply; nonzero `--cache-reuse` is refused, and slot save is refused for filled slots.
- Budget extra memory for the first filled prompt, beyond stock model and KV allocations. The [VRAM table](#ttft-and-vram-warm-median-of-3-same-gpu) reports overhead of 1592-3014 MiB with resident maps and 268-1732 MiB with streamed maps across 16K-120K contexts. TTFT in that table is measured after warmup.

## Tests

- CPU, tiny Llama and tiny Qwen3 (F32, random weights, the kva test fixtures): the filled KV rows and the tail logits
  agree with the PyTorch reference (`kva.kva.finish`) to about 1e-5 relative (flash attention off), over several batch
  and ubatch sizes; a deliberately wrong projector (layer L+1's map at layer L) fails the same check by five orders of
  magnitude; with RidgeFill off, or with a projector loaded but nothing filled, logits and KV cache are byte-identical
  to upstream llama.cpp. llama-server: greedy output of a filled prompt equals the reference's exact decode after its
  fill; prompt-cache rules; parallel slots; byte-identical responses to upstream with RidgeFill off.
- CUDA (RTX 4090, Llama-3.1-8B BF16 GGUF), against the PyTorch reference on wikitext prompts, last 256 positions:
  KL(reference || llama.cpp) at 8K 0.0005 (engine noise floor 0.0005), at 16K 0.0015 (floor 0.0006). At 32K with
  T = 4096 the gap is 0.050 (floor 0.0009), while RidgeFill itself moves the reference by 1.2-1.5 nats there; the
  filled K/V rows match the reference within the same 0.5-4 % as the exact rows of the lower layers, uniformly over
  positions, and with T = 8192 the gap falls to 0.0015. Upstream vs this fork without a projector: identical
  per-chunk perplexities on CUDA.

### Experimental S1/vtail

The settings and reporting contract are defined against RF27 C API v2. A build without that API refuses enabled S1/vtail by name. Enabled scorer and HTTP paths pass the tiny CPU reference gates; the final 27B CUDA measurements are [above](#measured-qwen38-27b-results). CLI, bench and perplexity refuse these features because they use a single-pass prefill. Server/scorer require quality+qattn. S1 requires a positive qattn share and header_rows=0; it refuses blind/overlapping/empty windows and windows beyond LOWER rows. Vtail requires a positive sample count and a first tail ubatch large enough for min(C, tail). Existing qwen35-only, LoRA, control-vector, embedding/non-causal, recurrent-rollback and replay-order restrictions still apply.

S1 replaces the base class/digit/header/qattn union with its mass-ranked rows, then unions special rows when requested. Reporting attributes final exact rows to special and S1, rather than claiming the discarded base rows were kept. The driver supplies completed S1 window bounds, the actual dense probe pair count and the actual vtail sample count. Missing execution metadata is refused. Disabled features retain the previous reporting JSON shape and MAC totals.

For prefix P, window W and hidden width H, each late attention layer charges S1_FILL P*(H^2 + K/V projections), because LATE recomputes that fill. S1_PROBE charges W full rows through every late layer, with attention visibility restricted to the prefix plus causal window rows. The implementation still evaluates dense matrices, so probe attention charges 2*query_heads*head_dim*probe_kv_pairs, using the driver's actual per-ubatch pair count including padded and masked cells. Vtail separately charges min(C,tail)*(H^2 + two V projections) per late attention layer for measuring the scale. Norms, softmax, norms of V and elementwise rescaling remain outside the analytic model. Token fractions count final prompt-row coverage, while MAC fractions include the extra passes. Enabled reports add s1 window/fill/probe costs and vtail sample/measurement costs.

The server runs as many unchanged sweep steps as fit in one stock logical n_batch chunk's work budget, after reserving generating rows. Each step is charged in full-model prompt row equivalents by adding its estimated MAC cost and model/projector weight traffic, without assuming perfect overlap. The sum is bounded by the full-model cost of the same physical rows when each estimate is smaller; an individually larger estimate still applies. This keeps one cheaper unchanged graph admissible when n_batch equals n_ubatch. MAC costs use the layer dimensions, actual exact rows and dense attention extent. Weight costs use the loaded tensor byte sizes normalized to a full-model physical microbatch; even one exact row reads the late FFN weights. Cheap lower/fill steps can therefore advance more than n_batch physical rows in an iteration, while sparse late steps pay their weight-read cost. Metadata phases have a minimum positive charge. The stock/off batch path and individual sweep graph shapes stay unchanged. Chunk traces report phase, steps, physical rows, work charged and elapsed time. A calibration or probe step that cannot fit is refused by name.

### Experimental lexical net

`--ridgefill-lexical on` (HTTP `ridgefill_lexical: true`) enables the word/unigram/unit-left selector. The default is four consecutive normalized words in the final 64 content tokens, clipped to the exact tail. For ChatML, the window ends before the last user turn's closing marker and assistant header. The scorer honors its selector endpoint and defaults to N-K so scored targets cannot affect words or boundaries. Control, user-defined, unknown and unused token bytes are NUL barriers; invalid UTF-8 bytes separate words.

Words contain only Unicode 16 Letter/Number codepoints. Punctuation, underscores and combining marks separate words. ASCII A-Z fold to lowercase; other case and Unicode forms stay unchanged. Exactly one final ASCII s is removed from words of at least four codepoints unless the word ends in ss. Units end at NUL, CR/LF, ASCII .!? followed by Unicode whitespace/end, or U+3002/U+FF01/U+FF1F; surrounding Unicode whitespace is trimmed. Phrases never cross units. Window-edge partial words and prefix-edge partial words are excluded.

For each prefix word start, extend equal window phrases as far as possible within their units. A phrase must contain a normalized unigram occurring at most twice in the eligible prefix. Deduplicate matched core row spans, then admit longer cores first, breaking ties by earlier position. Reserve the containing unit from its start, up to 80 rows, clipped to the prefix. A core outside this left-anchored span is skipped; the span never slides. Admit whole spans only, with their union limited to floor(prefix/20). This retains the full key/value unit when it fits; long units can still lose protection.

The common CLI, server HTTP and native scorer expose `ridgefill_lexical_window`, `ridgefill_lexical_min_words`, `ridgefill_lexical_span_rows` and `ridgefill_lexical_max_occurrences` (CLI uses hyphens). The defaults are 64, 4, 80 and 2. Window/span/rarity accept integers 1..4096; minimum words accepts 2..4096. Unigram anchors and the 5% prefix cap are fixed. Only the recommended defaults have the frozen Q6 coverage result; other settings need a quality screen.

The report records the resolved rule/Unicode version/parameters, window, sorted rows, matched cores, reserved unit spans, rare anchors and occurrence counts, candidates, cap and clipped/oversize counters. `capped` means a whole span was refused by the 5% cap. `clipped` counts unit truncations; `oversize_skipped` counts cores lost by truncation. Lexical rows have first attribution in disjoint source counts and count as exact in token and MAC totals. The production selector must match all 68 frozen Q6 expected-row fixtures before an engine smoke; these CPU fixtures establish selector parity, not inference quality.

The net requires quality+qattn with positive qattn share. It reserves rows within the existing base/S1 row budget instead of adding rows above it. A budget smaller than the lexical set (or the base/lexical union in ordinary qattn) is refused by name; S1 retains its existing special-row union. Lexical off keeps the previous selection and report shape. The E3 measurements above use this selector with the recommended defaults.

### Experimental projector streaming

`--ridgefill-stream-projector on` keeps projector maps and unused alpha-zero corrections in host RAM. The quality+qattn server/scorer sweep stages two layer maps per target device, owned by the context. This is a startup allocation choice, default off; HTTP can select supported modes but cannot change the projector's storage. Active reports include `projector_streamed: true`. Nonzero alpha and active modes outside the two-sweep driver refuse by name.

S1_FILL runs one attention layer over the prefix before advancing to the next layer. The ordered late pass runs one layer over the prefix and its first vtail chunk, then advances. F32 compact storage carries only the selected prefix rows and that first tail chunk. Individual projection, mixer and MLP row shapes stay the same; the rest of the exact tail uses the previous graph. The first tail span stays atomic across layers: if new generators leave too little batch capacity, it refuses by name; increase n_batch to leave space beside generators. The next map copies into the inactive buffer on a worker while the current layer computes. The driver joins before using that map and synchronizes before buffer reuse; immutable host sources and tensor metadata outlive the pending future.

After LOWER and question-mass evaluation finish, the first S1 fill graph is submitted and a worker reads the immutable h_S buffer to host RAM while fill compute runs. Sweeps without S1 keep the synchronous read. The sweep joins this read before row compaction or cancellation and owns the source until the worker finishes. The driver frees the full device buffer before the late pass and seeds the exact-row F32 carry from those saved values. Device h_S then holds bulk prefix rows, the first calibration span and one zero row. Exact prefix projections use that zero row because their outputs are overwritten by the seeded/carried exact residuals before the mixer. Remaining exact-tail h_S rows upload once per tail chunk through a bounded staging buffer, including the server's deferred final token. This avoids a full h_S upload per late layer. F16/BF16/F32 state and logit byte checks cover the compaction.

Dense causal text layouts build the streamed attention mask on device with existing fill, repeat, diagonal-mask and copy operations. Only the current chunk needs an FP32 diagonal band; the visible prefix is zero and future keys stay masked. Holes, shifted positions, foreign leading cells, incompatible M-RoPE coordinates, SWA, alibi and noncausal attention use the original host mask. That routine runs on exactly the consumed query rows, with a host cache capped at 1 GiB of matrix payload. A changed owner cell layout, mask type or query-position list forces regeneration. The stock path is unchanged.

Replay looks up positions in the cache's maintained per-sequence ordered index. The independent lower question-mass layer graphs run in order, retaining the original per-head operations and FP64 accumulation order while reducing live scratch. Streamed captures move to host RAM after LOWER; each mass graph stages only its current window back to its original device before normalization and projection. Consecutive single-sequence batches walk selected rows and replay-cache positions with sorted cursors. Sparse or mixed batches retain per-row binary lookup. These paths affect only RidgeFill.

S1_PROBE retains its original chunk and mass-reduction order. CPU checks compare complete KV/convolution/recurrent-state bytes, full-vocabulary logits through later decodes, S1 mass/rows and vtail scales; cancellation after carry allocation must permit identical replay. Memory still scales with h_S and the exact-row carry, so report the tested context/selection with every VRAM comparison.

After synchronization, the log records cumulative context map-upload bytes. A difference across a single prefill measures its host-to-device W/b payload, including prefetch maps. It excludes model loading, ordinary inputs and readback, host-to-host copies and backend-internal transfers; concurrent sequences share the context counter.

The two debug environment variables are read once when the engine library initializes. Set them before starting the process. `LLAMA_RIDGEFILL_PROFILE=1` records absolute monotonic host intervals for map copies, graph reset/build/allocation, input preparation, compute submission and synchronization. Worker setter intervals end after the copy completes. Replay lookup includes temporary-container destruction. Compaction records separate wait, allocation, download, pack and upload intervals. Compute submission intervals are API timings; synchronization ends bound completion. Keep these diagnostic runs separate from unprofiled TTFT repetitions. `LLAMA_RIDGEFILL_GRAPH_CHECK=1` reports graph tensor peaks and input-view bytes during reservation. Neither setting can change between requests. The only other supported RidgeFill environment variable is the deprecated `LLAMA_RIDGEFILL_TRIM_KV=0|1` override described below; it is read when the projector loads.

### Experimental S1 bulk K/V reuse

Use `--ridgefill-trim-kv on|off` when starting the server/scorer to reuse S1_FILL's bulk attention cache in the ordered late pass. This startup switch defaults on and takes effect in the S1 two-sweep path. `LLAMA_RIDGEFILL_TRIM_KV=0|1` remains a deprecated override for existing harnesses and logs a warning. The pass skips the bulk ridge projector, retains the original K/V GEMM dimensions and updates only exact-row cache entries. Keys are gathered before normalization/RoPE to preserve the fused cache-store arithmetic. The GDN Q projection and decode path are unchanged. The 16K resident-projector smoke matched full logit bytes and reduced median TTFT by0.95%; other configurations still need separate measurement.

For S1 probes in a unified cache with multiple sequence slots, only the active sequence's K/V cells are gathered, one KV head at a time, into prompt position order before attention. The causal mask still excludes the unrun tail gap and future query rows. Probe mass maps directly to prompt positions, and reported probe pairs exclude other sequences. This prevents unrelated physical cache placement from creating long wholly masked leading ranges in the probe softmax. Single-sequence probes and ordinary decoding keep their previous graphs. The server's scheduling estimate remains a conservative bound based on the shared cache extent.

The two-sweep flow also supports separate KV streams with parallel slots (`--no-kv-unified`). RidgeFill reserves its one-sequence graphs against a full-size view of one KV stream; runtime replay uses the active sequence's stream. This view changes no cache cells or recurrent state and allocates no device memory. Single-pass parallel quality still requires `--kv-unified`.

RidgeFill refuses a draft model (`--model-draft` / `-md`) and built-in MTP (`--spec-type draft-mtp`) at startup, before model loading. Remove the draft option or select `--spec-type none` to use RidgeFill, or remove `--ridgefill` to use speculative decoding. MTP is optional and is not needed by RidgeFill.
