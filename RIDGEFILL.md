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

RidgeFill is by Dylan Johnston and tcclaviger. Method, reference implementation and paper: TODO (Dylan).

## Qwen3.8-27B (`qwen35`)

**Release on hold:** chat-template quality must meet the CHAT-TUNE/CHAT-HELDOUT bars before release. The new chat-row controls are experimental; defaults remain unchanged.

The Qwen3.8-27B port implements speed, class/digit quality and server question-attention quality prefill. The format-2 projector converter and tiny Qwen3.5 fixtures are available. On engine `3ac42957d`, the canonical tiny CPU G1 passed 27 checks each for speed (alpha 0/1) and quality with explicit exact rows (alpha 0), across three chunk sizes. The gates compare logits, KV caches, convolution state and recurrent state with the oracle and require positive/negative controls. Quality alpha 1 is refused by name until the rho-scaled correction is implemented. Real-model measurements and their limitations are below; the later Llama/Qwen3 tables are earlier measurements on different models.

The source projector uses split 16 of 64 layers, exact tail 4096, minimum tail 512, and minimum prompt 8192. Defaults come from the projector: alpha 0, class share 0.5, digits on, question-attention share 0.2. A format-2 GGUF checks every model fingerprint field and the vocabulary token hash; the tokenizer JSON hash and source file hashes are retained as provenance. A different quantization of the same base model is allowed.

| mode | exact prefix rows R | implemented/tested status |
|---|---|---|
| `off` | all prompt rows run stock | no approximate prefill; tiny off controls passed |
| `speed` | empty | tiny CPU G1 passed at alpha 0/1; row-selector flags do not add exact rows |
| `quality` | union of class and digit rows | implemented at alpha 0; 24-document tuning NLL measured below; tuning bar FAIL |
| `quality+qattn` | quality rows plus question-attention rows | server and native scorer implemented at alpha 0; tuning NLL bar PASS; measured TTFT/needles below; bench/perplexity refuse this mode |

For a prompt of N tokens and exact tail T, P=max(0,N-T) is the prefix length. Selection is per prompt, not per prefill chunk:

- Class rows qualify when the token's vocabulary score is finite. Keep `rint(share * qualifying_rows)` positions with highest scores, ties by lower position. Rounding is half to even; the default class share is 0.5. Class share 0 disables only class rows; digit, special, header and question-attention rows still apply. With all row sources disabled and alpha 0, empty quality selection is byte-identical to speed mode on the tiny CPU fixture.
- Digit rows contain an ASCII digit in their vocabulary token text. Unknown, control, user-defined and unused tokens are excluded. Digits are on by default. Quality uses the sorted union of class and digit positions below P.
- Question-attention adds `round(qattn_share * P)` prefix positions with highest causal attention mass, ties by lower position and half-even budget rounding. The default share is 20%, using the last 64 prompt rows as queries at attention layers 3, 7, 11 and 15 below split 16. Mass sums causal `softmax(Q K^T / sqrt(head_dim))` over these layers, heads and queries. The union with class/digit rows can exceed 20% of P. `--ridgefill-qattn-blind` places the query window immediately before P instead of at the prompt end. Share 0 runs ordinary quality with identical rows and no sweep allocation.

Positive-share question-attention runs lower layers over the whole prompt first, saves h_S on the backend, selects rows, then replays the late prefix and exact tail. It never transfers the full h_S through host memory. The server yields between sweep steps; slot cancellation and shutdown release the buffers while the context is alive. Question-dependent requests recompute their prompt rather than reuse a stale question-selected prefix. L-C's tiny split-4 native G1 compared 25 cache/state/logit arrays at three chunk sizes (worst relative error 6.64e-7, threshold 1e-4); its HTTP gate passed 17 comparisons, cancellation/reuse and shutdown controls. These are CPU fixture checks, not 27B measurements.

The server stores h_S as F16, with a checked 1 GiB maximum per active sweep: `N * n_embd * sizeof(element) <= 1073741824`. At width 5120, F16 admits at most 104857 rows; 104858 refuses before allocating h_S. A 32768-row h_S is 320 MiB. F32/BF16 buffers are available through the engine API, with the same byte bound (F32 admits 52428 rows). Window captures, attention scratch space, projector and model/KV buffers add memory outside that h_S bound.

Speed mode writes attention KV and GDN convolution/recurrent state from each predicted layer input. Quality mode mixes predicted bulk inputs with the carried exact-row residuals, then restricts the remaining late-layer work to exact rows. The terminal GDN correction is applied once at the prefix boundary in speed mode. Quality needs scaling by the gate-decayed bulk share; this rho-scaled correction is pending, so nonzero quality alpha is refused. The source package already stores corrections as `[heads, V, K]`; GGUF conversion reorders value heads into llama.cpp order without transposing K/V again.

### Measured Qwen3.8-27B results

These are lane Q's measurements from REPORT-Q.md (2026-10-20, 22:05 CDT). Machine: AMD Ryzen 5 9600X and one AMD Radeon AI PRO R9700. Vulkan uses RADV GFX1201; HIP uses the ROCm 7.10.0a20251120 TheRock SDK with Clang 22.0.0. Host C/C++ is GNU 16.2.1. The model is `Qwen3.8-27B-UD-Q4_K_XL.gguf`, 17548181504 bytes, SHA256 `3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e`; the split-16 projector is identified below. Builds are Release, based on upstream `1242be66f`. Every comparison uses a mode-off control from the same build and backend, with the projector loaded in both arms.

| measurement | build commit / build number | backend |
|---|---|---|
| speed prefill, decode and streaming TTFT | `6dac68bca` / 11811 | Vulkan and HIP native bench; Vulkan TTFT |
| Tier 3 streaming TTFT and needles | `e3fc64ef2` / 11857 | Vulkan |
| all three modes' native NLL | `36bf1218f` / 11859 | Vulkan |

Common runtime settings: `-ngl 99 -fa on -ctk f16 -ctv f16 -b 2048 -ub 512 -t 4`, one sequence, split 16, T=4096 and alpha=0. Quality adds class share 0.5 and digits on; Tier 3 adds qattn share 0.2, window 64 and blind off. The sweep's h_S is F16; the scorer expresses these types as `--cache-type f16 --h-type f16`. Each instrument's additional settings and repeat counts follow.

**G4 native prefill:** mean tokens/s across three separate timed processes per arm, alternating off/speed, speed/off, off/speed. Each process uses `-p N -n 0 -r 1 -o jsonl`; native warmup is enabled and excluded. Values after +/- are sample standard deviations, not confidence intervals.

| backend | mode | build | prompt tokens | stock off tokens/s | candidate tokens/s | candidate / stock |
|---|---|---|---|---|---|---|
| Vulkan / RADV | speed | `6dac68bca` | 8192 | 1109.166113 +/- 13.273352 | 1535.722486 +/- 1.927811 | 1.384574x |
| Vulkan / RADV | speed | `6dac68bca` | 16384 | 1062.330660 +/- 5.392335 | 1856.252181 +/- 1.812477 | 1.747339x |
| Vulkan / RADV | speed | `6dac68bca` | 32768 | 986.738162 +/- 1.462639 | 2015.278014 +/- 1.498124 | 2.042364x |
| HIP / ROCm 7.10.0a20251120 | speed | `6dac68bca` | 8192 | 919.448272 +/- 5.549820 | 1194.427135 +/- 5.084983 | 1.299069x |
| HIP / ROCm 7.10.0a20251120 | speed | `6dac68bca` | 16384 | 882.038781 +/- 1.783686 | 1372.149912 +/- 15.232679 | 1.555657x |
| HIP / ROCm 7.10.0a20251120 | speed | `6dac68bca` | 32768 | 830.496230 +/- 0.384159 | 1467.261410 +/- 5.688077 | 1.766729x |

The speed-mode analytic MAC ceilings at 8K/16K/32K are 1.433150x/1.840689x/2.174029x. They include GDN convolution/scan arithmetic and exclude norms, pointwise work, RoPE and memory traffic; they are not wall-clock guarantees. Native prefill for quality and quality+qattn is NOT RUN on either backend; llama-bench refuses quality+qattn by name.

**G4 streaming TTFT:** seconds to the first generated token, three paired repetitions per arm and length, one excluded warmup per arm/length. The server uses context 33792, one slot, `--cache-ram 0 --no-warmup`; each request has `cache_prompt=false` and `n_predict=1`. Values after +/- are sample standard deviations. Each row retains its own measured stock control.

| backend | mode | build | prompt tokens | stock off seconds | candidate seconds | stock / candidate |
|---|---|---|---|---|---|---|
| Vulkan / RADV | speed | `6dac68bca` | 8192 | 7.537798 +/- 0.036793 | 5.438809 +/- 0.017547 | 1.385928x |
| Vulkan / RADV | speed | `6dac68bca` | 16384 | 15.637596 +/- 0.017706 | 8.962633 +/- 0.005620 | 1.744755x |
| Vulkan / RADV | speed | `6dac68bca` | 32768 | 33.426977 +/- 0.011801 | 16.410729 +/- 0.006820 | 2.036898x |
| Vulkan / RADV | quality+qattn | `e3fc64ef2` | 8192 | 7.459345 +/- 0.062714 | 6.546534 +/- 0.035149 | 1.139434x |
| Vulkan / RADV | quality+qattn | `e3fc64ef2` | 16384 | 15.556874 +/- 0.022621 | 12.208560 +/- 0.012700 | 1.274260x |
| Vulkan / RADV | quality+qattn | `e3fc64ef2` | 32768 | 33.241329 +/- 0.022334 | 25.502722 +/- 0.005813 | 1.303442x |

Quality TTFT is NOT RUN on Vulkan or HIP. HIP TTFT for speed and Tier 3 is NOT RUN: the upstream HIP server hung during warmup and was killed; no timed result was accepted. Tier 3 native prefill throughput is not inferred from server TTFT.

**G4 decode:** with `-p 0 -n 128`, three separate measurements per arm on `6dac68bca`, Vulkan stock/speed means are 31.045438/30.825002 tokens/s (0.992900x); HIP stock/speed means are 29.255828/29.262585 tokens/s (1.000231x). Both mean ratios meet the +/-1% decode bar. Quality and Tier 3 decode benchmarks are NOT RUN.

**G3 NLL:** 24 raw-text tuning documents, eight each from encyclopedia, papers and prose, cut to 16384 tokens; score the final 512 targets. The primary estimator is the reference-compatible novel-token mean: exclude repeated 8-token spans with `kva.data.novel_mask`, then weight each document equally. One whole-document native prefill per arm/document uses [llama-ridgefill-score](tools/ridgefill/README.md), context 16896 and F16 KV/h_S. For NLL, `--qattn-window-end 15872` freezes selection at [15808,15872), before target IDs [15872,16384); no scored target influences selection. Quality and speed share the saved same-session off controls from the Tier 3 jobs.

NLL values and deltas are nats per novel token. Paired BCa intervals use all 24 document deltas, 5000 resamples, seed 20261004. The tuning bar is **one-sided 95% upper <= +0.00995**. That upper is the upper endpoint of the central 90% BCa interval; the two-sided 95% interval is reported separately.

| mode | stock off mean | candidate mean | candidate - stock | two-sided 95% BCa | one-sided 95% upper | tuning NLL bar |
|---|---|---|---|---|---|---|
| quality+qattn | 1.883891079 | 1.889337384 | +0.005446305 | [+0.001662961,+0.010649454] | +0.009789065 | PASS |
| quality | 1.883891079 | 1.904015577 | +0.020124498 | [+0.009833257,+0.033912766] | +0.031551826 | FAIL |
| speed | 1.883891079 | 1.914023519 | +0.030132440 | [+0.017180030,+0.045889920] | +0.043318243 | FAIL |

**G3 needles:** Vulkan measurements use same-job stock controls. On `e3fc64ef2`, each of 48 tuning needles receives one request per arm, context 16896, greedy generation for 32 tokens, appended answer IDs removed, Qwen thinking-off chat template and seven-digit boundary scoring. The earlier speed HTTP run stopped after 24 of 48 needles; its subset is not a full-set result.

| mode | build | paired needles | stock correct | candidate correct | accuracy delta (percentage points) | 95% paired BCa (percentage points) | status |
|---|---|---|---|---|---|---|---|
| quality+qattn | `e3fc64ef2` | 48/48 | 48/48 | 48/48 | 0 | [0,0] | MEASURED |
| speed | `6dac68bca` | 24/48 | 24/24 | 23/24 | -4.166667 | [-20.833333,0] | PARTIAL |
| quality | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

Tier 3's interval is degenerate and is not a general noninferiority guarantee. HIP G3 NLL/needles are NOT RUN.

**Recommended experimental baseline: Tier 3 (`quality+qattn`), alpha 0**, with the defaults above. It is the only measured mode meeting the tuning NLL bar, and its 48 tuning needles all succeed. This recommendation does not establish held-out confirmation. In separate held-out reference runs, the four chat-template documents show mean Tier 3 degradation of **+0.291 nats per novel token**; the registered held-out confirmation was REFUTED. That is a reference result, not an engine measurement, and chat-template quality is a material limitation despite the tuning needles succeeding. The cause and a general fix have not been established.

Full-model engine/reference agreement also remains limited: G2-as-written is INCONCLUSIVE because the wrong-map control is not separable, the registered G2-shift amendment is FAIL, and Tier 3 row unions differ from the reference. The tuning NLL bar does not resolve these checks or establish an overall release verdict.

| backend / placement | measured status | remaining scope |
|---|---|---|
| CPU | tiny oracle/tool and clean archive/refusal gates passed | 27B CPU measurements NOT RUN |
| Vulkan / one GPU | speed G4; Tier 3 TTFT/needles; all-mode native NLL measured above | quality G4, full held-out engine G3 and GPU archive gate NOT RUN |
| HIP / one GPU | speed native prefill/decode G4 measured above | server TTFT, G3, quality G4 and GPU archive gate NOT RUN |
| Vulkan / two GPUs | NOT RUN | coordinator's separate layer-split smoke |
| CUDA | NOT RUN | no Qwen35 RidgeFill backend gate run |
| Metal | NOT RUN | no Qwen35 RidgeFill backend gate run |

### Projector conversion

The planned release location is `TheCacheIsALie/RidgeFill-Projector-Qwen3.8-27B-i8`; publication is pending. Projector: Dylan Johnston and tcclaviger, Apache-2.0, DOI `10.5281/zenodo.23179168`. The converted artifact is `ridgefill-projector-qwen3.8-27b-i8-S16.gguf`: Q8_0 maps, F32 biases/corrections/row scores, 1452188032 bytes, SHA256 `8646d41a5a28d60ef53cc5e1fee4b2bfb20d0775db88d3f89d0731ad9f6b629d`.

```sh
# Once the release is published:
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
| `--ridgefill-share N` | projector (27B: 0.5) | per-prompt class-row share in [0,1] |
| `--ridgefill-digits on\|off` | projector (27B: on) | add digit rows in quality |
| `--ridgefill-special-exact on\|off` | off | keep unknown/control/user-defined token rows exact in quality modes |
| `--ridgefill-header-rows K` | 0 | keep the next K prefix rows after each special token exact, independently of special-exact |
| `--ridgefill-qattn-window-mode last\|turn` | last | last uses the fixed query window; turn uses rows after the last complete user header |
| `--qattn-turn-cap N` (alias `--ridgefill-qattn-turn-cap`) | 256 | maximum query rows in turn mode; independent of --ridgefill-qattn-window |
| `--ridgefill-alpha N` | projector (27B: 0) | state correction scale in [0,1]; quality currently requires 0 |
| `--ridgefill-qattn-share N` | projector (27B: 0.2) | question-attention prefix share in [0,1]; server only |
| `--ridgefill-qattn-window N` | 64 | positive query-window size in last mode only; server only |
| `--ridgefill-qattn-blind` | off | use queries immediately before the prefix end; server only |

```sh
llama-server -m model.gguf -c 32768 -ngl 99 --ridgefill projector.gguf --ridgefill-mode speed --ridgefill-tail 4096
```

In experimental turn mode (`--qattn-window-mode turn`, alias `--ridgefill-qattn-window-mode`), the driver parses the literal `<|im_start|>user\n` with special tokens enabled and no BOS. It finds the last complete header entirely before END, excludes the header, and captures `[max(turn_start, END-cap), END)`. END is the prompt end for HTTP and N-K for native NLL, which excludes scored targets from selection. Missing/empty user turns and turn plus blind selection refuse by name. Turn mode has no measured full-model chat quality verdict; release remains on hold. Defaults remain last/64 with special-exact off and header-rows 0.

Quality settings are `--ridgefill-mode quality --ridgefill-share 0.5 --ridgefill-digits on --ridgefill-alpha 0`. For question-attention, use `--ridgefill-mode quality+qattn --ridgefill-qattn-share 0.2 --ridgefill-qattn-window 64 --ridgefill-alpha 0` on llama-server. Tier 3 is the recommended RidgeFill mode, subject to the raw-text tuning scope and chat-template limitation above.

| per-request field | effect |
|---|---|
| `ridgefill` | boolean; false selects stock processing regardless of mode |
| `ridgefill_mode` | off, speed, quality or quality+qattn; inherits server mode when omitted |
| `ridgefill_share` | finite number in [0,1]; inherits the startup class share |
| `ridgefill_tail` | integer; 0 inherits the server tail, positive values must meet projector tail_min |
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
| parallel quality with non-unified KV | names unified KV and `-kvu`; enable `--kv-unified` |
| bench/perplexity `quality+qattn` | names the server question-attention sweep; use the server or ordinary quality |
| positive-share server qattn with shared/parent/child prompts, speculative decoding, LoRA or a non-completion task | names the requirement for an independent text completion without speculative decoding or LoRA |
| qattn replay with recurrent rollback (`n_rs_seq > 0`) | names recurrent rollback snapshots; disable that combination |
| qattn split without lower attention layers, or a requested layer that is recurrent | names the split/layer; the 27B split-16 projector has four usable layers |
| h_S exceeding 1 GiB, invalid shape/type, or failed backend allocation | names the sweep buffer dimensions/type/byte limit/allocation; no oversized allocation or host fallback |
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
on every backend. **Tested: CPU and CUDA.** Metal, Vulkan, ROCm/HIP and SYCL are expected to work but are untested.

## Get a projector

A projector belongs to one model (architecture, layer count, hidden size are checked when it is loaded) and one split.

Convert a projector from the kva safetensors format (`layer.<i>` tensors, `[W | bias]`, plus `split` in the metadata):

```sh
python convert_ridgefill_to_gguf.py llama-3.1-8b-instruct-sweep-s12.safetensors \
    --base /path/to/Llama-3.1-8B-Instruct --outtype bf16     # or q8_0 (half the size, same quality), f16, f32
```

`--base` is the Hugging Face model directory the projector was fitted for (only its `config.json` is read). The GGUF
records the split, the layer count, the hidden size and the sha256 of the source file.

Ready projectors (Dylan: TODO where they are hosted):

| model | split | file | size |
|---|---|---|---|
| Llama-3.1-8B-Instruct | 12 of 32 | `llama-3.1-8b-instruct-s12-bf16.gguf` | 671 MB |
| Llama-3.1-8B-Instruct | 12 of 32 | `llama-3.1-8b-instruct-s12-q8_0.gguf` | 357 MB |
| Qwen3-8B | 13 of 36 | `qwen3-8b-s13-bf16.gguf` | 772 MB |
| Qwen3-8B | 13 of 36 | `qwen3-8b-s13-q8_0.gguf` | 410 MB |

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

- The earlier validated paths are dense Llama (`llama`) and dense Qwen3 (`qwen3`). Qwen3.8-27B (`qwen35`) measurements and limitations are described above. MoE, sliding-window attention, other hybrid architectures, multimodal RidgeFill inputs and Llama 4 remain unsupported.
- Approximate: outputs differ from normal llama.cpp. Quality drops with the share of the prompt that is filled.
- No speedup for short prompts: below `--ridgefill-min-prompt`, or when the prompt is not longer than the tail,
  nothing is filled. The speedup grows with prompt length (analytic ceiling about 1.35x at 8K, 1.67x at 16K and 1.93x
  at 32K for split 12 of 32 with T = 4096).
- Not combined with LoRA adapters, control vectors, embeddings or reranking (refused), with requests that need
  logits inside the prompt (for example MTP drafting: those prompts are computed exactly), or with media inputs.
- The approximate prompt cache rules above; nonzero `--cache-reuse` refused by name; slot save refused for filled slots.
- The projector adds its size to memory (671 MB bf16 / 357 MB q8_0 for Llama-3.1-8B), on the same device as the
  layers it serves.

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
