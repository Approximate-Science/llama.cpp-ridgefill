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

## Qwen3.8-27B (`qwen35`): development status

The Qwen3.8-27B port is in progress. The format-2 projector converter and tiny Qwen3.5 fixtures are available. On engine `3ac42957d`, the canonical tiny CPU G1 passed 27 checks each for speed (alpha 0/1) and quality with explicit exact rows (alpha 0), across three chunk sizes. The gates compare logits, KV caches, convolution state and recurrent state with the oracle and require positive/negative controls. Quality alpha 1 is refused by name until the rho-scaled correction is implemented. Real-model results below are **NOT YET MEASURED** for this port; the later Llama/Qwen3 tables are earlier measurements on different models.

The source projector uses split 16 of 64 layers, exact tail 4096, minimum tail 512, and minimum prompt 8192. Defaults come from the projector: alpha 0, class share 0.5, digits on, question-attention share 0.2. A format-2 GGUF checks every model fingerprint field and the vocabulary token hash; the tokenizer JSON hash and source file hashes are retained as provenance. A different quantization of the same base model is allowed.

| mode | exact prefix rows R | implemented/tested status |
|---|---|---|
| `off` | all prompt rows run stock | no approximate prefill; tiny off controls passed |
| `speed` | empty | tiny CPU G1 passed at alpha 0/1; row-selector flags do not add exact rows |
| `quality` | union of class and digit rows | tiny CPU server, bench, perplexity and paired G3 passed at alpha 0; real-model results pending |
| `quality+qattn` | intended quality rows plus question-attention rows | tools refuse by name until selector and sweep integration are available |

Class/digit selection is per prompt: a class row qualifies when its vocabulary score is finite; keep `rint(share * qualifying_rows)` positions with the highest scores, ties by lower position. Rounding is half to even. Digit rows contain an ASCII digit in the vocabulary token text, with added/special tokens excluded. The engine API and common helper are tested against hand-computed tiny-fixture rows. The intended question-attention rule adds `round(qattn_share * prefix_length)` prefix rows by causal attention mass from the last `qattn_window` prompt rows at attention layers 3, 7, 11 and 15 below split 16, summed over layers, heads and queries, ties by lower position. `--ridgefill-qattn-blind` moves the query window immediately before the prefix end. The sweep integration remains pending; its caller adds question rows to the common helper's class/digit base rows. Zero qattn share requires the quality path rather than substituting speed.

Speed mode writes attention KV and GDN convolution/recurrent state from each predicted layer input. Quality mode mixes predicted bulk inputs with the carried exact-row residuals, then restricts the remaining late-layer work to exact rows. The terminal GDN correction is applied once at the prefix boundary in speed mode. Quality needs scaling by the gate-decayed bulk share; this rho-scaled correction is pending, so nonzero quality alpha is refused. The source package already stores corrections as `[heads, V, K]`; GGUF conversion reorders value heads into llama.cpp order without transposing K/V again.

| same-build paired result | mode off | speed | quality | quality+qattn |
|---|---|---|---|---|
| NLL of last 512 tokens, 24 documents at 16K | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED |
| needles, 48 prompts with thinking off | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED |
| TTFT at 8K / 16K / 32K | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED |
| prefill tokens/s and analytic ceiling | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED |
| decode tokens/s | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED |

Machine/backend, build revision, cache precision and paired 95% BCa intervals will accompany each measured row. The reference package's quality numbers are not engine measurements and are not substituted here.

| Qwen35 RidgeFill backend gate | status |
|---|---|
| CPU | tiny oracle gates and archive/refusal checks passed; real-model results NOT YET MEASURED |
| Vulkan | NOT YET MEASURED |
| HIP/ROCm | NOT YET MEASURED |
| CUDA | NOT RUN |
| Metal | NOT RUN |

### Projector conversion

The planned release location is `TheCacheIsALie/RidgeFill-Projector-Qwen3.8-27B-i8`; publication is pending. The expected GGUF filename is `ridgefill-projector-qwen3.8-27b-i8-S16.gguf`.

```sh
# Once the release is published:
hf download TheCacheIsALie/RidgeFill-Projector-Qwen3.8-27B-i8 ridgefill-projector-qwen3.8-27b-i8-S16.gguf --local-dir projector

# Or convert the verified int8 source folder, with NumPy and safetensors installed:
python3 convert_ridgefill_to_gguf.py projector-source/ --model-gguf model.gguf --outfile projector.gguf
```

Folder conversion needs no torch and defaults to Q8_0 maps. `--outtype f32|f16|bf16|q8_0` changes the maps; biases, row scores and corrections remain F32. The converter verifies SHA256SUMS and manifest file checksums before writing, and computes `adapter.ridgefill.vocab_sha256` from the base model GGUF's token strings, each followed by one NUL byte. A different tokenizer or model fingerprint is refused by name.

### Options and evaluation

With no projector, processing is stock. `--ridgefill FILE` defaults to speed mode; `--ridgefill-mode off` keeps the projector loaded and requests stock processing. All flags below are parsed, but an unavailable quality/selector/sweep path refuses by name.

| option | default | meaning |
|---|---|---|
| `--ridgefill FILE` | off | projector GGUF |
| `--ridgefill-tail N` | projector (27B: 4096) | last prompt tokens computed exactly; at least projector tail_min (27B: 512) |
| `--ridgefill-min-prompt N` | 8192 | shorter prompts are not filled; CLI/server/perplexity only |
| `--ridgefill-mode MODE` | speed with a projector | off, speed, quality, quality+qattn; availability above |
| `--ridgefill-share N` | projector (27B: 0.5) | per-prompt class-row share in (0,1] |
| `--ridgefill-digits on\|off` | projector (27B: on) | add digit rows in quality |
| `--ridgefill-alpha N` | projector (27B: 0) | state correction scale in [0,1]; quality currently requires 0 |
| `--ridgefill-qattn-share N` | projector (27B: 0.2) | question-attention share in [0,1]; sweep pending |
| `--ridgefill-qattn-window N` | 64 | positive query-window size; sweep pending |
| `--ridgefill-qattn-blind` | off | use queries immediately before the prefix end; sweep pending |

```sh
llama-server -m model.gguf -c 32768 -ngl 99 --ridgefill projector.gguf --ridgefill-mode speed --ridgefill-tail 4096
```

The intended quality settings are `--ridgefill-mode quality --ridgefill-share 0.5 --ridgefill-digits on --ridgefill-alpha 0`. They are not a measured recommendation for this engine yet. Per request, `ridgefill: false` overrides the requested mode and selects stock processing; `ridgefill_mode: "off"` also selects stock. `ridgefill_tail` overrides the exact tail, with 0 inheriting the server default. An eligible unavailable quality mode returns an error instead of using speed.

The RAM prompt cache carries the mode, exact-row set and fill marker. The unchanged fill boundary can reuse its exact cached tail. Changing mode or selected rows in a retained approximate region requires recomputation; a changed boundary restarts a recurrent/hybrid context when it cannot rewind safely. Ordinary cache_prompt reuse and --cache-ram remain available. KV shifting through nonzero --cache-reuse or request n_cache_reuse is refused by name. Slot save of filled rows is refused; an exact slot can be saved.

With a projector configured, LoRA (--lora), control vectors (--control-vector), embeddings (--embeddings), nonzero --cache-reuse and row splitting (-sm row) refuse before model/adapter I/O. This also applies when that loaded projector is in mode off. Layer split is the intended placement mode, but multi-GPU layer split is NOT RUN in the CPU gate. A filled prefix row requesting logits is refused. Short prompts and T>=N run stock. Multimodal chunks do not fill, and this text-only port has no multimodal RidgeFill validation.

Perplexity and bench support off, speed and quality at alpha 0. Both still refuse quality+qattn by name. Perplexity fills at most the unscored first half of each chunk, so its effective tail is `max(T, n_ctx/2)`; it selects quality rows separately for each chunk, including its actual BOS token. Parallel quality chunks require --kv-unified (-kvu). Failed decoding and a missing/nonfinite perplexity score exit nonzero. Bench selects rows from its full generated prompt and logs the exact-prefix row count. Startup and per-request quality with nonzero alpha refuse by name until the rho-scaled correction is implemented. Tiny CPU integration checks cover actual quality runs and these refusals, without establishing real-model quality or performance.

The paired client and CPU gates are documented in [tests/ridgefill/README.md](tests/ridgefill/README.md). `run_g3.py --mode ... --docs ... --needles ... --out ...` runs NLL, needles, TTFT and native bench against one build's mode-off arm. NLL uses only the last 512 teacher-forced targets and keeps the approximate prefix fixed; it requires T-512 >= tail_min and N-512 >= min-prompt. Needle evaluation removes appended answer IDs, applies the loaded Qwen chat template with `enable_thinking=false`, generates greedily, and checks seven digits at digit boundaries. Timing runs alternate off and candidate for at least three repeats. Paired 95% BCa intervals use the oracle's `finals/stats.py`; an inactive eligible fill or missing probability fails the run. Tiny instrumentation checks are not real-model quality/speed measurements.

`run_g5.py --ref COMMIT --backend cpu|vulkan|hip --fixture ... --out ...` audits and builds a clean git archive, then tests named refusals and exact/filled slot-save controls. It uses the caller's environment and does not provide container isolation. Container, GPU backend and multi-GPU gates must be reported separately; the CPU archive result does not establish them.

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

- The earlier validated paths are dense Llama (`llama`) and dense Qwen3 (`qwen3`). Qwen3.8-27B (`qwen35`) is under development as described above. MoE, sliding-window attention, other hybrid architectures, multimodal RidgeFill inputs and Llama 4 remain unsupported.
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
