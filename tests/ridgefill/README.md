# RidgeFill CPU checks

Run from the repository root with NumPy and safetensors installed. The converter uses the repository's `gguf-py` and needs no torch.

```sh
python3 tests/ridgefill/test_converter.py -v
python3 convert_ridgefill_to_gguf.py /path/to/int8-package --model-gguf /path/to/model.gguf --outfile out/projector.gguf
python3 tests/ridgefill/check_converter.py /path/to/int8-package out/projector.gguf --model-gguf /path/to/model.gguf
```

The package check verifies all SHA256SUMS entries and manifest file hashes, metadata preservation, every map's error bound, exact biases and row scores, and the correction head permutation. It must reject a different layer's map, an unpermuted correction, and a correction with its key/value axes swapped. Packages without nontrivial controls produce SKIPPED with a nonzero exit code.

The format-2 GGUF requires the base model's GGUF to compute the checked vocabulary fingerprint: SHA256 over `tokenizer.ggml.tokens` in token ID order, each token followed by one NUL byte. The manifest's tokenizer JSON hash is preserved separately as provenance.

The Qwen27 package writer (`kva`'s `scripts/q27_package.py`) transposes the fitted oracle correction from `[H, K, V]` into `[H, V, K]`. The GGUF converter reorders its value heads from grouped to tiled and preserves those last two axes. An oracle harness must transpose the package tensor back when adding it to the oracle's recurrent state.

Q8_0 maps are compared with `codes * scale` using each 32-element block's rounding bound, including its fp16 scale error. F32 maps must be exact. F16 and BF16 maps must equal the corresponding rounding of the source values. These are conversion checks; they do not claim oracle agreement of the inference engine.

## Tiny Qwen3.5 and G1

Create a CPU oracle venv in the repository root. Install torch from the CPU wheel index, then the oracle repository's requirements. For example, with `uv`:

```sh
uv venv --python python3 .venv-oracle
uv pip install --python .venv-oracle/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-oracle/bin/python -r /path/to/kva/requirements.txt
.venv-oracle/bin/python tests/ridgefill/make_tiny_qwen35.py
.venv-oracle/bin/python tests/ridgefill/check_fixture.py --kva /path/to/kva
tests/ridgefill/run_gate1_qwen35.sh --kva /path/to/kva --split 3 --mode speed
tests/ridgefill/run_gate1_qwen35.sh --kva /path/to/kva --split 3 --mode quality
```

The fixture has 8 layers, attention at 3 and 7, hidden size 64, attention head size 16 and 2 KV heads, GDN with 2 key heads and 4 value heads of size 16, and a kernel of 4. It is converted by upstream `convert_hf_to_gguf.py --no-mtp --outtype f32`; only vocabulary export is replaced with a synthetic vocabulary. The HF directory also contains a valid fast tokenizer with the same token strings and IDs.

Artifacts under `out/gate1-qwen35/`:

- `tiny-qwen35-hf/` and `tiny-qwen35-f32.gguf`: one shared random model.
- `projector-s3-f32.gguf`: split 3 for SPEC speed/quality cases.
- `projector-s4-f32.gguf`: split 4, with attention layer 3 below it for the question-mass case.
- `projector-s{3,4}-wrong-f32.gguf`: the next layer's map as a positive control.
- `projector-s{3,4}-wrong-st-f32.gguf`: the same maps with the correction's K/V axes swapped, as an alpha-1 positive control.

The harness pins `--engine-ref` (default `rf27-engine`), builds its `git archive` inside `out/gate1-engine/<revision>/` with `-j4`, then compiles `rf_dump`. Use `--dump /path/to/rf_dump` to reuse a binary. No engine sources are edited. Build logs identify the pinned revision. An engine build failure returns SKIPPED and exits nonzero.

`check_fixture.py` checks the alpha correction against its source orientation, exercises the oracle's digit set and question-window controls, and writes `qattn-s4-oracle.npz` for the sweep lane. It checks preparation of the reference; it does not claim engine agreement.

The common class/digit helper calls the real `llama_ridgefill_select_rows` API (engine `ff598e03d` or later). Resolve projector defaults with `common_ridgefill_resolve` first, then call `common_ridgefill_rows(params_base, rf, tokens.data(), P, request_mode)` with the effective request mode. Both quality modes return the same sorted class/digit base rows; the sweep caller adds question-attention rows. Speed/off and empty prefixes return no rows. Missing inputs, unresolved knobs, invalid tokens and invalid selector output refuse by name.

`test_row_selection.cpp` exercises the helper against the actual tiny model/projector with hand-computed positions: class scores, half-even rounding, ties by position, digit union, both quality modes, off/speed/empty-prefix controls and named bad-input refusals. Compile it against the built common and engine libraries, then run with `tiny-qwen35-f32.gguf projector-s3-f32.gguf` as its two arguments. No mock selector implementation is used.

Speed cases compare tail logits, every layer's final KV/conv/recurrent state, and every late layer's prefix state at N=700/T=200 over batch/ubatch pairs 700/512, 128/128 and 50/50. Prefix dumps are necessary to expose a wrong alpha correction before tail decay. Alpha is 0 and 1. Controls require byte-identical mode-off and T>=N outputs, a wrong map that fails by over 100x tolerance, a wrong correction orientation at alpha 1, and refusal of a filled row asking for logits. The tolerance is 1e-4 relative F32.

The same harness is the canonical G1 for quality with hand-picked rows through the RF27 API. It compares alpha-0 inference against the oracle and requires wrong maps, wrong exact rows and speed-instead-of-quality to fail that comparison. Its T>=N control supplies no exact rows because the prefix is empty. Quality alpha 1 is an expected refusal (rc 2 naming the rho-scaled correction), not a passing inference comparison. Speed retains its alpha-0/1 inference and wrong-correction-orientation checks. The quality+qattn sweep integration remains pending and returns nonzero SKIPPED. Split 4 is already available for L-C's mass comparison.

## Paired server evaluation

`run_g3.py` starts one server, measures last-512 NLL and Qwen thinking-off needles against per-request mode off, then measures streaming TTFT and native prefill/decode benchmarks from the same build:

```sh
python3 tests/ridgefill/run_g3.py --bin-dir build/bin --model model.gguf --projector projector.gguf --mode speed --docs evaldata/nll.jsonl --needles evaldata/needle.jsonl --out out/g3-speed --kva /path/to/kva --build-label COMMIT-BACKEND-CACHE --gpu-layers 99 --flash-attn off
```

Defaults are 16,384 tokens per document, exactly 512 scored targets, T=4096, tail_min=512, min-prompt=8192, TTFT/prefill at 8192/16384/32768, three paired timing repetitions, tg128, F16 KV, and one server slot with context 33792. There are no silent mode substitutions. Quality/qattn must have their engine paths available; otherwise the command exits nonzero with FAILED.json. Native bench refuses quality+qattn: use `--metrics nll,needles,ttft` to omit those bench stages explicitly. Omitted stages are labeled SKIPPED.

`--out` is a new directory; existing outputs are refused. It contains provenance.json, server.log, one JSONL per requested metric, and summary.json only after the requested measurements finish. Summaries report paired 95% BCa intervals and flag degenerate intervals; COMPLETE means measurement completed, without a quality or noninferiority verdict. The runner checks that server and bench report the same build, that the projector's tail_min matches the requested value, and that the server uses the requested model and one slot. Extra backend flags use repeated `--server-arg=...` / `--bench-arg=...`; the same explicit model, projector, mode and knob values feed both instruments. Server and bench cache types are recorded separately because native bench does not accept F32. Quality with alpha 0 is supported by server, bench and perplexity; nonzero quality alpha still refuses the missing rho-scaled correction.

The end-to-end tiny control runs speed, quality and off through the same command and verifies all measured NLL probabilities against direct logits, off/off identity, NLL and quality-vs-speed positive controls, suffix removal, the Qwen thinking-off template, seven-digit positive/negative controls, and all five measurement wrappers. Quality reference rows are selected independently in Python from the GGUF score/vocabulary tables before passing them explicitly to the native dump.

```sh
.venv-oracle/bin/python tests/ridgefill/check_g3_live.py --bin-dir out/tools-cpu-build/bin --dump out/rf_dump-tools --kva /path/to/kva --chat-template-file /path/to/Qwen3.8-chat-template.jinja --work out/g3-controls
```

For the tiny run, extract the template from the real Qwen3.8 GGUF metadata and supply it with --chat-template-file. The real-model command uses its own GGUF template by default. The tiny model has random weights; its needle accuracy and timing figures are instrumentation controls, not model quality/speed evidence.

`eval_client.py` needs Python, NumPy, and read access to the oracle repository's `finals/stats.py`; it needs no torch. Start one server with the desired projector, tail, class/digit/alpha settings and cache types. Each run alternates the candidate mode with per-request mode off on that same server/build. Use one slot for reproducible timing runs and record its launch command with the build label.

```sh
python3 tests/ridgefill/eval_client.py nll --url http://127.0.0.1:8080 --data /path/to/evaldata/nll.jsonl --kva /path/to/kva --build-label COMMIT-BACKEND-CACHE --mode speed --output out/nll-speed.jsonl
python3 tests/ridgefill/eval_client.py needle --url http://127.0.0.1:8080 --data /path/to/evaldata/needle.jsonl --kva /path/to/kva --build-label COMMIT-BACKEND-CACHE --mode quality --output out/needles-quality.jsonl
python3 tests/ridgefill/eval_client.py ttft --url http://127.0.0.1:8080 --data /path/to/evaldata/nll.jsonl --kva /path/to/kva --build-label COMMIT-BACKEND-CACHE --mode speed --lengths 8192,16384,32768 --repeats 3 --output out/ttft-speed.jsonl
python3 tests/ridgefill/eval_client.py bench --binary build/bin/llama-bench --model model.gguf --projector projector.gguf --kva /path/to/kva --build-label COMMIT-BACKEND-CACHE --mode speed --bench-arg=-ngl --bench-arg=99 --output out/bench-speed.jsonl
python3 tests/ridgefill/eval_client.py bench --binary build/bin/llama-bench --model model.gguf --projector projector.gguf --kva /path/to/kva --build-label COMMIT-BACKEND-CACHE --mode speed --lengths 0 --predict 128 --output out/decode-speed.jsonl
```

NLL scores exactly the last 512 targets of each document at `--length` (default 16384). It uses one `/completion` call per target, forces that target using sampling bias, and reads its pre-sampling `logprob` with `post_sampling_probs=false` and `n_probs=1`. The sampler copies logits before applying bias; the reported raw probability remains the model probability. The first request disables prompt reuse; later requests teacher-force the true document continuation. The tail override grows with the known context, keeping the prefix end fixed. This requires `tail-512 >= tail_min` and `N-512 >= min-prompt`. Missing/non-finite logprobs, a wrong sampled token, or an eligible candidate that fills nothing are errors.

Needle source rows contain answer tokens appended for teacher-forced reference scoring. The client removes `answer_len` IDs, detokenizes only the remaining prompt, applies `/apply-template` with `enable_thinking=false`, and tokenizes the formatted prompt with special-token parsing. Numeric answers must match at digit boundaries. The raw source answer is never appended to the generation prompt.

TTFT uses the first generated SSE token, disables the prompt cache for measured requests, warms each mode, and alternates at least three paired runs at each length. Bench runs the same executable/model with mode off and the candidate mode, alternating native `-r 1` runs; backend arguments are passed as a list, without a shell. Different lengths get separate intervals. A decode-only bench uses `--lengths 0 --predict 128`.

Outputs are JSONL: provenance, each paired unit, then document-weighted NLL or unit-weighted accuracy/speed differences with 95% BCa intervals from `finals/stats.py` (5000 resamples, its declared seed). Existing output files are refused. No units, unpaired data, or an undefined interval fails rather than producing a passing verdict. Degenerate intervals are labeled; they are not a quality guarantee.

Controls:

```sh
.venv-oracle/bin/python tests/ridgefill/test_eval_client.py -v
.venv-oracle/bin/python tests/ridgefill/check_eval_live.py --engine-dir /path/to/pinned-engine-directory --kva /path/to/kva
```

The live control directory must contain `rf_dump` and `build/bin/llama-server`. It compares 512 teacher-forced server probabilities against direct F32 logits on the tiny model, checks an identical off/off rerun, rejects post-sampling probabilities, exercises streaming TTFT, and checks positive/negative BCa integration controls. These are eval-client controls, not real-model quality/speed measurements.

The tool integration control takes the directory containing the three built executables and a matching `rf_dump`:

```sh
.venv-oracle/bin/python tests/ridgefill/check_tool_live.py --bin-dir out/tools-cpu-build/bin --dump out/rf_dump-tools
```

It compares server speed/off logprobs with direct F32 logits, changes request modes and exact tails through a live cache, checks CLI propagation through /props, and completes off/speed/quality bench and perplexity runs. It checks named refusals for quality alpha 1, qattn, non-unified parallel perplexity and a document that scores no chunk. Quality runs must log a nonempty exact-row selection and finite measurements. Bench uses F16 KV because its existing parser does not accept F32. This is a tiny CPU integration check; it produces no real-model performance claim.

## G5 archive portability and refusals

Run the gate on a committed revision containing the tools and refusal checker, with a new output directory:

```sh
python3 tests/ridgefill/run_g5.py --ref COMMIT --backend cpu --fixture out/gate1-qwen35 --out out/g5-cpu-COMMIT
```

It extracts git archive into a clean source directory, prevents Git from discovering the parent checkout, pins the build commit/count, builds server/bench/perplexity with at most -j4, checks their build IDs, audits the RidgeFill source files with rg, and runs the live refusal controls. The source list and patterns are in audit.json; portability-grep.log must be empty (rg exit 1). Positive grep controls insert an absolute user path, hostname, device ID and TP-size assumption; the negative control is portable code. Logs and exact build commands stay under --out. Existing output directories refuse.

Use the same script in lane Q's bridge with --backend vulkan or --backend hip and --gpu-layers 99. HIP toolchain/target settings are explicit --cmake-arg=-DOPTION=VALUE arguments inherited from the bridge setup. The gate does not select GPU IDs or assume a target architecture/TP degree. It does not provide container isolation; run the command in a clean container when one is available. Its result covers only the selected backend, archive build, source audit and refusal tests; multi-GPU layer split remains a separate hardware gate.

`check_refusals.py` can also run against an existing build:

```sh
python3 tests/ridgefill/check_refusals.py --bin-dir build/bin --fixture out/gate1-qwen35 --work out/g5-refusals
```

It requires named startup refusals for LoRA, control vectors, embeddings, nonzero --cache-reuse and -sm row; bench must refuse embeddings and row split. Per-request n_cache_reuse must refuse. A live speed prompt must fill before slot save is refused without creating a file. Mode off must fill nothing and save a nonempty exact slot. Missing LoRA/control files are intentional: the RidgeFill combination must be rejected before file I/O, while stock with the same missing LoRA must reach its normal loader. Ordinary cache_prompt reuse and RAM prompt caching remain supported; cache-reuse means KV shifting.
