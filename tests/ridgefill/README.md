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

The same test covers the lexical net on synthetic multikey-style prompts: case/leading-space normalization, the needle plus its 32-row value span, distractor exclusion, one/two/three prefix occurrences, ChatML boundaries and selector endpoint exclusion, longest-first admission under the 5% cap, report attribution and exact-row budget reservation. It also checks named refusals. It runs on CPU and needs no full-model weights.

```sh
c++ -std=c++17 -O2 -Icommon -Iinclude -Isrc -Iggml/include tests/ridgefill/test_row_selection.cpp -Lbuild-cpu/bin -Wl,-rpath,"$PWD/build-cpu/bin" -lllama-common -lllama -lggml -lggml-base -o build-cpu/test-row-selection
build-cpu/test-row-selection /path/to/tiny-qwen35-f32.gguf /path/to/projector-s3-f32.gguf
```


Speed cases compare tail logits, every layer's final KV/conv/recurrent state, and every late layer's prefix state at N=700/T=200 over batch/ubatch pairs 700/512, 128/128 and 50/50. Prefix dumps are necessary to expose a wrong alpha correction before tail decay. Alpha is 0 and 1. Controls require byte-identical mode-off and T>=N outputs, a wrong map that fails by over 100x tolerance, a wrong correction orientation at alpha 1, and refusal of a filled row asking for logits. The tolerance is 1e-4 relative F32.

The same harness is the canonical G1 for quality with hand-picked rows through the RF27 API. It compares alpha-0 inference against the oracle and requires wrong maps, wrong exact rows and speed-instead-of-quality to fail that comparison. Its T>=N control supplies no exact rows because the prefix is empty. Quality alpha 1 is an expected refusal (rc 2 naming the rho-scaled correction), not a passing inference comparison. Speed retains its alpha-0/1 inference and wrong-correction-orientation checks. The canonical shell harness still returns nonzero SKIPPED for quality+qattn. L-C's separate native/HTTP sweep gates now pass on split 4, including real class/digit selection, question mass/rows, 25 cache/state/logit arrays, zero-share equality, cancellation/reuse and shutdown. Those gate measurements are recorded in NOTES-sweep.md on rf27-sweep; the shell harness does not claim to execute them.

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

It extracts git archive into a clean source directory, prevents Git from discovering the parent checkout, pins the build commit/count, builds server/bench/perplexity with at most -j4, checks their build IDs, audits the RidgeFill source files with ripgrep when available or a Python standard-library scan otherwise, and runs the live refusal controls. The source list and patterns are in audit.json; portability-grep.log must be empty (scan status 1). Positive grep controls insert an absolute user path, hostname, device ID, TP-size assumption, GPU model/target, SDK version and full/short PCI slot IDs; the negative control is portable code. Logs and exact build commands stay under --out. Existing output directories refuse.

Use the same script in lane Q's bridge with --backend vulkan or --backend hip and --gpu-layers 99. HIP toolchain/target settings are explicit --cmake-arg=-DOPTION=VALUE arguments inherited from the bridge setup. The gate does not select GPU IDs or assume a target architecture/TP degree. Its audit patterns also detect machine GPU model names, compiler targets, SDK versions and PCI slot IDs. It does not provide container isolation; run the command in a clean container when one is available. Its result covers only the selected backend, archive build, source audit and refusal tests; multi-GPU layer split remains a separate hardware gate.

`check_refusals.py` can also run against an existing build:

```sh
python3 tests/ridgefill/check_refusals.py --bin-dir build/bin --fixture out/gate1-qwen35 --work out/g5-refusals
```

It requires named startup refusals for LoRA, control vectors, embeddings, nonzero --cache-reuse and -sm row; bench must refuse embeddings and row split. Per-request n_cache_reuse must refuse. A live speed prompt must fill before slot save is refused without creating a file. Mode off must fill nothing and save a nonempty exact slot. Missing LoRA/control files are intentional: the RidgeFill combination must be rejected before file I/O, while stock with the same missing LoRA must reach its normal loader. Ordinary cache_prompt reuse and RAM prompt caching remain supported; cache-reuse means KV shifting.

### GPU G5 commands

Use the host bridge for both backends. `run_g5_gpu.sh` takes a pinned commit, the tiny refusal fixture, the real 27B model/projector and a new output directory. The archived commit must contain these scripts and the bench completion marker. It runs the archive build/source audit/refusal gate, then one pp8192 speed-mode llama-bench with T=4096, alpha 0, F16 KV, FA off, batch 512, ubatch 128, four threads and no warmup. This is a short execution smoke, not a timing table or numerical equivalence gate.

```sh
# Each command runs as one kind=host bridge job, with its existing one-card pinning.
tests/ridgefill/run_g5_gpu.sh vulkan COMMIT out/gate1-qwen35 model.gguf projector.gguf out/g5-vulkan-COMMIT
tests/ridgefill/run_g5_gpu.sh hip COMMIT out/gate1-qwen35 model.gguf projector.gguf out/g5-hip-COMMIT /path/to/rocm-sdk
```

HIP uses a host SDK containing HIP, hipBLAS and rocBLAS. Its SDK directory is optional: use the positional SDK argument or --rocm-sdk, otherwise discovery checks ROCM_PATH, HIP_PATH, then hipconfig -R. The gate exports ROCM_PATH, HIP_PATH, CMAKE_PREFIX_PATH, PATH and LD_LIBRARY_PATH for its subprocesses. It accepts any SDK at least as new as the backend's minimum in ggml/src/ggml-hip/CMakeLists.txt (currently 6.1). --sdk-min-version VERSION can require a newer SDK; it cannot lower the backend minimum. No SDK release is pinned in the scripts. --hip-targets 'TARGET_A;TARGET_B' supplies explicit compiler targets; omitting it leaves target detection to CMake. --device-name-pattern REGEX optionally filters descriptions, defaulting to any GPU. `hipcc-version.log` and `rocm-sdk-version.log` preserve observed compiler/SDK evidence. The wrapper never changes visible-device variables; lane Q's bridge selects its single GPU. This box's paths, targets, SDK version and commands live in NOTES-tools.md.

The smoke checks the model's qwen35 architecture, 64 trunk layers (excluding an optional MTP layer) and width 5120, the projector's split/tail metadata, the pinned build, exactly one visible backend device, finite positive samples, actual model tensors on that GPU and the completed filled prefix. It records model/projector SHA256, device inventory, benchmark JSON, stderr and argv. Missing GPU, CPU fallback, inactive fill and an unexpected model fail nonzero. Archive and smoke outputs share one results.json; a smoke failure cannot leave a passing overall G5 result.

The coordinator runs this separately outside the single-card bridge, reusing the clean Vulkan build:

```sh
tests/ridgefill/run_g5_two_card.sh out/g5-vulkan-COMMIT/build/bin COMMIT model.gguf projector.gguf out/g5-two-card-COMMIT
```

This discovers exactly two visible Vulkan devices of any model, requires at least 3 GiB free on each, and passes their discovered names with `-sm layer -ts 1/1`. Append `--device-name-pattern REGEX` to restrict the accepted descriptions, including when other devices are visible. It refuses a one-card bridge or a device set that does not contain exactly two matches. Both devices must hold nontrivial model buffers; a single-card run cannot pass the two-card check. Backend IDs and visible-device masks are never fixed in these scripts. Two-card output is separate from the one-card archive result; no container isolation or long-context quality claim follows from a smoke.

GPU evidence/refusal unit tests need Python 3.10 or newer and its standard library. Ripgrep (`rg`) is optional: when it is absent, the audit scans the same UTF-8 sources and case-insensitive patterns with Python `re`, reports file/line matches, and keeps the same status convention (0 matches found, 1 clean). Read/decoding failures and errors from an installed ripgrep still fail the audit. The suite explicitly exercises the no-ripgrep fallback. Full archive builds also require Git, tar, CMake, a compiler and the refusal checker's dependencies described above.

CPU validation of evidence/refusal parsing uses synthetic GPU records and explicitly makes no GPU execution claim:

```sh
python3 tests/ridgefill/test_gpu_smoke.py -v
bash -n tests/ridgefill/run_g5_gpu.sh tests/ridgefill/run_g5_two_card.sh
```

## Experimental chat-row plumbing

The extended common selector forwards special-exact (default off) and header-rows (default 0) to libllama; the defaults retain the previous rows. Common CLI, bench, perplexity, scorer and server request fields apply these choices in quality modes. Speed refuses active chat-row options. Per-request changes recompute incompatible retained rows; request off remains stock. Window mode last/turn and turn-cap (default 256) are passed to the sweep. Bench/perplexity refuse turn mode because they do not run that sweep.

```sh
.venv-oracle/bin/python tests/ridgefill/check_chat_tools.py --bin-dir build/bin --fixture out/gate1-qwen35 --kva /path/to/kva --out out/chat-tools
```

This CPU gate uses a synthetic multi-turn conversation in the existing tiny vocabulary, not lane T's frozen data. It checks independent special/header row unions, prefix clipping, unchanged question mass with the correct additional exact-row union on split 4, server startup inheritance and per-request overrides, cache changes against native target logprobs, named invalid/unsupported options, actual quality bench/perplexity measurements, and paired scorer flag propagation. It does not claim full-model chat quality. The existing four-mode scorer oracle gate checks the unchanged defaults.


The native ChatML gate in tools/ridgefill/check_chat_score.py first creates chat.gguf and projector-s4.gguf in its work directory. This separate HTTP gate uses those files and compares per-request turn-cap, window mode and special/header overrides to fresh native scoring, including missing/empty-user and fractional/overflowing count refusals, followed by recovery:

```sh
.venv-oracle/bin/python tests/ridgefill/check_turn_http.py --bin-dir build/bin --fixture out/gate1-qwen35 --chat-fixture out/score-chat-gate --work out/turn-http
```

G5 startup checks now give each unsupported combination five seconds, use nonexistent model/projector/adapter paths and an invalid device name, and require the RidgeFill refusal before either model or device loading. Server, perplexity and bench each reject LoRA, control vectors, embeddings, nonzero cache reuse and row splitting. Flags are tested before and after --ridgefill. The no-projector LoRA loader and live filled/exact slot-save controls remain active.


## Skipped-work reporting gate

```sh
.venv-oracle/bin/python tests/ridgefill/check_reporting.py --bin-dir build/bin --fixture out/gate1-qwen35 --chat-fixture out/score-chat-gate --dump out/rf_dump_rs --kva /path/to/kva --out out/reporting
```

The chat fixture comes from tools/ridgefill/check_chat_score.py. Build rf27-gates/rf_dump_rs.cpp with -DRF27_API against the same engine to provide --dump. The gate compares source breakdowns with actual select_rows_ext row sets, analytic budgets with the read-only kva/compute.py functions across off/speed/quality/chat/qattn and two turn caps, and the all-exact prefix control that still pays selector cost. It compares native/server and cached reports, checks fully/partially exact caches, completion/chat usage including streams, two bench repetitions and an actually scored perplexity chunk. The G3 live gate verifies that all per-request records and statistical summaries retain the reports. No GPU or full-model quality claim follows from these CPU controls.

`check_chat_score.py` also checks class share 0 for qattn-only, special-only, digits-only, qattn+special, header-only and empty selection against `kva.select`/`finish()` across three chunk sizes (18 additional comparisons). `check_zero_share.py` uses that gate's `--work` output as `--chat-fixture`; it checks empty selection's logits and prefix/final cache bytes against speed, HTTP class-share overrides and named invalid-share refusals, and bench/perplexity with a zero-default projector. Run with `--bin-dir BUILD/bin --fixture TINY_FIXTURE --chat-fixture CHAT_GATE_WORK --dump RF27_API_DUMP --out NEW_OUTPUT_DIR`. HTTP null values inherit the server default, following the common request schema.

S1/vtail tests: compile `test_s1_reporting.cpp` against common/engine libraries and run through `check_s1_reporting.py --binary TEST_BINARY --fixture TINY_FIXTURE --kva REFERENCE_REPO --out NEW_OUTPUT_DIR`. This checks reporting math on real model dimensions, not S1 inference: 16 split/window/endpoint/vtail configurations against independent kva.compute formulas, including the recomputed S1_FILL attention K/V, dense padded/masked probe pairs, and two calibration V projections. `check_s1_tools.py --bin-dir BUILD/bin --baseline-binary ISOLATED_OLD_SCORER --fixture TINY_FIXTURE --chat-fixture CHAT_GATE_WORK --out NEW_OUTPUT_DIR` checks disabled-default bytes, API-independent refusals and existing unsupported combinations with each feature enabled. `--dense-model DENSE_GGUF --dense-projector DENSE_PROJECTOR` adds three named qwen35-only refusal controls; missing fixtures are explicitly SKIPPED. `check_s1_http.py --bin-dir BUILD/bin --fixture TINY_FIXTURE --chat-fixture CHAT_GATE_WORK --kva REFERENCE_REPO --out NEW_OUTPUT_DIR` checks enabled S1, vtail, their combination, HTTP/native logits and row counts, dense MAC costs, rank rounding/saturation, cache invalidation and named bounds refusals. The native/frozen-reference gate is `tools/ridgefill/check_s1_score.py` (see its README). The release remains ON HOLD; these are CPU tiny-model checks.

`check_scheduling.py --config CONFIG.json --out NEW_OUTPUT_DIR` starts a server from the config's `command` array, streams two requests from its `background` token array, then submits its `prompt` token array to slot 2. Supply `batch`, three slots, unified KV, a large enough context and `--verbose`. It requires the sweep to advance in consecutive scheduling iterations, charges at most batch-2 rows, and completes the target while both generators remain active. It records stream gaps for a separate stock-chunk comparison; it has no machine-specific timing threshold. `expect_steps: false` supports a stock control; `expect_error` checks a named refusal, including an atomic vtail sample that cannot fit beside the generators. Optional `vram` names a host-readable byte counter. Example configs and raw lane-A runs are under `out/a2/`; no model is downloaded.
