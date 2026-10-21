# L-B tools notes

## 2026-10-20: initial read

Read KICKOFF.md, LANE-LLAMACPP.md, SPEC.md, FACTS.md sections 1-3, HANDOVER.md, the repository code-review skill, CONTRIBUTING.md, the existing converter and RIDGEFILL.md, the release manifest, and kva/gguf_ckpt.py.

The required `/var/home/dylan/.claude/skills/build-anything/SKILL.md` was unavailable because `.claude` is hidden. Resolved by reading the coordinator's copy at `/var/home/dylan/AI-Work/ridgefill-ports-20261020/skills/build-anything.md`.

Branch is `rf27-tools`; the initial working tree was clean.

Read `rf27-engine:RF27-INTERFACE.md` at engine commit `fffb9069b` when relayed by the coordinator. The converter now follows its format-2 metadata names and types.

## Converter milestone

Implemented the format-1 int8 folder to format-2 GGUF converter with no torch dependency. The legacy Llama/Qwen3 single-file conversion stays supported. All 55 SHA256SUMS entries and every manifest file checksum were verified. The output preserves the complete manifest, source hashes, credit, model metadata, tokenizer hash, layer types and defaults. Each map is dequantized from codes times bf16 scales before Q8_0 conversion. Correction heads are tiled from HF grouped order into llama.cpp order. Biases, correction and score tensors are F32. Format-2 adds the checked GGUF vocabulary fingerprint, authors as a string array, and `adapter.ridgefill.source.<file>` source hashes.

Output: `out/ridgefill-projector-qwen3.8-27b-i8-S16.gguf` (1452188032 bytes). The initial converter artifact (`df3dee45...`) is superseded by the interface-compatible format-2 artifact.

- GGUF SHA256: `8646d41a5a28d60ef53cc5e1fee4b2bfb20d0775db88d3f89d0731ad9f6b629d`
- Source ridgefill.json SHA256: `1ba7dd4a71b20c815c59578e2b5401d510b5cd3ca7e2fa888587b06a869ec558`
- Checked GGUF vocabulary SHA256: `49930cc8a75188cf69a07c57914f0aabe383db11494aa03ed15629adee4c7e33`

Commands run from this folder:

```sh
python3 convert_ridgefill_to_gguf.py /var/mnt/qwen-storage/dylan/AI-Work/ridgefill-q27-projector-20261008/release/ridgefill-projector-qwen3.8-27b-i8/ --model-gguf /var/home/dylan/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf --outfile out/ridgefill-projector-qwen3.8-27b-i8-S16.gguf
python3 tests/ridgefill/check_converter.py /var/mnt/qwen-storage/dylan/AI-Work/ridgefill-q27-projector-20261008/release/ridgefill-projector-qwen3.8-27b-i8/ out/ridgefill-projector-qwen3.8-27b-i8-S16.gguf --model-gguf /var/home/dylan/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf
python3 tests/ridgefill/test_converter.py -v
git diff --check
```

Measured round-trip output:

```json
{"bias_and_score_exact":"PASS","head_order_control":"PASS","maps":48,"max_map_abs_error":0.014892578125,"metadata":"PASS","orientation_control":"PASS","result":"PASS","states":36,"tensors":133,"wrong_map_control":"PASS"}
```

Regression output: `Ran 9 tests ... OK`. Covers F32/F16/BF16/Q8_0 outputs, wrong-map, wrong-head and wrong-orientation controls, corruption preserving existing output, missing file, wrong correction shape removing partial output, base-model metadata mismatch, tokenizer mismatch, GGUF vocabulary hash/count with a NUL-terminator control, required model GGUF, and legacy conversion. Logs are `out/converter.log`, `out/converter-roundtrip.json`, and `out/converter-tests.txt`.

Correction orientation is established by the actual package writer, not by the equal K/V dimensions: `kva/scripts/fit_sterm.py` fits `[H, dk, dv]`; `scripts/q27_package.py:108` writes `ds[L].float().transpose(-1, -2).contiguous()` into `correction.safetensors`. Thus the release already contains `[H, V, K]`. Apply only grouped-to-tiled head reordering for GGUF; do not transpose its last axes again. The G1 oracle must transpose the package tensor back to `[H, K, V]` when setting `Arm.st`. The alpha-1 fixture will use random non-symmetric corrections so an extra or missing transpose fails.

Rejected: loading maps with torch (unnecessary dependency); NumPy safetensors get_tensor for BF16 (unsupported by NumPy). Use safetensors' validated deserialize and expand the BF16 bit representation. Rejected holding all expanded maps in memory: spool tensors beside the destination and process one map at a time. Support the tiny fixture's partial 128-column group with width 64.

Type checking: ty 0.0.84 passed on all three changed Python files, using the system Python and an explicit search path for its user-installed NumPy and safetensors. `git diff --check` passed. The release round-trip is a container/conversion check; G0/G1 engine agreement is NOT RUN.

Next milestone: tiny qwen3_5 and G1, recorded below.

Coordinator assigned L-B the tool-side calls in server, perplexity and bench, including per-request modes and class/digit row selection. L-C retains its server sweep driver function. No tool-side implementation is claimed at this converter milestone.

## Tiny fixture and G1 speed attempt

Produced the 8-layer random Qwen3.5 model (attention at 3 and 7, hidden 64, attention head size 16, 2 KV heads, GDN 2 key heads / 4 value heads of size 16, conv kernel 4). Upstream conversion used F32 and `--no-mtp`; the initial attempt without `--no-mtp` was rejected because this tiny model has no MTP tensors. The synthetic vocabulary is exported as in the Oct-8 gates, and the HF directory contains a valid fast tokenizer with the same IDs for oracle digit selectors.

Artifacts:

- Model: `out/gate1-qwen35/tiny-qwen35-f32.gguf`
- HF checkpoint: `out/gate1-qwen35/tiny-qwen35-hf/`
- Split-3 projector: `out/gate1-qwen35/projector-s3-f32.gguf`, SHA256 `6d0bf255da5c728f1193c525857213262b9ec77a98a00d43a33892210438c781`
- Split-4 projector: `out/gate1-qwen35/projector-s4-f32.gguf`, SHA256 `195b590b75550b14095dd78dcc5554d05b9680294a24482b212635a7e05ea9a0`
- Both splits also have `-wrong-f32.gguf` (wrong map) and `-wrong-st-f32.gguf` (wrong correction orientation) controls, plus their sealed int8 source folders.
- L-C mass reference: `out/gate1-qwen35/qattn-s4-oracle.npz` contains token IDs, layer-3 inputs of the last 64 rows, layer-3 keys, all h_S rows, mass, and the top 100 of 500 prefix positions.

The local `.venv-oracle` has torch `2.14.1+cpu` (`torch.version.cuda` and `torch.version.hip` are None), transformers 5.18.0 and the oracle repository's pinned requirements. Installation logs are under `out/`. The venv is gitignored. Converter fixture serialization was adjusted for compatibility with both safetensors 0.6 and the oracle's 0.8, whose public serialize API changed.

Implemented `run_gate1_qwen35.sh`, a Python driver taking `--split 3|4`, a pinned archive builder, `rf_dump.cpp`, and a Qwen35 oracle/state comparison module. `rf_dump` dumps prefix state and final state separately. This prevents the 200-row tail from erasing the alpha-1 orientation control. The state parser handles the M-RoPE cell extension, the hybrid KV/state split, tiled value heads, conv channels, and key-fastest recurrent state. Its stock integration with actual engine dumps passed after L-A fixed the rebase build error (below); filled-state integration remains NOT RUN.

Commands and results:

```sh
.venv-oracle/bin/python tests/ridgefill/make_tiny_qwen35.py
.venv-oracle/bin/python tests/ridgefill/check_converter.py out/gate1-qwen35/projector-s3 out/gate1-qwen35/projector-s3-f32.gguf --model-gguf out/gate1-qwen35/tiny-qwen35-f32.gguf
.venv-oracle/bin/python tests/ridgefill/check_converter.py out/gate1-qwen35/projector-s4 out/gate1-qwen35/projector-s4-f32.gguf --model-gguf out/gate1-qwen35/tiny-qwen35-f32.gguf
.venv-oracle/bin/python tests/ridgefill/check_fixture.py --kva /var/home/dylan/projects/research/kva-wt-q27
tests/ridgefill/run_gate1_qwen35.sh --kva /var/home/dylan/projects/research/kva-wt-q27 --split 3 --mode speed
```

Both projector round-trips PASS with zero map error; metadata and wrong-map/head/orientation controls PASS. The oracle fixture checks PASS: alpha-1 prepares 27 reference arrays; the wrong correction orientation differs by maximum absolute 0.233485952; split 3 has no lower attention layers, split 4 has layer 3; question mass sums to 255.999999155 (= 4 heads x 64 queries within tolerance), top rows count 100; moving the query window before the prefix end changes mass by maximum absolute 0.500046968; explicit/default end=700 is byte-identical; the digit selector includes token 269 (`Digit7`) and excludes normal token 259.

The speed G1 attempt pinned engine `9e1c3d0764069ed4a5e8b886c45a496241670b6b` (rebased interface-note tip). CMake configure passed; `cmake --build ... -j4 --target llama` FAILED. Exact compiler error:

```text
src/models/qwen3.cpp:87:13: error: return-statement with no value, in function returning 'ggml_tensor*' [-fpermissive]
```

`build_backbone` now returns a tensor, while the old RidgeFill early return was bare `return;`. L-A fixed this in `72d79baa7`. No engine source was changed by L-B. Historical log: `out/gate1-engine/9e1c3d076406/build.log`; historical G1 runner output: `out/gate1-speed-run.log`.

Reran against `72d79baa775d8a7ce7f8d210d1536386b518b4a8` with `--engine-ref 72d79baa7`. CPU engine and `rf_dump` BUILT. The builder uses CMake's actual library output directory `build/bin`; the initial `build/src`/`build/ggml/src` link paths were rejected by the linker and corrected. Stock 700/512 oracle comparison PASSED on all 33 arrays (prefix/final KV, conv and recurrent states and tail logits), maximum relative error `9.938011089616339e-7`, tolerance `1e-4`. The next case, projector loaded with mode off, was refused by the existing loader:

```text
llama_ridgefill_init: failed to load RidgeFill projector: model arch 'qwen35' (or its configuration) is not supported
rf_dump: projector load failed
```

Current runner log: `out/gate1-speed-current-run.log`. Loader log: `out/gate1-qwen35/g1-s3-speed/off-b700-ub512.log`. Built harness: `out/gate1-engine/72d79baa775d/rf_dump`. G1 overall is SKIPPED with a nonzero exit, not PASS: fill agreement, alpha-1 engine orientation, wrong-map engine control, mode-off byte comparison, T>=N and filled-logit refusal remain NOT RUN. All test Python files pass ty; shell syntax and `git diff --check` pass; converter regressions in the new venv report `Ran 9 tests ... OK`.

At that milestone, quality alpha-1 needed a rho reference and quality+qattn needed sweep dump/mass integration. Server/perplexity/bench plumbing and the evaluation client were pending. Subsequent work is recorded below.

## Tool plumbing, evaluation client, and speed G1

Rebased `rf27-tools` onto `rf27-engine` at `72d79baa7`, as requested. Replayed converter/fixture commits became `952a24dd2`, `4191dd49f`, and `e0e1fa9e8`. No engine source was edited.

Implemented common args for mode, share, digits, alpha, qattn share/window/blind; tail and row defaults resolve from the projector through the RF27 C API. Mode off produces no fill boundary. The parser refuses non-finite/out-of-range values, trailing garbage and invalid mode/digit strings by flag name. Added a common-init projector accessor and shared class/digit selection helper. Row split refuses by name.

Server request schema accepts `ridgefill`, `ridgefill_tail`, and `ridgefill_mode`. Quality calls select_rows once per prompt, sets the exact rows after the fill end, and checks API return statuses. RAM cache entries carry mode and selected rows with their fill marker; changing either inside a retained approximate region restarts from zero. This is required because the old fill-end-only cache rule cannot distinguish speed and quality. Existing slot-save refusal remains. Extra /props metadata is emitted only with an enabled projector mode; off responses retain their existing fields. Until L-C's driver is integrated, positive-share quality+qattn refuses at startup or per request by name; zero share uses class/digit selection.

Perplexity selects quality rows per chunk (with its actual BOS token), resets per-sequence markers after each memory clear, and retains effective tail max(T, n_ctx/2). Bench generates a known full prompt before quality selection, resets the marker/rows for each repeated prefill, and names its mode in output. Both refuse quality+qattn by name. These calls use the interface's getters and setters. L-A's actual setters now return int32 status, and all tool/dump calls check it.

Independent C++ checks:

- `out/test-tool-args`: `PASS: 18 RidgeFill argument/boundary checks` (real common.cpp functions, linked with unused sections discarded; no inference stubs).
- Syntax checks passed for common/common.cpp, common/arg.cpp, server-task.cpp, server-schema.cpp, server-context.cpp, perplexity.cpp and llama-bench.cpp. The check supplied declarations from RF27-INTERFACE.md, with setters updated to int32. No fake implementation or successful unsupported path was supplied. Logs: `out/tools-syntax-results.txt`, `out/tools-syntax.log`.
- Full tool linking is BLOCKED: engine `45199c4dc` has not yet declared/defined `llama_ridgefill_select_rows`. L-A must provide that promised C API. No server quality/perplexity/bench integration pass is claimed. Need to rerun full builds against L-A's header once it lands.

Implemented `tests/ridgefill/eval_client.py`: paired same-build mode-off vs candidate NLL, needles, SSE TTFT and native llama-bench wrappers, JSONL provenance, per-length speed summaries, and 95% BCa intervals using kva finals/stats.py (5000 resamples, its declared seed). Existing outputs, missing/non-finite probabilities, inactive eligible fill, no units, unpaired units and undefined intervals refuse. No torch is needed by the client.

NLL scores exactly 512 teacher-forced targets. Each request samples the true next target with a large bias but reads its raw pre-sampling logprob (`post_sampling_probs=false`, `n_probs=1`), over the full vocabulary; common/sampling.cpp copies logits before changing the sampling candidates. The first request turns off cache reuse, subsequent requests reuse the true teacher-forced continuation. The request tail grows so P is constant. Require T-512 >= tail_min and N-512 >= min-prompt. This avoids requesting logits on bulk rows and avoids silently changing the approximation per target.

Needle data are raw prompts plus an appended answer suffix. The client strips answer_len IDs before detokenization, applies the model's chat template through /apply-template with enable_thinking=false, then parses special tokens while tokenizing it. Answers match at digit boundaries. TTFT warms both modes and alternates at least 3 repetitions with prompt caching off; bench alternates the same executable/model and can measure pp or tg128. Synthetic controls are labeled as such and are not release quality/speed numbers.

Evaluation tests:

```text
test_eval_client.py: 7 tests OK
check_eval_live.py: PASS (stock tiny model, 512 targets)
maximum absolute server-logprob vs direct-logit error: 8.506307640487876e-7
off/off repeated logprobs: byte-identical
post-sampling probability control: REFUSED
BCa positive control estimate: 0.025; negative control estimate: 0.0 (degenerate)
```

The live control built the unmodified archived server at `72d79baa7` with -j4. Both instruments used F32 KV, flash attention off, 4 CPU threads. Rejected the initial comparison between default server F16 KV and F32 direct-dump KV: its max logprob error was about 2.3e-4. Matched cache precision instead of widening the tolerance. Logs: `out/eval-live-portable-control.json`, `out/eval-live-portable-control-error.log`, `out/eval-client-tests.txt`. This validates instrumentation only; 27B NLL/needles/TTFT/bench are NOT YET MEASURED.

Commands for the live controls and unit tests:

```sh
.venv-oracle/bin/python tests/ridgefill/test_eval_client.py -v
.venv-oracle/bin/python tests/ridgefill/check_eval_live.py --engine-dir out/gate1-engine/72d79baa775d --kva /var/home/dylan/projects/research/kva-wt-q27
out/test-tool-args
```

Updated RIDGEFILL.md with the Qwen27 section, row rules, planned projector line, modes/knobs, backend status, limits and stock fallback. Every real-model result cell says NOT YET MEASURED. Evaluation commands are in tests/ridgefill/README.md.

### G1 speed on the landed engine

L-A landed the loader/fill in `45199c4dc`. Built a fresh pinned archive and ran:

```sh
tests/ridgefill/run_gate1_qwen35.sh --kva /var/home/dylan/projects/research/kva-wt-q27 --engine-ref 45199c4dc --split 3 --mode speed
```

```json
{"gate":"G1","result":"PASS","checks":27,"split":3,"mode":"speed"}
```

All 27 checks passed: stock oracle comparison, byte-identical mode off, byte-identical T>=N, fill oracle at alpha 0/1, wrong-map rejection at alpha 0/1, wrong-state-orientation rejection at alpha 1, and filled-logit refusal, over 700/512, 128/128 and 50/50. Maximum relative error across correct comparisons was `9.938011089616339e-7`, tolerance `1e-4`. The wrong-orientation control differs by up to about 1.37 relative in prefix recurrent state; it is observed before tail decay. The dump now checks the actual setter return statuses as well.

Artifacts: `out/gate1-engine/45199c4dc*/rf_dump`, `out/gate1-speed-45199-run.log`, and `out/gate1-qwen35/g1-s3-speed/results.json`. The comparison checks 33 arrays for stock and 27 for fill. No GPU was used. Quality/rho and question-attention engine gates remain NOT RUN. Next: L-A supplies select_rows, full tool builds/integration, then quality/sweep G1 and lane Q's real-model measurements.

Final sync for this report: rebased the tools branch onto `bd9ede799` after L-A's rollback-split fix. Current implementation commits are `1978cc39b` (tool plumbing), `c746b7bc6` (eval client/controls), and `d5a050d2c` (Qwen27 docs/status). Recompiled common.cpp against the actual engine header, without the supplied interface declarations: FAILED exactly because `llama_ridgefill_select_rows` is undeclared. Log: `out/tools-actual-api-check.log`. This is the expected dependency named in L-A's 18:25 interface correction; the selector can be supplied independently while quality decode continues to refuse. Full tool builds and app integration remain blocked, rather than being reported as passed.

G1 results in this report are on `45199c4dc`, not a claim about the later rollback commit. Python type checking reports `All checks passed!`; the eval unit suite reports 7 tests OK. The home-layout-audit executable is not available in this sandbox, so that audit was NOT RUN. No files were written outside this worktree and no GPU, Docker, push or publishing action was used.

### Buildable plumbing handoff for L-C

The mode/share/digits/alpha/qattn common fields and server request fields were already committed in `1978cc39b`. Removed the unresolved selector dependency from the active helper: eligible quality requests now refuse with `mode quality requires llama_ridgefill_select_rows, which is unavailable in this build`. Bench and perplexity refuse quality before loading a model; they already refuse quality+qattn. No selector implementation, mock symbol, or successful empty-row substitute is supplied. L-C's guarded server driver-call hunk is unchanged.

Configured a clean CPU build without the temporary API declarations: `cmake -S . -B out/tools-cpu-build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON`. Built `llama-server llama-bench llama-perplexity -j4` successfully against the real engine header/library at base `bd9ede799`. Logs: `out/tools-cpu-configure.log`, `out/tools-cpu-build.log`, `out/tools-cpu-rebuild.log`.

The first live cache control exposed a crash: the old fill cache rule clamped a repeated prompt from its recent recurrent snapshot to position 500, outside the rollback window. Rejected masking the failure or disabling the cache test. The server now preserves the exact cached tail when the approximate boundary is unchanged; a changed boundary restarts recurrent/hybrid contexts from zero when the old approximate prefix would require an unsupported rewind. The direct-logit checks cover speed -> speed, T=200 -> T=201, speed -> off, T>=N, and ridgefill=false with an otherwise unavailable mode.

```text
real CPU build: llama-server, llama-bench, llama-perplexity BUILT
out/test-tool-args-real: PASS: 18 RidgeFill argument/boundary checks
check_tool_live.py: PASS, 30 checks
test_eval_client.py: 7 tests OK
ty 0.0.84: All checks passed!
```

`check_tool_live.py --bin-dir out/tools-cpu-build/bin --dump out/rf_dump-tools` verifies all CLI fields through /props, actual speed/off logits within 1e-5 of native F32 logits, filled counts 500/0/499, named server and command refusals, and finite bench/perplexity measurements. Perplexity speed filled [0,256) and scored one chunk (fixture PPL 330.5232 +/- 3.53619; instrumentation only). Bench uses its supported F16 KV; an initial F32 argument was refused by its existing parser. Raw evidence: `out/tools-live-control.json`, `out/tool-live-control/`, `out/test-tool-args-real.log`, `out/tool-milestone-eval-tests.log`, `out/tools-live-typecheck.log`.

The cached ty is version 0.0.84; an offline request for the CI's 0.0.88 was unavailable. The check used the existing cached executable. Quality/rho/qattn positive inference gates and real-model measurements remain NOT RUN and are not claimed by this handoff.

L-C handoff: buildable tool plumbing is committed at `997892598cb3db27e1ce025a01108c2ab4fff6e1` (common/server fields originated in `1978cc39b`; CPU builds and 30 live + 18 argument checks pass; unavailable modes refuse by name).

### Canonical speed and quality G1 on 3ac42957d (2026-10-20)

Read L-A's dated interface correction: exact-row vectors are a stable graph parameter pointer and graph-reuse key, and the selected-row positions tensor is I32 `[n_R * n_pos]` in M-RoPE section-major order. The public set_rows call used by rf_dump needs no signature change.

Fixed the canonical `run_gate1_qwen35.sh` driver in two places. T>=N supplies no --rows because no prefix positions exist. Quality alpha 1 now requires an actual decode refusal (rc 2 and the phrase `rho-scaled correction`), rather than attempting an unsupported oracle comparison. Speed still compares alpha 0 and 1 and exercises the non-symmetric correction-orientation control. No rf_dump or src changes were needed.

Quality alpha 0 compares all expected arrays against the independent oracle for R=[0,7,31,127,255,499]. Its positive controls require a wrong map, wrong rows [1,6,30,126,254,498], and speed used in place of quality to fail that oracle by more than 1e-2. Rejected treating a successful decode alone as a quality pass. Every batch setting also tests byte-identical off and T>=N, the named alpha-1 refusal, and filled-logit refusal.

Built a clean archive of engine `3ac42957d0bfb7449b11bc77ed82685ef21b7148` with -j4, then ran:

```sh
tests/ridgefill/run_gate1_qwen35.sh --kva /var/home/dylan/projects/research/kva-wt-q27 --engine-ref 3ac42957d --split 3 --mode speed
tests/ridgefill/run_gate1_qwen35.sh --kva /var/home/dylan/projects/research/kva-wt-q27 --engine-ref 3ac42957d --split 3 --mode quality
```

```json
{"checks":27,"gate":"G1","mode":"speed","result":"PASS","split":3}
{"checks":27,"gate":"G1","mode":"quality","result":"PASS","split":3}
```

Both modes passed over batch/ubatch 700/512, 128/128 and 50/50, CPU F32 KV, flash attention off, 4 threads, N=700/T=200. Maximum correct oracle relative error was 9.938011089616339e-7 in each run (tolerance 1e-4). Quality positive controls had max relative errors 1.4805277585983276 (wrong map) and 1.482216715812683 (wrong rows and speed instead of quality). Quality alpha 1 returned rc 2 with the required message in all three settings. Speed wrong-orientation error remained 1.370469093322754.

Evidence: `out/gate1-speed-3ac42957d-run.log`, `out/gate1-quality-3ac42957d-run.log`, `out/gate1-engine/3ac42957d0bf/{build.json,build.log,harness-build.log}`, and `out/gate1-qwen35/g1-s3-{speed,quality}/results.json`. Type check passed (`out/gate1-3ac42957d-typecheck.log`); git diff --check passed. This canonical gate tests explicit rows, not the missing class/digit selector API. Rho correction, qattn, and real-model measurements remain outside these passing verdicts.

### One-command G3 handoff to lane Q (2026-10-20)

Implemented `tests/ridgefill/run_g3.py` with --mode speed|quality|quality+qattn|off, --docs, --needles, --out, plus explicit binary/model/projector/backend knobs. It manages one server and checks the native server/bench build IDs. Both arms use the same process/build/model/projector. Each stage writes metadata, individual paired units, and BCa summaries using kva finals/stats.py (5000 resamples, seed 20261004). Outputs are exclusive: summary.json exists only after requested stages complete; failures produce FAILED.json and a nonzero exit. An explicitly omitted stage is SKIPPED. COMPLETE is not a quality/noninferiority verdict.

NLL scores exactly 512 teacher-forced targets, with n_probs=1 and raw pre-sampling logprob, one target per /completion call. P=N-T remains fixed as the known exact tail grows. Require T-512 >= projector tail_min and N-512 >= min-prompt. The runner reads tail_min from the actual projector GGUF. Native probabilities are never requested for filled prefix rows. Needles strip answer_len IDs before detokenization, use the loaded Qwen chat template with enable_thinking=false and a closed empty assistant think block, and generate greedily. Answers must be seven ASCII digits and match at digit boundaries. Prompt hashes, removed answer counts, thinking/greedy flags and raw answers are retained. TTFT warms both modes and alternates three measurements per length. Native bench alternates prefill and decode runs and refuses unexpected/zero/nonfinite measurements.

Lane Q: after integrating this tools commit and rebuilding the selected engine/tools from one revision, run the following command inside its existing HIP host bridge environment, with cwd `/var/home/dylan/projects/inference/llama.cpp-ridgefill-validate`. The SDK setup/card assignment stays with Q's bridge; this script does not select a physical device.

```sh
python3 tests/ridgefill/run_g3.py --bin-dir build-hip-engine/bin --model /var/home/dylan/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf --projector /var/home/dylan/models/qwen3.8-27b/projector/ridgefill-projector-qwen3.8-27b-i8-S16.gguf --mode speed --docs /var/mnt/qwen-storage/dylan/AI-Work/ridgefill-q27-projector-20261008/evaldata/nll.jsonl --needles /var/mnt/qwen-storage/dylan/AI-Work/ridgefill-q27-projector-20261008/evaldata/needle.jsonl --out out/g3-hip-speed --kva /var/home/dylan/projects/research/kva-wt-q27 --build-label "$(git rev-parse HEAD)-HIP-f16-fa-off" --gpu-layers 99 --flash-attn off
```

For Vulkan, change --bin-dir to build-vulkan-engine/bin and the output/build label accordingly. No Python code changes are required. Defaults measure all 24 documents at N=16384, all 48 needles, TTFT and native prefill at 8192/16384/32768, and native tg128. Each output is paired against mode off within the same build, never the separately compiled upstream build. Run mode off as a negative control into a separate --out directory. Quality and quality+qattn flags are accepted, but they must fail by name on an engine lacking their paths. Bench quality+qattn is deliberately unsupported; when that server path lands, Q can explicitly request --metrics nll,needles,ttft instead of bench. No mode is silently substituted. These real-model measurements remain NOT YET MEASURED here.

End-to-end CPU control command:

```sh
.venv-oracle/bin/python tests/ridgefill/check_g3_live.py --bin-dir out/tools-cpu-build/bin --dump out/rf_dump-tools --kva /var/home/dylan/projects/research/kva-wt-q27 --chat-template-file out/qwen38-chat-template.jinja --work out/g3-cpu-controls3 --port 18095
```

The control extracted the actual 9993-byte Qwen3.8 GGUF template (sha256 12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce) into out/, then used it with the existing tiny model and split-3 projector. Source weights/vocab/projector and the canonical G1 fixture were unchanged. The first attempt used context 1536; the synthetic tokenizer expanded a formatted needle to 3797 tokens, and the server correctly refused it. Increased the tiny control context to 4096 instead of truncating the needle or ignoring the error.

```json
{"result":"PASS","checks":43,"max_logprob_abs_error":9.219664844906106e-7,"documents":2,"scored_per_document":512,"needles":2,"modes":["speed","off"],"metrics":["nll","needles","ttft","bench_prefill","bench_decode"]}
```

All five stages ran end to end in both modes. For each document/arm, all 512 probabilities matched direct F32 logits within 1e-5. Off/off logprobs were identical and NLL delta was 0 (degenerate interval, labeled). Speed's positive-control NLL delta was 0.0034566130489110947, 95% BCa [0.003109533339738846, 0.0038036927580833435]. The two random-model needle accuracy deltas were 0 with degenerate intervals; this is not evidence of useful retrieval. Seven-digit checker controls separately require valid values/leading zeros to match and longer numbers not to match. Bench used supported F16 KV, server/direct-logit comparisons used F32. Native server/bench both report build 11796, commit 22dd8f0da; source flags/logs and the control build label are recorded, not treated as a clean release build.

Evidence: `out/g3-cpu-controls3.json`, `out/g3-cpu-controls3/{speed,off}/summary.json`, per-metric JSONL/provenance/server/native logs in that directory, `out/g3-unit-tests.log` (8 tests OK), and `out/g3-typecheck.log` (All checks passed!, cached ty 0.0.84). Next milestone is G5 archive-build portability and unsupported-combination controls.

Also ran the actual one-command client against unavailable quality and quality+qattn on the CPU tools build. Both exited 1, wrote FAILED.json, and produced no summary.json. The errors named `llama_ridgefill_select_rows` and `question-attention sweep driver`, respectively. Evidence: `out/g3-cpu-controls3/refusals.json` and the two *-refused logs/directories. This is tested refusal, not positive quality/qattn inference coverage.

### G5 implementation and local refusal controls (2026-10-20)

Implemented `run_g5.py`: git archive into a new clean directory, GIT_CEILING_DIRECTORIES to prevent parent-checkout discovery, explicit build commit/count, CPU/Vulkan/HIP selection, -j4 maximum, source audit with rg, positive/negative grep controls, full server/bench/perplexity build, build-ID checks and archived live refusal checks. It cannot provide container isolation in this sandbox; its output states that limitation and leaves multi-GPU NOT RUN. HIP target/toolchain choices are caller arguments; there is no physical-device or TP-degree default.

The G5 requirements exposed two earlier gaps. The old common-init compatibility check ran after loading LoRA, so an invalid/missing adapter could hide the intended unsupported-combination error. Moved explicit LoRA/control/embedding/row-split refusals before model/adapter I/O. Nonzero --cache-reuse now refuses there instead of being silently disabled by server setup. Per-request n_cache_reuse also refuses after normal schema validation. Bench explicitly refuses embeddings as well as row split. Kept a late embedding guard for models that enable embeddings automatically. L-C's guarded sweep call remains untouched.

The new live checker passed 15 checks on the rebuilt CPU tools: five named startup refusals, two bench refusals, a stock missing-LoRA loader control, per-request cache-shift refusal, actual fill activation (500/700 rows), filled-slot save refusal with no output file, zero filled rows in off mode, and successful nonempty exact-slot save. Evidence: `out/g5-local-refusals2.log`, `out/g5-local-refusals2/results.json`, and its per-case logs. Source audit preflight examined 35 files and rg returned 1 with empty output (`out/g5-preflight-grep.log`). Type check passed (`out/g5-typecheck.log`). Full clean archive build will run on this committed script/guard revision next; no archive-build pass is claimed by this paragraph.

### G5 CPU archive result and Qwen27 documentation (2026-10-20)

Committed scripts and fail-closed guards in `44832f7161673ea687d093e7aad6c44045056e3a`, then ran the committed gate with system Python/NumPy:

```sh
python3 tests/ridgefill/run_g5.py --ref 44832f716 --backend cpu --fixture out/gate1-qwen35 --out out/g5-cpu-44832f716 --port 18113
```

```json
{"gate":"G5","result":"PASS","backend":"cpu","revision":"44832f7161673ea687d093e7aad6c44045056e3a","checks":10,"refusal_checks":15,"scope":"archive build, source audit, named refusals on selected backend","isolation":"caller environment; container isolation NOT PROVIDED","multi_gpu":"NOT RUN"}
```

Built all three tools from a clean git archive with -j4, GGML_NATIVE=OFF, GPU backends disabled, tests/examples/UI/OpenSSL disabled. The archive cannot discover the parent checkout; build IDs were explicitly pinned. Server, bench and perplexity all report build 11801, commit 44832f716, GNU 16.2.1, Linux x86_64. The portability grep scanned 35 engine/common/tool/converter/test source files and returned 1 with empty output. Its injected path/hostname/device/TP controls produced four matches; portable code produced none. The archived refusal checker passed all 15 live checks, including actual active fill, refusal without a saved file, and a successful 300876-byte exact-slot save.

Evidence: `out/g5-cpu-44832f716-run.log`, `out/g5-cpu-44832f716/results.json`, `audit.json`, empty `portability-grep.log`, positive/negative grep logs, configure/build command JSON and logs, and `refusals/results.json` plus its per-case logs. The 30 existing live tool checks and 18 argument checks also passed on the local guard build (`out/g5-tool-live.json`, `out/g5-tool-args.log`). Type check and git diff --check passed.

An initial local refusal probe was started before the schema rebuild finished and detected the old request-level cache-shift behavior. Its failed output is retained in `out/g5-local-refusals.log`; the completed rebuild and fresh run in `out/g5-local-refusals2.log` passed. The archive result uses only compiled committed sources, so no stale live binary supplies the accepted result.

For Q, the same command uses --backend vulkan or --backend hip, --gpu-layers 99, a new output directory, the integrated release ref, and explicit toolchain settings through --cmake-arg. The bridge supplies the environment and selected GPU. This lane ran CPU only: clean-container isolation, Vulkan/HIP/CUDA/Metal execution and multi-GPU layer split are NOT RUN here. The scoped CPU pass is not a pass for those remaining G5 hardware/environment gates.

Updated RIDGEFILL.md's Qwen3.8-27B section: distinguish supported speed/off from explicit-row quality engine tests and pending automatic selectors/sweep; state class/digit/qattn row rules and rounding, all knobs/defaults/ranges, exact-tail/cache behavior, startup/request/persistence refusals, quality rho limitation, per-tool mode limits, and G3/G5 commands. Moved the shared flag table there and linked the generic usage section to it to avoid conflicting copies. Real-model result cells remain NOT YET MEASURED; no lane-Q stock/reference numbers were substituted.

### Common class/digit helper for L-C (2026-10-20)

L-A landed the real `llama_ridgefill_select_rows` API in `ff598e03d`. Rebased rf27-tools onto that engine revision, then replaced the temporary common helper's unavailable-API refusal with the real call. This commit makes no tools/server edits and leaves L-C's slot lifecycle/cancellation/deferred-token hunks untouched. Before any later tools/server edits, rebase over L-C's lifecycle commit as the coordinator requested.

Contract: call `common_ridgefill_resolve(params_base, rf)` during initialization, then `common_ridgefill_rows(params_base, rf, tokens.data(), P, effective_request_mode)`. Quality and quality+qattn both return only sorted, unique class/digit base rows. Qattn share/window/blind do not change this base set; the sweep caller computes and unions question rows. Off, speed and an empty prefix return no rows. An active quality request with a missing projector/token buffer, negative prefix, unresolved/invalid share or digit flag refuses by name; a negative/out-of-range API result or unsorted/duplicate/out-of-prefix positions also refuses. No selector logic is duplicated in common and no mock API is supplied.

Built the real engine/common libraries CPU -j4 and compiled `tests/ridgefill/test_row_selection.cpp` against them. With the existing tiny model and split-3 format-2 projector, 21 checks passed: manifest defaults; class-only vs class+digit union; half-even rounding at 0.5 and 1.5; lower-position score ties; digit-only selection at a rounded-zero class budget; quality+qattn base invariance at qattn shares 0/0.2/1; off/speed/empty controls; and named refusals for invalid token IDs/modes/prefixes, missing projector, NaN share and unresolved digit choice. Example tokens [259,261,269,297,299,260,269] select [2,3,4,6] at share 0.5/digits on, versus [2,3,4] with digits off. The 18 existing argument/boundary checks still passed.

```text
cmake --build out/tools-cpu-build --target llama-common -j4: BUILT
out/test-row-selection out/gate1-qwen35/tiny-qwen35-f32.gguf out/gate1-qwen35/projector-s3-f32.gguf
PASS: 21 common class/digit selector checks
out/test-tool-args-real
PASS: 18 RidgeFill argument/boundary checks
```

Evidence: `out/select-helper-build.log`, `out/select-helper-test-build.log`, `out/select-helper-tests.log`, `out/select-helper-argument-tests.log`, and `out/select-helper-rebase.log`. Native test uses real libllama/common and hand-computed expectations, with invalid-token rejection as an API failure control. No HTTP/qattn positive inference claim is made here; L-C owns that integration, and bench/perplexity retain their temporary quality guards until their own follow-up tests.

### Quality tool enablement for lane Q (2026-10-20)

Rebased onto the integration tip `6dac68bca` (which includes the accepted helper commit be962e03a). The common helper was already calling the real selector. Removed the remaining pre-API quality refusal guards from bench and perplexity, so their existing per-prompt/per-chunk selection now reaches `llama_ridgefill_select_rows` with the resolved share/digit args. Exact row counts are logged. Added named alpha/rho validation in common default resolution and in the effective-mode helper, covering startup mode and per-request quality overrides. Nonzero quality alpha and unsupported qattn still refuse. No tools/server source or lifecycle hunk was edited.

Perplexity rejects non-unified parallel quality chunks with the required --kv-unified (-kvu) flag. A failed RF decode (PPL or KL) throws into its named error exit; a PPL run that scored no chunk or returned a missing/nonfinite/nonpositive score exits nonzero. Bench catches selector errors and returns failure instead of an uncaught exception. Retained the G5 LoRA/control/embedding/cache-shift/row-split/persistence guards.

Built all three tools with CPU -j4. The updated live tool check passed 37 cases. Quality bench completed pp700 with 445 exact rows of 500 prefix rows; quality perplexity selected 255 of 256 prefix rows and completed one chunk with fixture PPL 326.5109 +/- 2.81513. Both use share 0.5, digits on and alpha 0. Alpha 1, qattn, --no-kv-unified parallel chunks and a too-short document all produced named exit-1 refusals. One initial refusal control used perplexity's default batch 2048/context 512 and correctly hit the parallel-cache refusal before alpha; the final control explicitly uses batch 512 to isolate rho. Failed probe logs are retained rather than counted as passes.

Extended `check_g3_live.py` to run the one-command paired pipeline for quality as well as speed/off. Its independent Python selector reads the actual GGUF score and vocabulary/type tables, applies the finite-score ranking/half-even budget/digit union, and passes those explicit rows to rf_dump for the native reference. The first probe found that GGUFReader represents token-type arrays as scalar parts, not one contiguous part; corrected the test reader and reran in a fresh output directory. No engine behavior/tolerance was changed to hide the test preparation error.

```text
CPU llama-server, llama-bench, llama-perplexity: BUILT
check_tool_live.py: PASS, 37 checks (off/speed/quality runs, rho/qattn and negative controls)
check_g3_live.py: PASS, 66 checks, modes speed/quality/off, all five measurement stages
maximum server raw-logprob vs direct-logit absolute error: 1.1241689676921851e-6 (tolerance 1e-5)
test_row_selection.cpp: PASS, 24 checks (includes startup/request alpha-rho refusals)
test_tool_args.cpp: PASS, 18 checks
test_eval_client.py: 8 tests OK
ty: All checks passed!
```

Each mode's G3 run completed paired NLL (2 documents, exactly last 512 targets), Qwen3.8 thinking-off greedy needles (2), TTFT, native prefill and tg128 with BCa via kva finals/stats.py. All 512 logprobs of every document/arm matched direct F32 logits. Off/off NLL arrays were identical. Quality was distinct from speed as a positive control. Quality's synthetic NLL delta was 0.0030389586463570595, 95% BCa [0.0027102669700980186, 0.0033676503226161003]; random-model needle deltas were degenerate and are not retrieval evidence. These are instrumentation controls, not 27B release measurements.

Commands/evidence: `cmake --build out/tools-cpu-build --target llama-server llama-bench llama-perplexity -j4`; `check_tool_live.py --bin-dir out/tools-cpu-build/bin --dump out/rf_dump-quality-tools --work out/quality-tool-live3 --port 18128`; `check_g3_live.py --bin-dir out/tools-cpu-build/bin --dump out/rf_dump-quality-tools --kva /var/home/dylan/projects/research/kva-wt-q27 --chat-template-file out/qwen38-chat-template.jinja --work out/g3-quality-tools2 --port 18123`. Logs: `out/quality-tools-{build,rebuild,typecheck}.log`, `out/quality-perplexity-final-build.log`, `out/quality-tool-live3.json` and its per-case directory, `out/g3-quality-tools2.json` and its mode directories, `out/quality-row-selector-tests.log`, `out/quality-argument-tests.log`, `out/quality-eval-unit-tests.log`. Q can now use the same G3 command with --mode quality and --alpha 0, including native bench metrics, against the integrated rebuild. Real-model cells remain NOT YET MEASURED pending Q's own results.

Quality enablement is committed at `23da65965e93f361149f0f6fe18a90c062d38d29`. Resumed G5 on that exact commit:

```sh
python3 tests/ridgefill/run_g5.py --ref 23da65965 --backend cpu --fixture out/gate1-qwen35 --out out/g5-cpu-23da65965 --port 18130
```

```json
{"gate":"G5","result":"PASS","backend":"cpu","revision":"23da65965e93f361149f0f6fe18a90c062d38d29","checks":10,"refusal_checks":15}
```

The archive source cannot discover the parent worktree. Its source grep covered 36 files and returned 1 with zero output; injected-assumption and portable-code controls passed. Built server/bench/perplexity with CPU -j4 and all three report build 11812, commit 23da65965. All 15 live refusal/persistence controls passed again against the archived binaries. Evidence: `out/g5-cpu-23da65965-run.log`, `out/g5-cpu-23da65965/{results.json,audit.json,portability-grep.log,build.log,refusals.log}` and its refusal per-case logs. Container isolation, GPU backends and multi-GPU layer split remain NOT RUN in this lane; no broader G5 claim is inferred from the scoped CPU pass.

## 2026-10-20: final Qwen3.8 docs and GPU G5 handoff

Rebased onto integration f06e89aea before this work. RIDGEFILL.md now describes all three modes, the per-prompt class/digit and 20%-question unions, last-64 causal mass at layers 3/7/11/15 below split 16, zero-share quality identity, the per-slot F16 h_S replay, 1 GiB bound, startup/request flags, named unsupported combinations, projector provenance and backend/build matrix. The question-attention implementation/CPU gate evidence is attributed to rf27-sweep NOTES-sweep.md (native 25 arrays, HTTP 17 comparisons, cancellation/reuse/shutdown); it was read from that branch, not rerun or claimed as this lane's new test. All 27B metric/backend cells remain NOT YET MEASURED. No tools/server or src file was edited.

Added run_g5_gpu.sh (both backends use kind=host), check_gpu_smoke.py, run_g5_two_card.sh, and GPU evidence parser/refusal tests. Extended the existing run_g5.py archive gate with optional --smoke-model/--smoke-projector: the overall archive result cannot report PASS if its real-model smoke fails. Each single-card wrapper performs a clean git archive build (-j4, GGML_NATIVE=OFF), portability grep and live tiny refusal checks, then exactly one short pp8192 speed prefill (T4096, alpha0, F16 KV, FA off, b512/ub128, threads4, no warmup). The archived revision must include this handoff. Both scripts refuse CPU fallback, missing/wrong devices, nonfinite samples, a wrong build, a non-27B model or inactive fill. A bench stderr completion marker is emitted after measured repetitions, outside their timing, so the script checks that a real filled prefix completed rather than inferring success from an exit code alone.

The coordinator's separate two-card script reuses that clean Vulkan build and discovers both R9700 names at runtime, requests -sm layer -ts 1/1, and requires nontrivial model buffers on both GPUs. No device ID, visible-device mask, hostname, user path or TP degree is fixed in production RidgeFill code. HIP takes its SDK directory explicitly, verifies .info/version is 7.10.0*, uses the exact ROCM_PATH/HIP_PATH/CMAKE_PREFIX_PATH/PATH/LD_LIBRARY_PATH exports from KICKOFF and builds gfx1201. The campaign SDK version is ROCm 7.10.0a20251120; hipcc-version.log and rocm-sdk-version.log capture the observed toolchain. The container is not used because it lacks hipBLAS/rocBLAS.

CPU validation performed:

```text
CPU llama-bench: BUILT (-j4)
Tiny speed: PASS, finite measurement, completed 500 filled prefix / 200 exact tail rows
Tiny off: PASS, finite measurement, completion marker absent
Tiny quality: PASS, finite measurement, completed 500 prefix / 200 exact tail rows
GPU parser/refusal unit tests: 5 tests OK (synthetic evidence; NOT a GPU gate)
Real 27B/projector metadata preflight: PASS (read-only CPU)
Tiny model passed to real-model preflight: REFUSED by name
CPU-only executable passed to Vulkan smoke: REFUSED by name, no devices, rc 1
Wrapper/CLI refusal controls: 5 passed (usage, missing SDK, unpaired paths, CPU smoke)
Portability grep: 44 sources, zero output, rg exit 1
bash -n: PASS, both new wrappers
ty: All checks passed!
git diff --check: PASS
```

The first preflight rejected the actual GGUF because its block_count is 65, including one nextn_predict_layers MTP layer. Corrected it to compare the 64-layer trunk (block_count minus nextn_predict_layers), matching libllama; the real model/projector preflight then passed. Retained the no-device failure log at out/gpu-smoke-no-gpu-control.log and its FAIL results.json; it is a refusal control, not a GPU test. Parser test log: out/gpu-smoke-parser-tests.log. Tiny actual measurements/logs: out/gpu-smoke-tiny-{speed,off,quality}.{json,log}. Build: out/gpu-smoke-cpu-build.log. Source audit: out/gpu-g5-portability-grep.log. Rejected accepting a one-card run as a two-card smoke, accepting a CPU backend with positive throughput, and using the HIP container or fixed GPU IDs. No GPU/Vulkan/HIP execution, full-27B benchmark, two-card execution or container isolation was possible in this lane; all remain NOT RUN here.

Exact lane-Q bridge commands (submit each as a separate kind=host job, timeout 14400; pin RF_REF after L-A merges this handoff, and create/link the shared tiny fixture at out/gate1-qwen35 if absent):

```sh
RF_REF=$(git rev-parse rf27-integ)
RF_MODEL=/var/home/dylan/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf
RF_PROJECTOR=/var/home/dylan/models/qwen3.8-27b/projector/ridgefill-projector-qwen3.8-27b-i8-S16.gguf
tests/ridgefill/run_g5_gpu.sh vulkan "$RF_REF" out/gate1-qwen35 "$RF_MODEL" "$RF_PROJECTOR" "out/g5-vulkan-$RF_REF"
```

```sh
RF_REF=$(git rev-parse rf27-integ)
RF_MODEL=/var/home/dylan/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf
RF_PROJECTOR=/var/home/dylan/models/qwen3.8-27b/projector/ridgefill-projector-qwen3.8-27b-i8-S16.gguf
RF_SDK=/var/home/dylan/projects/inference/strata/.venv/lib64/python3.14/site-packages/_rocm_sdk_devel
tests/ridgefill/run_g5_gpu.sh hip "$RF_REF" out/gate1-qwen35 "$RF_MODEL" "$RF_PROJECTOR" "out/g5-hip-$RF_REF" "$RF_SDK"
```

Coordinator outside the bridge, in lane Q's folder with the Vulkan archive build and both cards visible (set RF_REF to the exact same pinned value as that build; no rebuild):

```sh
RF_MODEL=/var/home/dylan/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf
RF_PROJECTOR=/var/home/dylan/models/qwen3.8-27b/projector/ridgefill-projector-qwen3.8-27b-i8-S16.gguf
tests/ridgefill/run_g5_two_card.sh "out/g5-vulkan-$RF_REF/build/bin" "$RF_REF" "$RF_MODEL" "$RF_PROJECTOR" "out/g5-two-card-$RF_REF"
```

The scripts never submit bridge jobs themselves. Lane Q/coordinator supply access and record their actual GPU outcomes. The pinned archive version checks prevent a changed branch or stale executable from being reported as the selected build.

## 2026-10-20: remove machine policy from the GPU G5 tools

Coordinator relayed L-A's independent audit on integration fc3ee1cbb. The prior source audit was incomplete: it found paths/device IDs/TP assumptions but had no patterns for GPU models, target names or SDK releases. Removed the machine policy rather than excluding the new scripts from the audit. Rebased onto that integration tip; no src or server source changes.

check_gpu_smoke.py now accepts any GPU descriptions by default, including any two Vulkan devices for --two-card. Optional --device-name-pattern REGEX restricts descriptions; both host wrappers accept it. The HIP target list is optional --hip-targets TARGETS (semicolon-separated); omission leaves CMake target detection intact. The optional SDK path keeps positional compatibility and also supports --rocm-sdk; default discovery uses ROCM_PATH, HIP_PATH, then hipconfig -R. SDK version validation moved into run_g5.py and derives the minimum from the pinned archive's ggml-hip/CMakeLists.txt. Any newer SDK is accepted. Optional --sdk-min-version VERSION raises that floor and cannot lower the backend requirement. SDK exports are prepared for child commands and preserve existing library search paths. Neither a GPU model/architecture nor a particular SDK release is required by code.

Extended the audit with device-model names, architecture targets, SDK version strings, and both full and short PCI slot IDs. Its synthetic controls detect all new classes. It also rejects the actual old GPU scripts extracted from 20261fbe3 (regression control). Generic device inventories and loopback addresses are negative controls; loopback IPv4 is not mistaken for an SDK version. An initial regex used unsupported lookahead in rg's default engine; replaced it with ordinary delimiters, retaining the loopback negative control. No PCRE dependency or exclusions were added.

```text
GPU evidence/SDK/audit unit tests: 7 passed
Wrapper/CLI invalid-input controls: 7 passed
Real installed SDK: accepted above backend minimum; optional higher minimum refused (no HIP execution)
Extended audit detects original machine-pinned scripts: PASS
Extended RidgeFill manifest audit: 44 code files; rg rc 1; empty output
Extended ALL changed-code audit vs upstream 1242be66f: 72 files; rg rc 1; empty output
bash -n: both wrappers pass
ty: All checks passed!
git diff --check: PASS
```

Evidence: out/g5-neutral-gpu-tests.log, out/g5-portability-extended.log, out/g5-portability-all-changed.log, out/g5-portability-all-changed-files.json, out/g5-extended-old-regression.log. The first SDK export probe compared a lib64 spelling with the resolved lib spelling and failed its test assertion; the corrected probe compares canonical paths and passes. It did not execute a compiler/GPU. No GPU, full archive rebuild or numerical rerun was needed or performed for these tool-policy changes.

This box's settings are explicit caller policy, only in these examples. After merging, pin REF to the integration SHA and use the existing MODEL/PROJECTOR/SDK paths from the previous handoff. Lane Q submits each single-card command as its own kind=host bridge job:

```sh
tests/ridgefill/run_g5_gpu.sh vulkan "$REF" out/gate1-qwen35 "$MODEL" "$PROJECTOR" "out/g5-vulkan-$REF"
tests/ridgefill/run_g5_gpu.sh hip "$REF" out/gate1-qwen35 "$MODEL" "$PROJECTOR" "out/g5-hip-$REF" "$SDK" --hip-targets gfx1201 --sdk-min-version 7.10.0
```

Coordinator outside the bridge, same Vulkan archive build and exact REF:

```sh
tests/ridgefill/run_g5_two_card.sh "out/g5-vulkan-$REF/build/bin" "$REF" "$MODEL" "$PROJECTOR" "out/g5-two-card-$REF" --device-name-pattern 'R9700|Radeon AI PRO 9700'
```

Without those optional arguments, the scripts accept any matching GPU model, CMake-detected targets and any SDK meeting the backend minimum. Real backend outcomes remain for lane Q/coordinator to measure; no new hardware result is claimed here.
