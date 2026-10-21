# L-C sweep notes

## 2026-10-20: storage and question-mass component milestone

Implemented `src/llama-ridgefill-sweep.{h,cpp}` and registered the source in `src/CMakeLists.txt` under the coordinator's explicit ownership exception. No L-A implementation files were edited. This is a library component milestone, not integrated G1 or a working server mode.

The buffer owns its backend storage and ggml metadata, defaults to F16 with a 1 GiB payload cap, and rejects invalid dimensions, unsupported types, oversized prompts, missing buffer types, allocation failures, and out-of-range slices by naming `quality+qattn`. Dimension products are checked before allocation. The caller supplies the buffer type of layer S; no host fallback occurs. Capture adds one graph copy per ubatch and slice returns a view. There is no per-token host loop or synchronization in these helpers. The slot must retain the buffer until all graphs using it complete.

The question graph reuses `build_norm`, `build_qkv`, and `ggml_rope_multi` from the stock qwen35 path. It computes Q/G jointly, uses only Q, normalizes and rotates it, then computes grouped causal attention against FP32 keys in logical prompt order. It reduces each KV-head group's mass to FP32 `[N, Hkv]`; the caller must accumulate groups and layers in FP64, matching `window_mass`. K/V projection nodes built by `build_qkv` are not expanded into this graph. The standalone tested graph has 16 nodes.

The row selector uses a prefix-relative budget, ties-to-even rounding, score order followed by lower position, and returns rows sorted by position. It refuses non-finite/negative mass and invalid shares. The server must bypass all sweep allocation/computation when qattn share is zero; that integration is not implemented here yet.

## Requirements and checks

| Requirement | Check | Result |
|---|---|---|
| CPU library builds, maximum four build jobs | `cmake --build build-sweep --target llama -j4` | PASS |
| Bounded storage, conversion/capture/replay, chunk invariance, corruption control | `build-sweep/check-sweep` | PASS |
| Question graph agrees with `select.window_mass` within 1e-4 | `check-frozen.py`, L-B shared split-4 oracle | PASS: graph max relative error 1.763e-7; live FP32 cache 3.017e-7 |
| h_S capture of a tiny split-4 lower pass matches PyTorch conversion | `check-frozen.py`, L-B frozen h_S, chunk sizes 700/128/50 | PASS |
| Engine lower-only graph stops at S and captures its own h_S | L-A dispatch/context hook plus L-B fixture | SKIPPED: touch points are specified but not implemented yet |
| Engine cache mapping and layer-input extraction feed the mass graph | `check-cache.cpp` plus `check-cache.py`, L-B tiny GGUF | PASS using stock full decode; LOWER dispatch still SKIPPED |
| Server driver compiles and refuses unavailable stages by name | `cmake --build build-sweep --target server-context -j4`, `build-sweep/check-driver` | PASS for the skeleton; server call awaits L-B mode plumbing |
| Zero qattn share uses quality without sweep allocation or context access | `build-sweep/check-driver`, replaced allocator and null-context control | PASS routing; quality cache/logit equivalence still SKIPPED |
| Sweeps 2-3, server driver, G1 caches/logits, qattn-share-zero equals quality | L-A late fill passes G1, then integrated sweep gate | SKIPPED |
| Device placement on non-CPU backends | Lane Q hardware gates after integration | SKIPPED: CPU only in this lane |

A component PASS above does not claim the skipped integration requirements passed. No unfinished public API or mode was added.

## Build and test evidence

Configure:

```sh
cmake -S . -B build-sweep -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-sweep --target llama -j4
```

Final build output:

```text
[ 13%] Built target ggml-base
[ 35%] Built target ggml-cpu
[ 39%] Built target ggml
[ 41%] Building CXX object src/CMakeFiles/llama.dir/llama-ridgefill-sweep.cpp.o
[ 43%] Linking CXX shared library ../bin/libllama.so
[100%] Built target llama
```

The temporary harness files are inside the ignored `build-sweep/` directory, since L-B owns `tests/ridgefill`. L-B can adopt `check-sweep.cpp`, `check-question.cpp`, and `check-mass.py`. The C++ programs link the built library, rather than substituting a separate implementation:

```sh
c++ -std=c++17 -O2 -I src -I ggml/include build-sweep/check-sweep.cpp -L build-sweep/bin -Wl,-rpath,'$ORIGIN/bin' -lllama -lggml -lggml-cpu -lggml-base -o build-sweep/check-sweep
c++ -std=c++17 -O2 -I src -I include -I ggml/include build-sweep/check-question.cpp -L build-sweep/bin -Wl,-rpath,'$ORIGIN/bin' -lllama -lggml -lggml-cpu -lggml-base -o build-sweep/check-question
build-sweep/check-sweep
PYTHONDONTWRITEBYTECODE=1 python3 build-sweep/check-mass.py /var/home/dylan/projects/research/kva-wt-q27
```

Logs: `build-sweep/build.log`, `build-sweep/check-sweep.log`, `build-sweep/check-mass.log`.

```text
PASS storage f16: capture/replay, chunking, corrupted-input control, range refusals
PASS storage bf16: capture/replay, chunking, corrupted-input control, range refusals
PASS storage f32: capture/replay, chunking, corrupted-input control, range refusals
PASS mass refusals: head grouping, window bounds, non-finite scale
PASS bounds/ranking: byte cap, overflow, type/shape refusals, ties, share zero, NaN refusal
PASS question question-e700-w64: max_abs=5.960e-08 max_rel=1.713e-07 rows=identical repeat=bitwise wrong_weight_rel=8.154e-01
PASS question question-e500-w64: max_abs=7.451e-08 max_rel=2.251e-07 rows=identical repeat=bitwise wrong_weight_rel=8.297e-01
PASS question question-e37-w37: max_abs=4.768e-07 max_rel=1.405e-07 rows=identical repeat=bitwise wrong_weight_rel=4.468e-01
PASS question question-e700-w1: max_abs=1.141e-08 max_rel=8.443e-07 rows=identical repeat=bitwise wrong_weight_rel=3.930e+00
PASS tiny oracle h_S capture f16: chunks=700/128/50 byte-identical to torch conversion; rounding max_abs=2.435e-04 max_rel=1.695e-02
PASS tiny oracle h_S capture bf16: chunks=700/128/50 byte-identical to torch conversion; rounding max_abs=1.952e-03 max_rel=3.883e-03
PASS tiny oracle h_S capture f32: chunks=700/128/50 byte-identical to torch conversion; rounding max_abs=0.000e+00 max_rel=0.000e+00
```

The wrong-weight control rotates query-head weight groups and must exceed 1e-2 relative error. The negative control repeats unchanged inputs and compares output bytes. Causal mass after the window end is checked as zero in the isolated mass tests; total mass equals query heads times window rows. Python `top_rows` compares selected rows from measured and reference mass; C++ selector tests separately exercise ties, rounding, zero share, and invalid mass.

## Fixture and precision

No L-B tiny model was available when these tests began. Under the coordinator's authorization, `check-mass.py` creates a throwaway local fixture and random split-4 projector in `build-sweep/tiny-qwen35-local/`: seed 20261020, eight layers, attention at 3/7, hidden 64, attention head dimension 16, four Q heads/two KV heads, GDN two key heads/four value heads of width 16, conv width 4, vocabulary 300. It writes safetensors directly, then runs the actual oracle `Qwen35`, `run_lower`, `window_mass`, and `top_rows` in CPU FP32. N=700, P=500; windows end at 700, 500, and 37, with window lengths 64, 37, and 1. These initial measurements are historical; the shared split-4 fixture below is now the gate input.

At width 5120 with two-byte elements, the 1 GiB cap admits at most 104857 tokens; 32768 tokens need 335544320 bytes. The cap is per buffer; graph scratch and captured query inputs are additional allocations. F32 is available internally for captured query windows and strict FP32 oracle checks. Four F32 query windows of 64 rows and width 5120 need 5242880 bytes.

Measured h_S rounding above demonstrates that two-byte storage is not bit-identical to the FP32 lower pass. Use F32 storage for the strict FP32 integration gate; separately measure the shipped F16/BF16 path instead of silently weakening the tolerance. End-to-end logits with either two-byte format remain unmeasured.

`select.choose` budgets against N, but `scripts/q27_quality.py:214` confirms the 27B caller budgets against P. Follow SPEC's prefix budget and `select.top_rows` position ties.

## Interface handoff and next step

Read L-A's `RF27-INTERFACE.md`, commit `9e1c3d076`. The proposed `llama_ridgefill_seq_set_part` and PART_LOWER/PART_LATE dispatch remain absent. The coordinator confirmed rebase readiness; the local branch is now rebased onto `rf27-engine` at `72d79baa7`, including upstream `1242be66f`.

- L-A can use `llama_ridgefill_sweep_buffer::tensor()` for the specified `h_buf`, allocated with `model.select_buft(split)`. `capture()` and `slice()` also provide per-ubatch graph copies/views, avoiding per-row host copying described in the interface note. These views bake the row offset into the graph, so a changed offset requires rebuilding or must be handled by the context copy path.
- Layer-input extraction from the existing embeddings API is the agreed query-window source. The mass helper accepts the actual window size through `llm_graph_context::n_tokens` and four positions per query row, then keys `[head_dim, N, Hkv]` in prompt order. The cache helpers now gather the slot's logical rows before calling it; the driver must fill the row-index input and the optional key-rotation input.
- L-B supplied the split-4 fixture variant and frozen oracle at `rf27-tools` commit `022983964`; it should adopt the standalone checks into its owned harness. Split 3 with attention at 3/7 contains no lower attention and cannot validate question mass.
- Next implement lower-sweep execution and slot cache mapping against L-A's landed touch points, then sweeps 2-3 and the server driver after G1 speed.

The initial missing skill path is resolved: read `/var/home/dylan/AI-Work/ridgefill-ports-20261020/skills/build-anything.md` sections 2-4. No GPU, Docker, push, or publication was attempted.


## 2026-10-20 18:06: real cache integration

Read the relayed interface through `git show rf27-engine:RF27-INTERFACE.md`. At inspection the branch tip is `9e1c3d076`, whose ancestry includes upstream `1242be66f`; this lane has not rebased because the requested L-A ready notification has not arrived. The earlier cited `fffb9069b` interface object is also readable.

Added logical-position mapping and device-side key gathering in L-C's existing new files. The mapping refuses missing/repeated positions, the wrong prompt length, invalid sequence IDs, and pending cache shifts. The graph selects the sequence's actual stream and gathers its physical rows into prompt order, including FP16-to-FP32 conversion through `ggml_get_rows`. Query reconstruction accepts the cache's Hadamard rotation and applies the same helper as stock attention before computing mass.

L-B's shared fixture is now available under its `out/gate1-qwen35/` directory. The new temporary `build-sweep/check-cache.cpp` test loads `tiny-qwen35-f32.gguf`, decodes N=700 in chunks of 50, captures the real layer-3 inputs, and reads real layer-3 keys. A second sequence with different tokens is interleaved to create physical gaps. A separate run disables unified KV and selects stream 1. Wrong-map controls reverse the physical row map; unchanged graph repetitions must produce identical float values. `build-sweep/check-cache.py` compares the graph with `select.window_mass` both on the engine's captured inputs/keys and on independent `run_lower` output from L-B's HF fixture.

```text
PASS live cache f32: graph max_abs=5.960e-08 max_rel=1.433e-07; FP32 lower oracle max_rel=3.017e-07; wrong_mapping_rel=6.636e-01
PASS live cache f16: graph max_abs=4.470e-08 max_rel=1.223e-07; FP32 lower oracle max_rel=2.319e-04; wrong_mapping_rel=6.635e-01
PASS live cache stream-f32: graph max_abs=5.960e-08 max_rel=1.590e-07; FP32 lower oracle max_rel=3.469e-07; wrong_mapping_rel=7.786e-01
PASS real llama.cpp cache rows and query window; input came from stock full decode, not LOWER dispatch
PASS key rotation: paired Q/K max_rel=1.428e-07; omitted Q rotation control max_rel=9.425e-01
```

FP32 live-cache cases meet 1e-4 versus the independent lower oracle and select identical rows. FP16 passes the graph comparison against the same stored keys; it does NOT meet 1e-4 against the FP32-key reference (2.319e-4). The key-rotation check uses paired synthetic Hadamard rotations; it is not a quantized-cache end-to-end gate. No full sweep or final-logit agreement is claimed.

Additional commands (four CPU threads maximum):

```sh
c++ -std=c++17 -O2 -I src -I include -I ggml/include build-sweep/check-cache.cpp -L build-sweep/bin -Wl,-rpath,'$ORIGIN/bin' -lllama -lggml -lggml-cpu -lggml-base -o build-sweep/check-cache
build-sweep/check-cache ../llama.cpp-ridgefill-tools/out/gate1-qwen35/tiny-qwen35-f32.gguf ../llama.cpp-ridgefill-tools/out/gate1-qwen35/tokens.bin build-sweep/cache-f32 f32
build-sweep/check-cache ../llama.cpp-ridgefill-tools/out/gate1-qwen35/tiny-qwen35-f32.gguf ../llama.cpp-ridgefill-tools/out/gate1-qwen35/tokens.bin build-sweep/cache-f16 f16
build-sweep/check-cache ../llama.cpp-ridgefill-tools/out/gate1-qwen35/tiny-qwen35-f32.gguf ../llama.cpp-ridgefill-tools/out/gate1-qwen35/tokens.bin build-sweep/cache-stream-f32 f32 separate
PYTHONDONTWRITEBYTECODE=1 python3 build-sweep/check-cache.py /var/home/dylan/projects/research/kva-wt-q27 ../llama.cpp-ridgefill-tools/out/gate1-qwen35
PYTHONDONTWRITEBYTECODE=1 python3 build-sweep/check-rotation.py
```

Logs are `build-sweep/check-cache.log`, `check-cache-f32.log`, `check-cache-f16.log`, `check-cache-stream.log`, and `check-rotation.log`. L-B can adopt `check-cache.cpp`, `check-cache.py`, and `check-rotation.py` along with the previous harnesses. The original storage and standalone mass checks were also rerun after the API change.

Exact remaining interface items for L-A/coordinator:

1. `llama_ridgefill_seq_set_part`, `LLM_RIDGEFILL_PART_LOWER/LATE`, and the qwen35 lower-stop/late-input dispatch are described but not present in the implementation at this inspection. Lower-only execution and sweeps 2-3 cannot be exercised until they land.
2. The interface's layer-input extraction description needs correction: current `llama_context::extract_layer_inputs` copies ALL rows of each enabled layer from a decode batch, not only output rows. The flag is context-wide and `llama_set_embeddings_layer_inp` requests scheduler reservation. The eventual driver must scope extraction to lower-sweep calls and restore it before LATE calls, or L-A must make extraction part-aware; a LATE graph has no layer-3 input to extract.
3. PART_LATE memory preparation must preserve lower recurrent state at N and reuse the lower sweep's physical KV cells instead of allocating duplicate prompt positions. This invariant needs an integrated test; the interface currently states graph dispatch and copies but does not spell out this memory transition.
4. Resolved by coordinator: L-C owns the guarded driver call in server prompt processing and its CMake entries; L-B owns the rest of the server. The current engine base has no request `ridgefill_mode` or common qattn fields. They exist in L-B's working tree but are not committed yet, so the guarded call cannot compile on this branch until that plumbing lands.

## 2026-10-20 18:12: shared split-4 frozen oracle gate

Switched the gate input to L-B's `out/gate1-qwen35/` fixture as relayed by the coordinator. The split-4 projector is the designated input for future integrated sweep gates; the question-mass component itself does not consume a projector. No new throwaway projector was generated or used for these checks. Read `rf27-tools:tests/ridgefill/run_gate1_qwen35.py` at `022983964`; its `quality+qattn` path explicitly returns SKIPPED until the sweep dump/mass API is available.

The new local `check-frozen.py` reads `qattn-s4-oracle.npz` without pickle, checks its tokens and total mass, and verifies the frozen mass against the local oracle's `window_mass`. It feeds the frozen layer-3 input and keys into the production question graph. It compares mass before calling the production C++ row selector through `check-rows.cpp`; all 100 rows must equal the frozen reference. Positive controls change query-head weights or reverse the live cache row map, while the unchanged-input control compares output bytes. Live cache runs use the shared GGUF and tokens, with two interleaved sequences. The capture checks use the frozen split-4 h_S rather than the earlier throwaway model.

```text
PASS frozen oracle provenance: mass_sum=255.999999155 local window_mass max_abs=0.000e+00 max_rel=0.000e+00
PASS frozen question graph: max_abs=4.470e-08 max_rel=1.763e-07 rows=100/100 repeat=bitwise wrong_weight_rel=7.076e-01
PASS frozen live cache f32: max_abs=1.341e-07 max_rel=3.017e-07 rows=100/100 wrong_mapping_rel=6.636e-01
MEASURE frozen live cache f16: max_abs=9.476e-05 max_rel=2.319e-04; FP32 1e-4 gate=FAIL
PASS frozen h_S capture f16: chunks=700/128/50 byte-identical to torch; rounding max_abs=3.040e-05 max_rel=1.163e-01
PASS frozen h_S capture bf16: chunks=700/128/50 byte-identical to torch; rounding max_abs=2.434e-04 max_rel=3.890e-03
PASS frozen h_S capture f32: chunks=700/128/50 byte-identical to torch; rounding max_abs=0.000e+00 max_rel=0.000e+00
SKIPPED integrated LOWER/LATE and server: L-A dispatch/context hooks are not present; shared split-4 projector reserved for the integrated gate
```

The component FP32 mass gate passes 1e-4. FP16 cache versus the FP32-key oracle still fails that tolerance; this result is not relabeled PASS or used to weaken the gate. F16/BF16 capture PASS means the stored bytes match PyTorch conversion, not that the rounded hidden state matches FP32 values or final logits.

Commands:

```sh
c++ -std=c++17 -O2 -I src -I ggml/include build-sweep/check-rows.cpp -L build-sweep/bin -Wl,-rpath,'$ORIGIN/bin' -lllama -lggml -lggml-cpu -lggml-base -o build-sweep/check-rows
build-sweep/check-cache ../llama.cpp-ridgefill-tools/out/gate1-qwen35/tiny-qwen35-f32.gguf ../llama.cpp-ridgefill-tools/out/gate1-qwen35/tokens.bin build-sweep/cache-f32 f32
build-sweep/check-cache ../llama.cpp-ridgefill-tools/out/gate1-qwen35/tiny-qwen35-f32.gguf ../llama.cpp-ridgefill-tools/out/gate1-qwen35/tokens.bin build-sweep/cache-f16 f16
PYTHONDONTWRITEBYTECODE=1 python3 build-sweep/check-frozen.py /var/home/dylan/projects/research/kva-wt-q27 ../llama.cpp-ridgefill-tools/out/gate1-qwen35
```

These commands exited zero; the test records the FP16 mismatch as a precision measurement and explicitly skips unavailable integration. Log: `build-sweep/check-frozen.log`. The new harness files are local and ignored, ready for L-B to adopt with the other component checks. Production sources did not change in this milestone; the existing CPU build remains green. L-A's inspected tip is `72d79baa7`; LOWER/LATE hooks and the rebase-ready notification are still pending.

Shared fixture SHA-256:

```text
tiny-qwen35-f32.gguf 7ca220ce7e310c0b8a2a24ef5b17f37ef083c6fd568cd689ce513e77c980c868
projector-s4-f32.gguf 195b590b75550b14095dd78dcc5554d05b9680294a24482b212635a7e05ea9a0
qattn-s4-oracle.npz 8a42699c91592637fc5f3d7337d68e29c642c057d4d5d07b754d86a74b706402
tokens.bin 6f050715842a38697e27ffb16cff0cc29fbda6bece55a19b914d385ef9e50dfb
```

## 2026-10-20 18:21: rebased driver skeleton and zero-share refusal controls

Rebased only L-C's three commits onto `rf27-engine` at `72d79baa7`, whose ancestry includes upstream `1242be66f`. The first plain `git rebase rf27-engine` tried to replay older engine commits and conflicted in L-A/L-B files. Aborted that attempt without resolving or changing their files, then ran `git rebase --onto rf27-engine 85b1c41fb rf27-sweep`. This succeeded without conflicts. Commit mapping: `f5bab7106` -> `f1c1173dd`, `956bf872e` -> `aa4085be5`, `41f6e1f8e` -> `4f0530a98`.

Added `tools/server/server-ridgefill-sweep.{h,cpp}` and only their two source/header entries in the server CMake file, under the coordinator's ownership decision. `server_ridgefill_sweep_step` is the named entry point, intended for a `mode == "quality+qattn"` guard. Its state owns the hidden buffer, mass, selected rows, phase and cursor. The skeleton orders lower ubatches, mass, union of class/digit/question rows, late-prefix ubatches, then exact-tail ubatches. One call can run at most one ubatch or one mass/selection stage; it never loops over the whole prompt in the server thread.

The execution methods explicitly throw named `RidgeFill quality+qattn` errors for missing LOWER dispatch, missing captured query inputs/cache rows, missing LATE fill replay and missing LATE tail logits. Nonzero share cannot report success: the first execution stage refuses without modifying the context. The row-union and later phase transitions are not reachable yet and are not claimed as tested integration. Zero share returns the `quality` routing result before the byte check, allocation or context access, leaving class/digit rows alone. A zero prefix also bypasses sweeps. All nonzero filled prompts check the existing shared buffer limit before calling the missing engine stage; the size constant is not duplicated.

The guarded server call is not installed yet. This base has neither `server_task_params::ridgefill_mode` nor `common_params::ridgefill_qattn_share/window/blind`. L-B has those fields and a temporary refusal hunk in its working tree, but `rf27-tools` at `e0e1fa9e8` does not contain them. The intended replacement is that temporary refusal in `server-context.cpp`, after `p_fill` is computed. Need L-B's committed plumbing before that owned hunk can compile. For actual interleaving, the sweep object must live with the slot across prompt updates and remain alive until pending graphs finish; a local temporary or a synchronous full-prompt loop is not a valid implementation.

CPU commands:

```sh
cmake --build build-sweep --target llama -j4
cmake -S . -B build-sweep -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_OPENSSL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-sweep --target server-context -j4
c++ -std=c++17 -O2 -Wall -Wextra -I tools/server -I include -I ggml/include build-sweep/check-driver.cpp tools/server/server-ridgefill-sweep.cpp -L build-sweep/bin -Wl,-rpath,'$ORIGIN/bin' -lllama -lggml -lggml-cpu -lggml-base -o build-sweep/check-driver
build-sweep/check-driver
```

```text
[100%] Built target llama
[100%] Built target server-context
PASS zero-share control: returns quality, no allocation/context access, class/digit rows unchanged, repeats unchanged
PASS h_S cap: 104857 rows admitted, 104858 refused by name for F16/BF16, extreme dimensions refused before allocation
PASS positive controls: nonzero share refuses missing LOWER dispatch; invalid shares/window/prefix refuse by name; no-prefix bypass
SKIPPED quality+qattn share 0 versus quality caches/logits: quality and LOWER/LATE engine dispatch unavailable
```

The local ignored `build-sweep/check-driver.cpp` is ready for L-B to adopt. It replaces C++ allocation functions to count zero-share allocations, passes a null context to detect accidental engine access, preserves class/digit row values, repeats the neutral case, and requires positive-control refusals by mode and reason. The boundary tests admit exactly 1073735680 bytes at 104857 rows of width 5120/F16, then require the next row to fail with `maximum 104857 tokens`. A skipped numerical quality comparison is printed explicitly, never counted as a passing check.

Recompiled `check-sweep`, `check-question`, `check-cache`, and `check-rows` against the rebased internal headers, then reran the storage checks, live FP32/FP16 cache runs, and `check-frozen.py`. Results are unchanged: question graph max relative error 1.763e-7, live FP32 cache 3.017e-7, selected rows 100/100, and the FP16-vs-FP32 1e-4 gate still fails at 2.319e-4. Logs: `build-rebased.log`, `configure-server.log`, `build-server.log`, `check-driver.log`, and `check-frozen-rebased.log` under `build-sweep/`.

Self-review checked scope, dimensions before allocation, lifetime, ASCII comments, and the server's one-ubatch scheduling requirement. `git diff --check` passed. Rejected: copying L-B's uncommitted plumbing, bypassing missing engine stages with stock/speed execution, allocating the sweep buffer for zero share, and blocking the server loop for a complete prompt. Remaining numerical and server integration checks stay SKIPPED until L-A's hooks and L-B's committed fields are available. The coordinator has relayed the part-aware extraction and lower-state/KV-cell preservation requirements to L-A.

## 2026-10-20 18:25: dispatch proposal scope and FP16 row-set gate

The coordinator authorized L-C to propose the part dispatch, API, extraction and replay changes in one separate commit for L-A review after speed G1. Rebased onto the current `rf27-engine` tip `f1b92645f`; no conflicts. Current prior commits are `bce3791cd`, `86e2e9502`, `c57cf886d`, and `976a2298a` (the accepted driver skeleton was previously `70a55d624`).

Completed the requested FP16 selected-row check with `build-sweep/check-f16-rows.py`. It uses the shared tiny GGUF token text/types, the split-4 `rowsel.safetensors`, and the real engine's FP32/FP16 mass dumps. The C++ selector chooses 100 question rows; oracle `top_rows` independently agrees. Class selection keeps 127 of 254 finite-score prefix rows at share 0.5 with position ties; digit selection returns 20 prefix rows. The union is checked after question rows, so class/digit overlap cannot hide a changed question set.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 build-sweep/check-f16-rows.py /var/home/dylan/projects/research/kva-wt-q27 ../llama.cpp-ridgefill-tools/out/gate1-qwen35
```

```text
PASS f32 controls: repeated rows identical; wrong-map union symmetric difference=123
PASS f16 controls: repeated rows identical; wrong-map union symmetric difference=123
PASS FP16/F32 question rows: f32=100 f16=100 symmetric_difference=0
PASS FP16/F32 exact-row union: f32=217 f16=217 symmetric_difference=0 class=127 digits=20
```

Per the coordinator, FP16 mass versus the FP32 oracle (2.319e-4) is informational. The required FP16 row-set check above passes. G1 remains a FP32 gate; no tolerance was weakened. Log: `build-sweep/check-f16-rows.log`. This does not claim that the quality engine or a two-byte h_S replay was executed.

Dispatch inspection found one additional memory API needed outside the files named in the ownership/interface note. `llama_kv_cache_context::apply()` calls `llama_kv_cache::apply_ubatch()`. The latter removes occupied destination cells and purges positions up to the overwritten maximum (currently lines 1138-1198). Calling it with the existing lower-sweep cells loses prompt positions. Skipping `apply()` also does not work: `n_kv` is private and is not initialized by the batch-processing constructor, but the graph needs its valid value.

Prepared, but did NOT apply, `build-sweep/proposed-kv-replay.patch`: a narrow `llama_kv_cache_context::apply_replay()` method in `src/llama-kv-cache.{h,cpp}` that refreshes `n_kv` from the already-mapped slot info without allocating/removing cells. It returns false for a non-successful or empty processing context. `git apply --check` passes. A sweep-specific hybrid context in L-C's own files can call this method and configure the existing recurrent cell for graph inputs without `find_slot()` advancing its position. The first LATE graph must zero only late-layer recurrent/conv state via its existing graph state inputs; lower-layer state remains at the end of the prompt.

Blocked on scope confirmation for those two extra KV files: the latest task says to touch L-A's files minimally and "only where the interface says", and the interface/ownership list does not name them. Rejected const-casting/private-field access, writing into occupied cells through normal `apply_ubatch`, and adding a duplicate cache implementation. No production dispatch files have been modified yet, and the requested single proposal commit has not been created. Lower h_S and all-rows late-replay gates are SKIPPED until this memory transition can be implemented.

## 2026-10-20 18:47: proposed LOWER/LATE dispatch and existing-cell replay

The coordinator approved the narrow `src/llama-kv-cache.{h,cpp}` addition and requested one proposal commit for L-A to review. Implemented the part setter, per-sequence cursors/buffer, homogeneous part codes, lower graph stop/capture, late graph entry, part-aware layer-input extraction, and replay memory context. Rebased the four accepted L-C commits onto `rf27-engine` `3ac42957d` after its rollback and masked-quality changes landed. Resolved the context/graph parameter overlaps by retaining L-A's `rows_seq`, row counting, alpha/rho and non-unified parallel refusals. The accepted driver skeleton is now `6c5cc6a5b`.

`apply_replay()` only refreshes `n_kv` from existing mapped cells. Normal `apply_ubatch()` cannot be used: it removes occupied destination cells and purges existing positions up to the overwritten maximum. The replay context maps logical positions to those cells, refuses shared/missing cells, preserves recurrent metadata at the full prompt end, and uses existing graph state inputs to zero only the late layers on the first replay ubatch. It calls the mutable attention context owned by its base through the existing const graph-facing getter; no private cache member is accessed. A per-sequence replay cursor supplies the batch validator's next position while the stored memory position remains N-1. Normal batches retain normal validation and application.

LOWER computes [0, S), writes h_S to its position-indexed device buffer with an in-graph copy, and has no output head. LATE starts at S and uses L-A's fill builder for BULK/MASKED prefix ubatches, then stock late layers for the tail. When every row in a LATE ubatch is exact, the stock late-layer path avoids a redundant gather/scatter; it has no prefix output head. The global R=all-prefix case has rho=0, so alpha=1 is neutral. Partial exact rows with nonzero alpha still refuse by name until L-A supplies rho correction. The buffer pointer and slice position are in the graph reuse key; reserving graphs uses FULL, avoiding an out-of-bounds dummy slice.

The part setter refuses unsupported LoRA/control-vector, embedding, non-causal, MTP-context and recurrent rollback configurations. LOWER requires an empty sequence; LATE requires the complete LOWER prompt in the same buffer; FULL requires completed replay or an explicitly cleared sequence. Aborted part execution clears the sequence because its recurrent state cannot be rolled back. Missing layer inputs are skipped and return null when absent throughout the batch, with zero rows for absent ubatches in a mixed batch. The availability vector is allocated only when layer-input extraction is enabled. A loaded projector with no filled/part rows leaves the marker vector empty without allocating it.

Interface clarification for L-A: the current upstream layer-input extraction copies all token rows, independent of logits flags. LOWER therefore refuses requested logits and still exposes the requested lower-layer inputs (including the input to S). No output flags are needed for the question window. The direct in-graph h_buf view/copy implements the row-position contract without a host h_S transfer; only CPU was tested.

The local ignored harnesses are `build-sweep/check-dispatch.cpp`, `check-dispatch-prefix.py`, and `check-dispatch-refusal.cpp`, ready for L-B to adopt. N=700, P=500, split=4, F32 model/KV/h_S, CPU four threads, FA disabled, shared L-B fixture. Stock and replay use the same chunk boundaries. The prefix parser independently checks each KV, conv and recurrent array using L-B's `read_state`, rather than accepting a nonempty dump as a pass. The two all-row runs use alpha 0/1; the positive control corrupts h_S[0] before replay. A six-row R={0,7,31,127,255,499} replay also matches a single-pass quality run byte-for-byte, including resumed decode.

Commands (all exit zero):

```sh
cmake --build build-sweep --target server-context -j4
c++ -std=c++17 -O2 -I src -I include -I ggml/include build-sweep/check-dispatch.cpp -L build-sweep/bin -Wl,-rpath,'$ORIGIN/bin' -lllama -lggml -lggml-cpu -lggml-base -o build-sweep/check-dispatch
c++ -std=c++17 -O2 -I src -I include -I ggml/include build-sweep/check-dispatch-refusal.cpp -L build-sweep/bin -Wl,-rpath,'$ORIGIN/bin' -lllama -lggml -lggml-cpu -lggml-base -o build-sweep/check-dispatch-refusal
build-sweep/check-dispatch ../llama.cpp-ridgefill-tools/out/gate1-qwen35/tiny-qwen35-f32.gguf ../llama.cpp-ridgefill-tools/out/gate1-qwen35/projector-s4-f32.gguf ../llama.cpp-ridgefill-tools/out/gate1-qwen35/tokens.bin build-sweep/dispatch-50 50 50
build-sweep/check-dispatch ../llama.cpp-ridgefill-tools/out/gate1-qwen35/tiny-qwen35-f32.gguf ../llama.cpp-ridgefill-tools/out/gate1-qwen35/projector-s4-f32.gguf ../llama.cpp-ridgefill-tools/out/gate1-qwen35/tokens.bin build-sweep/dispatch-128 128 128
build-sweep/check-dispatch ../llama.cpp-ridgefill-tools/out/gate1-qwen35/tiny-qwen35-f32.gguf ../llama.cpp-ridgefill-tools/out/gate1-qwen35/projector-s4-f32.gguf ../llama.cpp-ridgefill-tools/out/gate1-qwen35/tokens.bin build-sweep/dispatch-700 700 512
PYTHONDONTWRITEBYTECODE=1 python3 build-sweep/check-dispatch-prefix.py ../llama.cpp-ridgefill-tools build-sweep/dispatch-50 build-sweep/dispatch-128 build-sweep/dispatch-700
build-sweep/check-dispatch-refusal ../llama.cpp-ridgefill-tools/out/gate1-qwen35/tiny-qwen35-f32.gguf ../llama.cpp-ridgefill-tools/out/gate1-qwen35/projector-s4-f32.gguf
PYTHONDONTWRITEBYTECODE=1 python3 build-sweep/check-f16-rows.py /var/home/dylan/projects/research/kva-wt-q27 ../llama.cpp-ridgefill-tools/out/gate1-qwen35
build-sweep/check-driver
git diff --cached --check
```

```text
[100%] Built target llama
[100%] Built target server-context
PASS dispatch b50/ub50: h_S, final KV/conv/state, tail logits, resumed decode byte-identical; six-row LATE equals quality; loaded/no-fill equals off; all-exact alpha=1 unchanged; corrupted h_S changes caches
PASS dispatch b128/ub128: h_S, final KV/conv/state, tail logits, resumed decode byte-identical; six-row LATE equals quality; loaded/no-fill equals off; all-exact alpha=1 unchanged; corrupted h_S changes caches
PASS dispatch b700/ub512: h_S, final KV/conv/state, tail logits, resumed decode byte-identical; six-row LATE equals quality; loaded/no-fill equals off; all-exact alpha=1 unchanged; corrupted h_S changes caches
PASS build-sweep/dispatch-50: prefix late KV/conv/state 8/8 arrays byte-identical to stock; lower 8/8 unchanged through prefix and tail; corrupt-h_S changes 2 late arrays
PASS build-sweep/dispatch-128: prefix late KV/conv/state 8/8 arrays byte-identical to stock; lower 8/8 unchanged through prefix and tail; corrupt-h_S changes 2 late arrays
PASS build-sweep/dispatch-700: prefix late KV/conv/state 8/8 arrays byte-identical to stock; lower 8/8 unchanged through prefix and tail; corrupt-h_S changes 2 late arrays
PASS refusals: missing projector/buffer, invalid sequence, oversized prefix, LOWER logits/nonconsecutive rows, partial exact rows with alpha=1 preserve cache; LATE abort clears sequence; n_rs_seq=4 refused
PASS FP16/F32 question rows: f32=100 f16=100 symmetric_difference=0
PASS FP16/F32 exact-row union: f32=217 f16=217 symmetric_difference=0 class=127 digits=20
```

Logs: `build-sweep/build-dispatch-rebased.log`, `check-dispatch-{50,128,700}.log`, `check-dispatch-prefix.log`, `check-dispatch-refusal.log`, `check-f16-rows.log`. The selected-row check still includes repeated-row and wrong-key-order controls (wrong-map union symmetric difference 123). FP16 mass drift remains informational, as requested; the byte-equality replay gates use F32 h_S/KV. The driver zero-share routing/no-allocation and 1 GiB boundary checks were rerun and pass.

Self-review covered scope, buffer lifetime/index bounds, graph reuse, skipped layer inputs, pure part ubatches, named refusals, and stock/off behavior. Initial probes caught a missing LOWER loop hook and then M-RoPE's stock past-position rejection; both were fixed before the passing runs. Rejected normal cache insertion during replay, clearing/re-running lower state, weakening byte equality, and silently accepting partial-quality alpha correction.

Still SKIPPED: integrated server sweeps and the numerical server qattn-share=0 versus quality gate; driver functions remain explicit stubs pending integration with committed mode plumbing. This proposal does not claim the full question-selected G1 against `finish(exact_rows=R)`, multi-slot replay, two-byte h_S replay, or GPU validation. Next: L-A reviews/merges the proposal; L-C wires the existing server driver and runs the question-selected oracle gate. No push or external writes.

Before handoff, rebased once more onto `rf27-engine` `d3b20e862`. That additional engine commit only changes its G0 harness/logs and NOTES; the tested production source content is unchanged. The rebase completed without conflicts.

## 2026-10-20 18:58: executable sweep driver and split-4 question-selected G1

The dispatch proposal is `8de161386`. Imported L-B's committed plumbing through `f57291696` by cherry-picking its nine commits after `bd9ede799` (local import tip `0bfb45a68`). No L-B source was edited beyond those verbatim commits. L-A's current tip remains `d3b20e862`; the new driver is tested on the gated proposal dispatch while L-A prepares its implementation.

Implemented the previously named driver stages in `tools/server/server-ridgefill-sweep.{h,cpp}`. Each step runs at most one ubatch, or one mass/selection stage. LOWER stores h_S and captures only the question window at the oracle's lower attention layers in {3,7,11,15}. The runtime question graph in `src/llama-ridgefill-sweep.{h,cpp}` uses the context's existing backends in a separate scheduler, gathers logical cache rows, and sums its FP32 group results in FP64. It does not reset the context's scheduler or transfer h_S through the host. The driver unions supplied class/digit rows with its question rows, sets R, replays the prefix, then runs the exact tail and restores FULL. Query extraction flags are restored after capture.

The driver supports a deferred final token so the server can put that token in its normal output/sampling batch. The caller must then call `finish_external_tail`; an interrupted slot must call `cancel` before freeing its buffer. The deferred-token variant is independently run through the same numerical gate. HTTP wiring is not yet applied.

Native G1 uses the shared split-4 fixture, N=700/P=500, F32 model/KV/h_S, FA off, four CPU threads, alpha=0. The Python oracle independently selects 127 class rows plus 20 digit rows (147 distinct base rows), then 100 question rows, making a union of 217. Those 147 base rows are supplied to the driver because L-A's class/digit selector API is still missing; the driver computes mass and the question union itself. Mass is compared first to the shared `qattn-s4-oracle.npz`; exact R is checked next; then 25 arrays (late prefix KV/conv/recurrent states, all final caches, and 200 tail logits) are compared against a fresh `finish(exact_rows=R)` reference. Prefix snapshots retain N lower-cache cells, so the parser compares the late KV prefix slices [0,P), not the unused future late rows.

```text
PASS qattn b50/ub50:   mass relative error 2.659567080398772e-7; rows 217/217; 25 arrays; worst relative error 6.630082225456135e-7
PASS qattn b128/ub128: mass relative error 2.659567080398772e-7; rows 217/217; 25 arrays; worst relative error 6.630082225456135e-7
PASS qattn b700/ub512: mass relative error 2.659567080398772e-7; rows 217/217; 25 arrays; worst relative error 6.630082225456135e-7
PASS deferred-final-token variant: same bound at all three chunk sizes
PASS repeated qattn: prefix/final state and tail logits byte-identical at all three chunk sizes
PASS qattn-share=0 versus quality: prefix/final state and tail logits byte-identical at all three chunk sizes
PASS wrong-projector positive control: mass/R unchanged; worst cache/state error 1.337659239768982 at all three chunk sizes
PASS zero-share routing: no allocation/context access; base rows unchanged
PASS 1 GiB bound: width 5120/F16 admits 104857 rows and refuses 104858 before allocation
[100%] Built target server-context
```

Correct comparisons retain the 1e-4 FP32 threshold. The wrong-projector control must exceed 1e-2; its pass means it was detected, not that it agrees. The selected-row equality is exact. No F16 h_S replay or GPU numerical claim is made here; the earlier FP16-cache row-selection gate remains recorded above.

Commands:

```sh
cmake --build build-sweep --target server-context -j4
c++ -std=c++17 -O2 -I tools/server -I src -I include -I ggml/include build-sweep/g1-sweep.cpp tools/server/server-ridgefill-sweep.cpp -L build-sweep/bin -Wl,-rpath,'$ORIGIN/bin' -lllama -lggml -lggml-cpu -lggml-base -o build-sweep/g1-sweep
PYTHONDONTWRITEBYTECODE=1 python3 build-sweep/g1-sweep-oracle.py /var/home/dylan/projects/research/kva-wt-q27 ../llama.cpp-ridgefill-tools/out/gate1-qwen35
c++ -std=c++17 -O2 -Wall -Wextra -I tools/server -I include -I ggml/include build-sweep/check-driver.cpp tools/server/server-ridgefill-sweep.cpp -L build-sweep/bin -Wl,-rpath,'$ORIGIN/bin' -lllama -lggml -lggml-cpu -lggml-base -o build-sweep/check-driver
build-sweep/check-driver
git diff --check
```

Evidence: `build-sweep/g1-sweep-results.json`, `g1-sweep-oracle.log`, `g1-{50,128,700}-{qattn,defer,repeat,zero,quality,wrong}.{log,prefix.kv,kv,logits}`, `check-driver-current.log`, and `build-driver-final.log`. The new harness sources are local/ignored and ready for L-B to adopt. A stale aggregate initializer in the small routing harness failed compilation after adding `split`; updated it, recompiled, and reran successfully. The stale binary's intermediate output was not accepted as a new pass.

Server integration has two remaining dependencies. First, `common_ridgefill_rows()` in the imported plumbing still unconditionally refuses eligible quality requests because `llama_ridgefill_select_rows` is not declared/implemented in the current engine. L-A must provide the selector and L-B must restore the common helper. No class/digit selector was duplicated in this lane. Second, the earlier ownership grant covers the single guarded driver call, but the asynchronous server also needs slot-owned state, cancellation before prompt clearing/reset, completion after the deferred token, and suppression of cache truncation/checkpoints during replay. Prepared `build-sweep/proposed-server-lifecycle.patch` for those narrow extra hunks; `git apply --check` passes, but it is NOT applied, built or tested. It contains no sweep algorithm, only ownership/lifetime plumbing.

SKIPPED: HTTP/server integration, live request cancellation/concurrency, and server-level zero-share equality. The native driver and its deferred-token contract pass; the server's existing named refusals remain until the dependency and ownership items above are resolved. Rejected a blocking full-prompt server loop, a temporary local h_S owner, a duplicate class selector, and resetting the main context scheduler to compute mass.

## 2026-10-20 19:08: approved server lifecycle plumbing

Applied the coordinator-approved lifecycle hunks in `tools/server/server-context.cpp`: the slot owns the sweep and its parameters; prompt clearing/reset cancels before freeing storage; release cancels before callbacks; model unload cancels while the context is alive; active sweeps cannot enter the prompt cache, truncate cached positions, or create checkpoints; the normal final-token post-decode path finishes FULL before releasing h_S. This is a separate lifecycle commit. The guarded request-to-driver call is still pending.

`cmake --build build-sweep --target server-context -j4` and `git diff --check` pass. A local harness includes the actual server slot implementation and exercises prompt_clear, reset, release/callback ordering, and deferred completion. Its first cancellation passes, but restarting LOWER in the same context crashes in graph execution on the old proposal dispatch: `PASS slot lifecycle stage 0` followed by a crash at stage 1 step 1. No HTTP or completed lifecycle gate is claimed. The archive-linked native G1 control still passes. Logs: `build-sweep/build-lifecycle.log`, `check-slot-lifecycle-proposal.log`, `debug-slot-lifecycle.log`; harness: `build-sweep/check-slot-lifecycle.cpp` (ignored, for L-B to adopt).

L-A has now committed its replacement dispatch (`78d4a2551`, new capture-buffer API and graph-generation key) and selector (`ff598e03d`). Next, rebase without the superseded proposal implementation, adapt the driver to the committed window API, then rerun this cancellation/reuse probe and the question-selected G1. This prevents carrying a second dispatch implementation beside L-A's. Original proposal `8de161386` remains the review reference. HTTP remains SKIPPED until the common selector helper and guarded server hook are present. No push or external writes.

## 2026-10-20 19:15: engine dispatch adopted; cancellation/reuse gate passes

Rebased onto `rf27-engine` `ff598e03d` (dispatch `78d4a2551` and class/digit selector). Dropped the superseded implementation from proposal `8de161386`, retaining its evidence above; no L-A production files are changed by this lane. The lifecycle commit is now `643903107` (previously `db2a6cabf`). Rebase conflicts were limited to this lane's sweep files and notes; removed the old replay subclass/capture hooks and retained the runtime mass graph.

Adapted the driver to `llama_ridgefill_part_params`: per-slot F32 device window buffers capture the lower layer inputs in the graph. The question graph reads those buffers directly, without host query-window transfers or context-wide extraction flags. FULL completion explicitly synchronizes before releasing h_S/windows. The buffers remain alive through replay/cancellation.

```text
[100%] Built target server-context
PASS qattn and deferred-final-token G1 at b50/ub50, b128/ub128, b700/ub512:
  mass relative error 2.659567080398772e-7; selected rows 217/217;
  25 arrays; worst relative error 6.630082225456135e-7
PASS wrong-projector control: mass/R unchanged; worst error 1.337659239768982
PASS repeated qattn and qattn-share=0 versus quality: byte-identical at all three chunk sizes
PASS slot lifecycle stage 0: live sequence cancelled, buffer freed, prompt cache excluded
PASS slot lifecycle stage 1: live sequence cancelled, buffer freed, prompt cache excluded
PASS slot lifecycle stage 2: live sequence cancelled, buffer freed, prompt cache excluded
PASS cancelled-slot reuse: deferred completion caches byte-identical to G1; completed prompt cacheable
```

The slot probe includes the real `server-context.cpp` and uses one context across all four runs. Stage 0 calls prompt_clear after one LOWER ubatch; stage 1 calls reset after prefix replay; stage 2 calls release at deferred-final-token readiness and checks both callbacks see cleanup in the right order; stage 3 completes the deferred token and compares serialized caches byte-for-byte to the separate G1 run. The earlier restart crash occurred only with the now-dropped proposal; L-A's committed dispatch passes the same probe without a workaround.

Commands: CPU `cmake --build build-sweep --target server-context -j4`; recompile/run `g1-sweep.cpp` and `g1-sweep-oracle.py` as above; compile `check-slot-lifecycle.cpp` with the server-context archive, common and mtmd dependencies; execute on the shared split-4 fixture. Logs: `build-sweep/build-engine-window.log`, `g1-sweep-engine-oracle.log`, `check-slot-lifecycle.log`, `build-slot-lifecycle.log`. `git diff --check` passes. HTTP remains SKIPPED: the common helper still refuses even though the libllama selector is now present. The guarded server hook is the next independent step.

## 2026-10-20 19:18: guarded server hook built

Wired `quality+qattn` with positive share and an eligible prefix into the slot-owned driver. Each server update runs one step through `ridgefill_sweep_prompt`, using the existing queue yield mechanism; pending work leaves the slot processing. The ready state appends the already-computed prompt metadata through N-2, then the normal batch decodes N-1 and the lifecycle hook restores FULL. Active sweeps cannot share that final batch with another sequence, as required by the engine's replay contract. Prompt metrics count each input once despite the replay. Question-dependent requests restart their prompt rather than reuse stale question-selected rows. Shared/child prompts, speculative decoding, non-completion tasks and LoRA refuse by name on this path; zero share takes the existing quality route.

`cmake --build build-sweep --target llama-server -j4` and `git diff --check` pass (`build-sweep/build-server-sweep.log`). This commit records wiring/build only, not an HTTP pass. L-B's helper has just landed as `be962e03a`; next import its intervening committed changes and run the real HTTP checks. No common helper or class selector was reimplemented here.

## 2026-10-20: HTTP end-to-end and active lifecycle gates pass

Imported L-B's five committed changes after its prior handoff, through helper `be962e03a` (local import `eac6ccf9b`). The only conflict was the adjacent startup-refusal block in server-context: retained L-B's removal of silent cache-reuse disabling and this lane's removal of the now-implemented qattn-driver refusal. Other imported files are L-B's verbatim changes. The guarded hook is `53aa3a4ca`; the lifecycle commit remains separately reviewable as `643903107`, and the engine-window adaptation is `47dcad484`. L-A's source is unchanged relative to engine `ff598e03d`; the superseded proposal is no longer in the implementation stack.

HTTP uses the shared tiny split-4 model/projector, N=700/P=500, F32 weights/KV, default F16 h_S, FA off and four CPU threads. The independent native executable uses the same F16 h_S for its HTTP reference; FP32 acceptance remains the separately passing 25-array G1 against `finish(exact_rows=R)`. Seventeen HTTP comparisons cover positive qattn, ordinary quality, zero qattn share, repeated cached requests, two concurrent slots with unified KV, and disconnect/reuse. Each sampled token's pre-sampling log probability is compared to the native 300-token-vocabulary logits; worst absolute difference is 2.066796929156567e-7 (limit 1e-5). Positive qattn logs 217 exact prefix rows and counts the 700 prompt inputs once. Zero-share quality and qattn responses have byte-identical serialized probability fields. Positive question-selected requests deliberately recompute the prompt on cache reuse.

```text
PASS HTTP share=0.2 parallel=1: native logits, filled/processed counts, repeat, cancellation/reuse
PASS HTTP share=0.0 parallel=1: native logits, filled/processed counts, repeat, cancellation/reuse, zero == quality
PASS HTTP share=0.2 parallel=2: native logits, filled/processed counts, repeat, cancellation/reuse
PASS active HTTP cancellation: 3500-token sweep released with zero finalized prompt rows; next request matches native logits
PASS active shutdown with client stream still open: terminal response and clean exit
PASS active HTTP shutdown: incomplete sweep buffer released while context is alive; clean exit
[100%] Built target llama-server
```

The concurrent requests are assigned slot 0 and slot 1; the server log confirms both start before either finishes. A non-unified two-slot probe correctly hit the existing engine refusal requiring `-kvu`; the positive parallel gate uses `-kvu`. The first zero-share cache-repeat assertion incorrectly expected 500 newly filled rows; corrected the harness to require zero newly filled rows on that ordinary quality cache hit, while still checking its native logits. No implementation threshold was loosened.

The stronger cancellation probe found two missing lifecycle notifications, now fixed in server-context. Pending sweep stages bypass normal post_decode, so they now send the existing progress notification when requested. A 3500-token prompt with batch/ubatch 8 can then be disconnected and released before any prompt row is finalized, followed by a numerically correct reused slot. A second probe kept its client stream open during SIGTERM: shutdown initially waited over 10 seconds for the incomplete request (a mode-off control did not reproduce it). After the server processing loop stops, active sweep requests now receive a terminal shutdown error and release their buffers before HTTP shutdown waits for them. The same open-stream probe now exits cleanly; no client close is needed to unblock it. The earlier failure is preserved in `check-http-cancel-open-stream.log`.

Commands:

```sh
cmake --build build-sweep --target llama-server -j4
PYTHONDONTWRITEBYTECODE=1 python3 build-sweep/check-http-sweep.py ../llama.cpp-ridgefill-tools/out/gate1-qwen35
PYTHONDONTWRITEBYTECODE=1 python3 build-sweep/check-http-cancel.py ../llama.cpp-ridgefill-tools/out/gate1-qwen35
PYTHONDONTWRITEBYTECODE=1 python3 build-sweep/check-http-shutdown-recheck.py ../llama.cpp-ridgefill-tools/out/gate1-qwen35
git diff --check
git merge-base --is-ancestor rf27-engine HEAD
```

Evidence: `build-sweep/http-sweep-results.json`, `check-http-sweep.log`, `http-sweep-{0.2,0.0}-{1,2}.log`, `check-http-cancel.log`, `http-cancel-active.log`, `check-http-shutdown-recheck.log`, `http-shutdown-open-recheck.log`, `build-server-helper.log`, `build-server-progress.log`, and `build-server-shutdown.log`. Native and HTTP harnesses remain ignored local files for L-B to adopt; no tests/ridgefill files were authored by this lane. Source review checked batch isolation, per-slot buffer ownership, cleanup before callbacks/unload, no stale question-row cache reuse, one-ubatch scheduling, and single counting of prompt metrics. All required CPU/tiny checks described here pass. GPU, full 27B NLL/TTFT and release portability are outside this CPU lane and are not claimed. L-B's imported RIDGEFILL.md still labels server sweep integration pending; it needs its status updated from this evidence. No push or external writes.

## 2026-10-20: native G1 now executes the libllama class/digit selector

Addressed the coordinator's dispatch/selector relay. All four production port requirements were already present in `47dcad484` plus the integrated server stack through `6b37c11ab`: parameter-struct seq_set_part with nullptr for FULL, caller-owned device window captures, one sequence/ubatch per replay decode, and `common_ridgefill_rows(..., effective_request_mode)` from L-B's `be962e03a` feeding the driver. Verified engine `ff598e03d` is an ancestor, proposal `8de161386` is absent, and L-A's context, model, RidgeFill and replay-memory source files have no differences from that engine revision.

The remaining difference was the local native harness: it still consumed the oracle's class/digit rows as input. Updated `build-sweep/g1-sweep.cpp` to call `llama_ridgefill_select_rows(rf, ids, 500, 0.5f, true, out)` itself. Its CLI no longer takes a base-row input file; it writes `.base.rows` for independent verification. Updated `g1-sweep-oracle.py` and `check-http-sweep.py` accordingly. The Python oracle still independently computes expected class/digit/question rows; those values are never passed into the native execution. On positive-qattn runs, compare mass first, then the engine's 147 class/digit rows, the 217-row union, and finally caches/states/logits. The wrong-projector and zero-share arms also call the real selector.

Fresh CPU rerun on L-A's dispatch, split 4, N=700/P=500, FP32 model/KV/h_S, alpha=0, FA off:

```text
PASS b50/ub50:   base rows 147/147; union 217/217; mass 2.65956708e-7; 25 arrays; worst 6.63008223e-7
PASS b128/ub128: base rows 147/147; union 217/217; mass 2.65956708e-7; 25 arrays; worst 6.63008223e-7
PASS b700/ub512: base rows 147/147; union 217/217; mass 2.65956708e-7; 25 arrays; worst 6.63008223e-7
PASS deferred final token and repeated execution at all three chunk sizes
PASS wrong-projector control at all three chunk sizes: mass/R unchanged; worst error 1.337659239768982
PASS qattn-share=0 versus quality: prefix/final caches and 200 tail logits byte-identical at all three chunk sizes
PASS HTTP: 17 comparisons; worst logprob difference 2.066796929156567e-7
PASS active HTTP cancellation/reuse and open-stream shutdown
[100%] Built target llama-server
```

HTTP uses the real common helper for both quality modes and compares against the newly rebuilt native harness; its default F16 h_S reference and F32 acceptance distinction remain as documented above. The HTTP runs cover one slot, two concurrent unified-KV slots, repeated requests, and zero-share equality. L-T's real-vocabulary digit-set result (10 ids equal to kva) is accepted as relayed evidence, not a measurement performed in this CPU/tiny lane.

Commands: `cmake --build build-sweep --target llama-server -j4`; compile `build-sweep/g1-sweep.cpp` with `tools/server/server-ridgefill-sweep.cpp` against the current libllama/ggml; run `g1-sweep-oracle.py`, `check-http-sweep.py`, `check-http-cancel.py`, and `check-http-shutdown-recheck.py` with PYTHONDONTWRITEBYTECODE=1 and the shared fixture. The native CLI is now `g1-sweep MODEL PROJECTOR TOKENS OUTPUT_PREFIX BATCH UBATCH MODE f32|f16`. Logs: `build-selector-g1.log`, `g1-sweep-engine-selector.log`, `check-http-selector.log`, `check-http-selector-cancel.log`, and `check-http-selector-shutdown.log` in build-sweep. Updated native/HTTP harness sources remain local/ignored for L-B to adopt. No further production changes were necessary; no GPU/full-27B claims or pushes.


## 2026-10-20: shutdown under CPU load

Rebased rf27-sweep onto the merged integration head dbc6afea6. The reported intermittent `check-http-cancel` stall reproduced on iteration 11 with four busy Python processes and the server pinned to CPUs 0-3 (server -t 4 -tb 4). The preceding ten runs passed. The failing run passed active cancellation and native-logit reuse, then exceeded the unchanged 10-second shutdown timeout.

GDB attached to the stalled process before cleanup. The main thread was interrupted while `yield_to_queue()` held `mutex_tasks` and called `condition_tasks.notify_all()`. SIGTERM called the shutdown callback synchronously, and `server_queue::terminate()` tried to acquire that same mutex. The queue worker and the HTTP cancellation callback were also blocked on it. This is a signal-handler reentry deadlock, before the slot cleanup or HTTP join can run.

```text
#0  __lll_lock_wait
#1  pthread_mutex_lock
#2  server_queue::terminate()
#3  signal_handler(int)
#4  <signal handler called>
#5  pthread_cond_broadcast
#6  std::condition_variable::notify_all()
#7  server_queue::yield_to_queue(...)
#8  server_context_impl::ridgefill_sweep_prompt(...)
```

The fix in `tools/server/server.cpp` limits the signal handler to lock-free atomic operations and `_Exit(1)` on a second interrupt. A dispatcher thread polls the pending signal every 100 ms and invokes the existing shutdown callback in normal thread context. Its destructor joins before the captured server objects are destroyed. Embedded CLI termination keeps its existing direct callback. No queue locking or sweep ownership rules change.

Rejected options: increasing the timeout cannot resolve recursive acquisition of the mutex; removing the mutex in `terminate()` would break the condition-variable synchronization and still call non-signal-safe notification code from the handler. The polling dispatcher avoids adding platform-specific pipes/events or changing signal masks on threads already started by the HTTP server/backends. The tradeoff is one small thread and up to one polling interval of dispatch latency, plus scheduler delay. The second-interrupt path no longer uses stdio/normal exit, which can themselves take locks.

The local deterministic regression includes the actual server translation unit, holds a mutex required by the shutdown callback, and raises SIGTERM on the owning thread. It requires the signal handler to return before releasing the mutex and then requires the callback to receive SIGTERM. The patched build passes; the same source compiled against dbc6afea6's server.cpp times out after two seconds. This is a deliberately failing negative control, not a skipped check.


Validation on the patched CPU build:

```text
[100%] Built target llama-server
PASS after: 20/20 check-http-cancel runs under parallel CPU load
PASS HTTP share=0.2 parallel=1: native logits, filled/processed counts, repeat, cancellation/reuse
PASS HTTP share=0.0 parallel=1: native logits, filled/processed counts, repeat, cancellation/reuse, zero == quality
PASS HTTP share=0.2 parallel=2: native logits, filled/processed counts, repeat, cancellation/reuse
PASS 17 HTTP comparisons; worst error 2.066796929156567e-7
PASS active shutdown with client stream still open: terminal response and clean exit
PASS SIGTERM and SIGINT: signal returns while callback mutex is held; shutdown runs after unlock
PASS second signal exits immediately with code 1
PASS model SIGINT/SIGTERM and router SIGINT/SIGTERM: all four exit cleanly
```

All 20 load runs used the original `check-http-cancel.py` and its 10-second shutdown timeout. Each run cancels an active 3500-token sweep, verifies a subsequent 700-token request against native logits, and shuts down another incomplete 3500-token sweep. No retry/timeout increase/kill fallback counted as a pass. Total elapsed time for the 20 checks was 145.09 s (per-check range 3.58-17.24 s, including startup, cancellation, inference and shutdown). The four idle model/router signal checks each exited in 0.113 s. These are lifecycle timings, not prefill performance measurements.

Commands (fixture path abbreviated here as FIXTURE = ../llama.cpp-ridgefill-tools/out/gate1-qwen35):

```sh
git rebase dbc6afea6
cmake --build build-sweep --target llama-server -j4
python3 build-sweep/run-cancel-load.py before check-http-cancel-probe.py 20 FIXTURE
python3 build-sweep/run-cancel-load.py after check-http-cancel.py 20 FIXTURE
PYTHONDONTWRITEBYTECODE=1 python3 build-sweep/check-http-sweep.py FIXTURE
PYTHONDONTWRITEBYTECODE=1 python3 build-sweep/check-http-shutdown-recheck.py FIXTURE
PYTHONDONTWRITEBYTECODE=1 python3 build-sweep/check-http-signals.py FIXTURE
git diff --check
```

The before loop intentionally failed on run 11 and captured `build-sweep/shutdown-stall-backtrace.log`; the after loop passed all 20. Per-run client/server logs and JSON results are `build-sweep/cancel-{before,after}-*`. The other logs are `build-shutdown-fix.log`, `check-http-shutdown-fix.log`, `check-http-open-shutdown-fix.log`, `check-http-signals.log`, and `check-signal-dispatch{,-before}.log` in build-sweep. The local HTTP sweep harness now also fails on a shutdown timeout/nonzero exit, instead of allowing its cleanup kill to hide a failure. The signal regression harness is `build-sweep/check-signal-dispatch.cpp`; it includes the actual server.cpp and links the existing server/common/llama/ggml libraries, with CHECK_LEGACY_SIGNAL selecting the archived dbc6afea6 source for the negative control. These local harnesses remain available for L-B to adopt; this fix only changes server.cpp and this record. Linux/CPU validated; Windows was not run. No changes to model math or native Tier-3 gates, and no GPU/full-27B claims.

## 2026-10-20: one-prefill native G3 NLL scorer

Base: rf27-integ e3fc64ef2. Added target `llama-ridgefill-score` in tools/ridgefill, with a JSON/JSONL token-ID input, all four modes, and per-target full-vocabulary log-probabilities/logsumexp as JSONL. The model/context are reused across documents with memory cleared between them. There is no tokenization, sampling, or HTTP. Only K prediction rows enable the output head; host float64 reduction consumes each output batch without retaining full-vocabulary logits across batches. Split comes from the projector; defaults are T=4096, K=512, share=0.5, digits on, qattn=0.2, window=64, alpha=0, and F16 KV/h_S. Common class/digit selection and the engine sweep driver are reused. The new CMake target does not depend on enabling the HTTP server/UI.

Protocol correction found in the requested oracle inspection: `q27/rf27_quality.py` sets `known=N-scored`, captures `[known-window,known)`, freezes the row set, and calls `finish(...,keep=K+1)` followed by `score_tokens()` (drop the final logit row). `q27/drive.py` delegates quality tasks to q27.steps and does not define another qattn NLL window. Thus N=16384/K=512/T=4096 means P=12288, question rows [15808,15872), logits [15871,16383), and targets [15872,16384). Reading the whole document's last 64 rows would leak scored targets and does NOT match the reference. The scorer defaults to the reference boundary; an explicitly requested `--qattn-end document` diagnostic marks `selector_excludes_targets=false`. The paired wrapper always forces the reference boundary and checks target exclusion.

The driver gained optional `qattn_end` and a half-open exact-tail logits range, plus the first output position. Default server/deferred-token behavior is unchanged. The full lower pass still includes the document; causal attention makes target/future keys contribute zero selector mass. The gate replaces every target ID and checks bitwise-identical selector mass/rows and first predictive logsumexp.

`tools/ridgefill/run_nll.py` is the lane-Q wrapper. It prepares 16384-token documents from nll.jsonl, invokes off and quality+qattn once each over all documents (two model loads total), and writes the run_g3.py summary/paired-record structure. The reference's novel-token mask is primary, using kva.data.novel_mask; an all-token diagnostic is retained as nll_all because HTTP run_g3.py currently averages all tokens. BCa uses the existing eval_client.paired_summary and kva finals/stats.py (5000 resamples, seed 20261004, document weighting). No quality verdict is inferred; needles/TTFT/bench metrics are SKIPPED. Exact full-model commands and paths are in tools/ridgefill/README.md. Output directories must be new; failures produce FAILED.json and never a COMPLETE summary.

CPU tiny split-4 gates (FP32 model, KV and h_S; alpha=0; FA off):

```text
PASS native scorer: 24/24 four-mode oracle comparisons, two K values, three chunk sizes
K=64/T=200 and K=512/T=600, ubatches 50/128/512
worst absolute per-position logprob/logsumexp error: 1.81e-7 (bar 1e-4)
worst absolute document NLL error: 9.98e-9
PASS zero qattn share == quality byte-identical, at both K values and all chunk sizes
PASS off == separate stock rf_dump next-token scoring
PASS selector mass/rows equal window_mass/top_rows at the known boundary
PASS replacing all scored targets preserves selector mass/rows and first predictive LSE
PASS paired wrapper: actual BCa, repeated-span novel mask 496/512
PASS second document == fresh context byte-for-byte for off and quality+qattn
PASS named refusals: prediction outside exact tail, unsupported rho correction, invalid mode, nonfinite share
PASS existing Tier-3 native G1: 12/12 + 3/3 controls; unchanged worst error 6.63008223e-7
PASS existing HTTP: 17 comparisons, zero-share equality, repeat and concurrent-slot reuse
[100%] Built target llama-ridgefill-score
[100%] Built target llama-server
```

Commands: `cmake -S . -B build-sweep`; `cmake --build build-sweep --target llama-ridgefill-score llama-server -j4`; compile the existing tests/ridgefill/rf_dump.cpp against build-sweep/bin as score-stock-dump (README command); run tools/ridgefill/check_score.py with `--binary build-sweep/bin/llama-ridgefill-score --fixture ../llama.cpp-ridgefill-tools/out/gate1-qwen35 --kva /var/home/dylan/projects/research/kva-wt-q27 --stock-dump build-sweep/score-stock-dump --work build-sweep/score-gate-common`; rebuild the local g1-sweep harness against the extended driver and run g1-sweep-oracle.py and check-http-sweep.py. Evidence: build-sweep/check-score-common.log, check-score-driver-regression.log, check-score-http-regression.log, and score-gate-common/paired/summary.json. The committed gate also checks malformed/out-of-vocabulary token IDs; the final post-commit run is recorded in check-score-release.log.

Rejected options: 512 truncated HTTP prompts (512 prefills and a changing selector); a last-64-document selector for the primary result (target leakage); retaining logits for all T tail rows (unneeded output-head work and memory); substituting another bootstrap implementation (would change the registered interval). JSON/JSONL avoids adding an NPY parser. GPU/full-27B validation belongs to lane Q and was not run in this CPU sandbox.

## 2026-10-20: explicit selector endpoint for NLL and prompt protocols

Coordinator's reference relay matches `ridgefill-tune/REFERENCE-ARMS.md`, "For the engine lanes". Added `--qattn-window-end END`, accepting a positive exclusive row index, `document` (N), or `known` (N-K for each document). The default is now document end, matching server semantics; this supersedes the previous scorer-default statement above. `--qattn-end` remains an alias. G3's paired wrapper explicitly passes the numeric N-K endpoint and validates the resulting window and target exclusion. Its full-model command in tools/ridgefill/README.md is unchanged. A direct NLL invocation must pass `--qattn-window-end known` or the numeric endpoint, e.g. 15872 for N=16384/K=512. Teacher-forced needles can pass N-answer_length; G2 uses N. No server or driver default changes were needed.

For N=16384/K=512/T=4096 the NLL path still uses P=12288, question rows [15808,15872), prediction rows [15871,16383), and target IDs [15872,16384). Computing only the 512 prediction logits is equivalent to the reference's last 513 logits with the final unused row discarded. Lower computation and the frozen selector occur once per document.

Validation (CPU, split 4, FP32 model/KV/h_S, alpha=0, flash attention off):

```text
[100%] Built target llama-ridgefill-score
PASS native scorer: 24/24 four-mode oracle comparisons, two K values, three chunk sizes
worst absolute logprob/logsumexp error: 1.81e-7; tolerance: 1e-4
PASS window endpoints: default == N; N and N-32 match oracle mass/rows/logits; known and legacy alias == N-K
PASS paired wrapper: real BCa, novel mask 496/512, multi-document reuse equals fresh contexts
PASS window refusals: zero, negative, malformed, overflowing, beyond document and insufficient history
PASS existing zero-share, stock scoring, target replacement and invalid-input controls
```

Commands: `cmake --build build-sweep --target llama-ridgefill-score -j4`; `PYTHONDONTWRITEBYTECODE=1 python3 tools/ridgefill/check_score.py --binary build-sweep/bin/llama-ridgefill-score --fixture ../llama.cpp-ridgefill-tools/out/gate1-qwen35 --kva /var/home/dylan/projects/research/kva-wt-q27 --work build-sweep/score-gate-endpoint --stock-dump build-sweep/score-stock-dump`; `git diff --check`. Logs: build-sweep/build-score-endpoint.log and build-sweep/check-score-endpoint.log. The gate also verifies the wrapper command's numeric endpoint in provenance.json. Invalid numeric endpoints fail with the option name. No GPU or 27B run, and no quality/noninferiority verdict.

## 2026-10-20: chat selector controls and last-user-turn window

Base: rf27-integ d619c13d6. The coordinator resolved the window contract identically for L-C and lane T. Note for L-A to fold into RF27-INTERFACE.md: `turn` has a separate `qattn_turn_cap`, default 256; `qattn_window`, default 64, applies only to `last`. Tokenize the full literal `<|im_start|>user\n` with parse_special=true and no BOS. Search for its last complete occurrence in tokens[0,end) only. `turn_start` is the row AFTER the full header, including its newline; capture [max(turn_start,end-cap),end). The endpoint is N for server semantics, N-K for NLL. No marker or no rows after the marker: named refusal, no fallback. Blind turn selection is refused. This supersedes the earlier interface example that searched all N rows and included the header.

Implemented in server-ridgefill-sweep.{h,cpp}: qattn_window_mode (string, default last), qattn_turn_cap (int32, default 256), per-prompt resolved window bounds retained for output metadata, and buffers sized from those bounds. The scorer exposes --qattn-window-mode last|turn and --qattn-turn-cap; --special-exact on|off and --header-rows (with --ridgefill-* aliases) go directly to llama_ridgefill_select_rows_ext. Header counts are clipped to the prefix length before calling the selector, which is equivalent to clipping each selected range and prevents integer overflow for a very large count. Speed refuses enabled chat row rules. Off remains exact; zero qattn share uses ordinary quality with the same chat rows. The paired wrapper forwards all four controls, records them, validates the resulting window/options, and retains the reference novel-token/BCa protocol.

Driver ownership is unchanged: the caller supplies the sorted class/digit/special/header rows from select_rows_ext as base_rows, and the driver adds question rows. For L-B's server integration, fill sweep_params.qattn_window_mode and sweep_params.qattn_turn_cap from the effective request settings, and use common_ridgefill_rows after its select_rows_ext plumbing for base_rows. Its current fail-closed refusal of turn mode can be lifted when these fields are connected. This commit does not edit common or the per-request parser. The HTTP regression below covers existing/default behavior, not the pending HTTP chat knobs.

Gate tools/ridgefill/check_chat_score.py writes a synthetic ChatML fixture under its work directory: original tiny split-4 model weights and projector maps/scores, two renamed control-token pieces (<|im_start|>, <|im_end|>), disabled SPM dummy-space prefix, and the matching projector vocabulary hash. Role/newline tokens use the fixture's byte IDs, independently defining the complete user header as [1,120,118,104,117,13]. The oracle uses the unchanged HF model weights, the coordinator's window/row rules, and actual kva.select.window_mass / kva.kva.finish. This does not claim validation of lane T's not-yet-present turn-window wrapper. Cases include special-only, header-only, combined rules, overlapping headers at P-1, caps 32/256/1024, a 30-row turn, endpoint N versus N-K, and a future user header inside the scored continuation. Eight configurations run at ubatches 50/128/512.

Before editing, built the d619c13d6 scorer and saved build-sweep/bin/llama-ridgefill-score-before-chat. Its binary predates the source edits. Compared it with the new implicit and explicit defaults on raw and templated token streams, all four modes, F32 and F16 KV/h_S: every output log-probability, logsumexp, selected row, question-mass element and window bound is identical. Build IDs, timings and added parameter metadata are not part of the equality claim.

```text
[100%] Built target llama-ridgefill-score
[100%] Built target llama-server
PASS defaults: old binary == new default == explicit defaults, raw/templated tokens, all four modes, F32/F16 KV and h_S
PASS chat oracle: 24/24 comparisons (8 configurations, 3 chunk sizes)
worst absolute logprob/logsumexp error: 1.5213828952198583e-7 (bar 1e-4)
worst relative question-mass error: 2.394300415746324e-7 (bar 1e-4)
PASS future user headers/targets cannot change selector mass, rows, bounds or first predictive LSE
PASS turn ignores last-window size; zero qattn share == chat quality byte-identical
PASS chat paired wrapper: per-document turn bounds, fresh-context equality and actual BCa
PASS missing/future-only/partial/empty-header refusals and incompatible/invalid-option refusals
PASS original scorer gate: 24/24 four-mode comparisons, endpoint/stock/zero-share/target-exclusion/BCa controls
PASS HTTP defaults: 17 comparisons, zero-share equality, repeat, cancellation/reuse and concurrent slots
```

Commands: `cmake --build build-sweep --target llama-ridgefill-score llama-server -j4`; `PYTHONDONTWRITEBYTECODE=1 python3 tools/ridgefill/check_chat_score.py --binary build-sweep/bin/llama-ridgefill-score --baseline-binary build-sweep/bin/llama-ridgefill-score-before-chat --fixture ../llama.cpp-ridgefill-tools/out/gate1-qwen35 --kva /var/home/dylan/projects/research/kva-wt-q27 --work build-sweep/score-gate-chat-f16`; run the original check_score.py into build-sweep/score-gate-chat-regression; `PYTHONDONTWRITEBYTECODE=1 python3 build-sweep/check-http-sweep.py ../llama.cpp-ridgefill-tools/out/gate1-qwen35`; `git diff --check`. Evidence: build-sweep/build-score-chat.log, check-score-chat-f16.log, check-score-chat-regression.log, check-http-chat-default.log, and score-gate-chat-f16/summary.json. Rejected options: using the 64-row last-window size as the turn cap (different experiment), including header rows in the query window, searching scored future rows, and silently falling back when no user turn exists. CPU correctness only; no GPU/full-model chat quality or release claim.

## 2026-10-21: S1/vtail orchestration draft, blocked on libllama C API v2

Created rf27-sweep-s1 from 2e06536c2, after reading the entire S1 C API v2 addition in RF27-INTERFACE.md. That commit contains the interface only. At this checkpoint rf27-s1 still points to it and src/llama-ext.h has no API_VERSION/S1/vtail declarations. The v2-dependent code is compile-guarded and requests fail by name with API v1. This is a driver/scorer draft, NOT a built or passing S1 implementation; its v2 paths cannot yet be compiled or executed against L-A's library.

Implemented the orchestration in the driver: LOWER -> lower qattn mass/base count -> S1_FILL [0,P) -> S1_PROBE [end-W,end) -> get mass/top-k/special union -> set rows -> optional vtail -> LATE prefix/tail -> disable vtail -> FULL. Added an integer-count top-k helper so the half-even S1 budget is not converted back to a floating share. S1 exact rows replace the base rows; only requested special rows are explicitly united back in. Factor 0 selects at least one row; the budget saturates at P. Existing cancellation clears sequence memory before restoring FULL. The new C API must preserve that existing empty-sequence reset contract.

Vtail uses a first LATE tail decode of at least min(C,N-P) rows. The scorer raises context batch/ubatch capacities to fit; the driver refuses undersized contexts and the short-tail case where calibration would consume the server's deferred final token. The scorer refuses calibration reaching scored NLL targets rather than changing C. S1/vtail requests default to end=N-K in the scorer; explicit endpoint overrides still work, and serving defaults to N. Both features default off. S1 is restricted to quality+qattn with positive qattn share and no header-row rule; registered class/digit/special settings are supplied through the existing base selector. Exact controls can carry the new options with mode off and remain exact.

Reporting is explicit about extra work: S1_FILL = P*(H^2 + K/V projection MACs) at each late attention layer; S1_PROBE = W exact rows at every late layer over the visible prefix plus causal window; vtail = calibration rows*(H^2 + V projection MACs) at each late attention layer. The standalone scorer adds all three terms to the existing analytic cold-prefill cost and reports the counts/terms individually, with prefix_kv_reused=false. Final source counts use s1 rather than qattn. Note for L-B: wire the new driver fields s1/s1_window/s1_factor/projector/special_exact/vtail/vtail_rows, consume get_cost()/base_row_count()/get_s1_mass()/get_vtail_scales(), and extend the strict source-key validator to accept s1. run_nll.py and the HTTP parser/reporting have NOT been modified in this draft, per ownership; their current source-key validator rejects an S1 result.

Found lane T's completed dumps at kva-wt-q27/ridgefill-tune/engine-ref/s1-vtail without needing the pending path relay. MANIFEST records kva commit 9dcf6d1a487e39b16d8b6796bdba17539b5e745e, N=700/P=500/S=4/W=64/end=700, two prompts, T3s/S1gw with and without vtail. The gate verifies SHA256SUMS and the tiny input hashes. It compares the last 64 target log-probabilities/LSE against the corresponding last 65 oracle logits with the final row dropped, rows exactly, S1 mass, and vtail scales for c=128 (rescale_skip_last=72), at chunks 50/128/512. Cost expectations use kva.compute and explicitly add the S1 prefix projector/KV pass. The c=200 oracle case is left to L-A's native harness because that calibration reads this scorer's NLL targets and is refused. L-A's native harness/port is not yet present at this checkpoint; native-harness comparison and the N-K S1 mass gate remain pending.

Saved the baseline executable from 2e06536c2 as build-sweep/bin/llama-ridgefill-score-before-s1 (build 11883). Current completed checks:

```text
[100%] Built target llama-ridgefill-score
[100%] Built target llama-server
PASS defaults: byte-identical numerical outputs, row selection and cost report; two prompts, four modes, F32/F16
PASS native scorer: 24/24 four-mode oracle comparisons, two K values, three chunk sizes
PASS HTTP defaults: 17 comparisons, repeat, zero-share equality, cancellation/reuse, concurrent slots
PASS named API v1 refusals for S1 and vtail
SKIPPED S1 mass/rows/logits, vtail and S1 cost: libllama C API v2 is not available (gate exit 77)
```

Commands: `git switch -c rf27-sweep-s1 2e06536c2`; `cmake --build build-sweep --target llama-ridgefill-score llama-server -j4`; the original tools/ridgefill/check_score.py with work=build-sweep/score-gate-s1-default; tools/ridgefill/check_s1_score.py with the paths in its README command, work=build-sweep/score-gate-s1-final-v1 and the saved baseline; `PYTHONDONTWRITEBYTECODE=1 python3 build-sweep/check-http-sweep.py ../llama.cpp-ridgefill-tools/out/gate1-qwen35`; `git diff --check`. Logs: build-sweep/build-s1-final-v1.log, check-score-s1-default.log, check-score-s1-final-v1.log, check-http-s1-default.log. An initial server rebuild caught a missing relative include path for llama-ext.h; corrected it, and the subsequent full CPU build passed. No GPU work, no push, no quality or speedup claim. Next: rebase onto L-A's implementation, compile the guarded paths, run the frozen-reference/native-harness gates, and relay the source-key/cost interface to L-B.
