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
| Question graph agrees with `select.window_mass` within 1e-4 | `PYTHONDONTWRITEBYTECODE=1 python3 build-sweep/check-mass.py /var/home/dylan/projects/research/kva-wt-q27` | PASS for the standalone graph using oracle lower-layer inputs/keys |
| h_S capture of a tiny split-4 lower pass matches PyTorch conversion | Same Python command, chunk sizes 700/128/50 | PASS |
| Engine lower-only graph stops at S and captures its own h_S | L-A dispatch/context hook plus L-B fixture | SKIPPED: touch points are specified but not implemented yet |
| Engine cache mapping and layer-input extraction feed the mass graph | Integrated lower sweep and slot cache | SKIPPED |
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

No L-B tiny model was available when these tests began. Under the coordinator's authorization, `check-mass.py` creates a throwaway local fixture and random split-4 projector in `build-sweep/tiny-qwen35-local/`: seed 20261020, eight layers, attention at 3/7, hidden 64, attention head dimension 16, four Q heads/two KV heads, GDN two key heads/four value heads of width 16, conv width 4, vocabulary 300. It writes safetensors directly, then runs the actual oracle `Qwen35`, `run_lower`, `window_mass`, and `top_rows` in CPU FP32. N=700, P=500; windows end at 700, 500, and 37, with window lengths 64, 37, and 1. This is not yet L-B's shared fixture or a converted GGUF.

At width 5120 with two-byte elements, the 1 GiB cap admits at most 104857 tokens; 32768 tokens need 335544320 bytes. The cap is per buffer; graph scratch and captured query inputs are additional allocations. F32 is available internally for captured query windows and strict FP32 oracle checks. Four F32 query windows of 64 rows and width 5120 need 5242880 bytes.

Measured h_S rounding above demonstrates that two-byte storage is not bit-identical to the FP32 lower pass. Use F32 storage for the strict FP32 integration gate; separately measure the shipped F16/BF16 path instead of silently weakening the tolerance. End-to-end logits with either two-byte format remain unmeasured.

`select.choose` budgets against N, but `scripts/q27_quality.py:214` confirms the 27B caller budgets against P. Follow SPEC's prefix budget and `select.top_rows` position ties.

## Interface handoff and next step

Read L-A's `RF27-INTERFACE.md`, commit `9e1c3d076`. The proposed `llama_ridgefill_seq_set_part` and PART_LOWER/PART_LATE dispatch were still absent from L-A's implementation at the final inspection. The local branch remains based on `85b1c41fb`; rebase awaits the coordinator's/L-A's ready notification.

- L-A can use `llama_ridgefill_sweep_buffer::tensor()` for the specified `h_buf`, allocated with `model.select_buft(split)`. `capture()` and `slice()` also provide per-ubatch graph copies/views, avoiding per-row host copying described in the interface note. These views bake the row offset into the graph, so a changed offset requires rebuilding or must be handled by the context copy path.
- Layer-input extraction from the existing embeddings API is the agreed query-window source. The mass helper accepts the actual window size through `llm_graph_context::n_tokens` and four positions per query row, then keys `[head_dim, N, Hkv]` in prompt order. The future driver must gather the slot's logical rows from the cache before calling it.
- L-B should supply the split-4 fixture variant and adopt the standalone checks into its owned harness. Split 3 with attention at 3/7 contains no lower attention and cannot validate question mass.
- Next implement lower-sweep execution and slot cache mapping against L-A's landed touch points, then sweeps 2-3 and the server driver after G1 speed.

The initial missing skill path is resolved: read `/var/home/dylan/AI-Work/ridgefill-ports-20261020/skills/build-anything.md` sections 2-4. No GPU, Docker, push, or publication was attempted.
