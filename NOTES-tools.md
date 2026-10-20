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

Pending by name: quality alpha-1 needs a rho reference; quality+qattn needs the sweep dump/mass integration. Both refuse rather than returning success. Server/perplexity/bench plumbing and the evaluation client are still pending. Next: relay the current qwen35 loader refusal to L-A, rerun G1 speed when that path lands, then complete tool-side plumbing.
