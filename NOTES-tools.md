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

Next: tiny qwen3_5 fixture, local CPU oracle venv, then G1 speed harness. Generate both split-3 (speed/quality) and split-4 (quality+qattn) projectors for the same 8-layer model; split 3 has no attention layer below it, while split 4 has layer 3. Planned paths are `out/gate1-qwen35/projector-s3-f32.gguf` and `out/gate1-qwen35/projector-s4-f32.gguf`; neither fixture is produced yet. The harness will take the split as a parameter.

Coordinator assigned L-B the tool-side calls in server, perplexity and bench, including per-request modes and class/digit row selection. L-C retains its server sweep driver function. No tool-side implementation is claimed at this converter milestone.
