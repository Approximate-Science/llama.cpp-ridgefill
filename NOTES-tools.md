# L-B tools notes

## 2026-10-20: initial read

Read KICKOFF.md, LANE-LLAMACPP.md, SPEC.md, FACTS.md sections 1-3, HANDOVER.md, the repository code-review skill, CONTRIBUTING.md, the existing converter and RIDGEFILL.md, the release manifest, and kva/gguf_ckpt.py.

The required `/var/home/dylan/.claude/skills/build-anything/SKILL.md` was unavailable because `.claude` is hidden. Resolved by reading the coordinator's copy at `/var/home/dylan/AI-Work/ridgefill-ports-20261020/skills/build-anything.md`.

Branch is `rf27-tools`; the initial working tree was clean.

The engine interface note was also absent at the time of the initial read; the lane brief permits using its provisional metadata names until that note arrives.

## Converter milestone

Implemented the format-1 int8 folder converter with no torch dependency. The legacy Llama/Qwen3 single-file conversion stays supported. All 55 SHA256SUMS entries and every manifest file checksum were verified. The output preserves the complete manifest, source hashes, credit, model metadata, tokenizer hash, layer types and defaults. Each map is dequantized from codes times bf16 scales before Q8_0 conversion. Correction heads are tiled from HF grouped order into llama.cpp order. Biases, correction and score tensors are F32.

Output: `out/ridgefill-projector-qwen3.8-27b-i8-S16.gguf` (1452188224 bytes).

- GGUF SHA256: `df3dee457f8f444788a5eab16915ad2147762282d79bb5d2ab7dbbb80eac826f`
- Source ridgefill.json SHA256: `1ba7dd4a71b20c815c59578e2b5401d510b5cd3ca7e2fa888587b06a869ec558`

Commands run from this folder:

```sh
python3 convert_ridgefill_to_gguf.py /var/mnt/qwen-storage/dylan/AI-Work/ridgefill-q27-projector-20261008/release/ridgefill-projector-qwen3.8-27b-i8/ --outfile out/ridgefill-projector-qwen3.8-27b-i8-S16.gguf
python3 tests/ridgefill/check_converter.py /var/mnt/qwen-storage/dylan/AI-Work/ridgefill-q27-projector-20261008/release/ridgefill-projector-qwen3.8-27b-i8/ out/ridgefill-projector-qwen3.8-27b-i8-S16.gguf
python3 tests/ridgefill/test_converter.py -v
git diff --check
```

Measured round-trip output:

```json
{"bias_and_score_exact":"PASS","head_order_control":"PASS","maps":48,"max_map_abs_error":0.014892578125,"metadata":"PASS","result":"PASS","states":36,"tensors":133,"wrong_map_control":"PASS"}
```

Regression output: `Ran 7 tests ... OK`. Covers F32/F16/BF16/Q8_0 outputs, wrong-map and wrong-head controls, corruption preserving existing output, missing file, wrong correction shape removing partial output, base-model metadata mismatch, tokenizer mismatch, and legacy conversion. Logs are `out/converter.log`, `out/converter-roundtrip.json`, and `out/converter-tests.txt`.

Rejected: loading maps with torch (unnecessary dependency); NumPy safetensors get_tensor for BF16 (unsupported by NumPy). Use safetensors' validated deserialize and expand the BF16 bit representation. Rejected holding all expanded maps in memory: spool tensors beside the destination and process one map at a time. Support the tiny fixture's partial 128-column group with width 64.

Type checking: ty 0.0.84 passed on all three changed Python files, using the system Python and an explicit search path for its user-installed NumPy and safetensors. `git diff --check` passed. The release round-trip is a container/conversion check; G0/G1 engine agreement is NOT RUN.

Next: tiny qwen3_5 fixture, local CPU oracle venv, then G1 speed harness. The engine interface note must be checked when it appears; current GGUF names follow LANE-LLAMACPP.md.
