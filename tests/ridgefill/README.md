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

Speed cases compare tail logits, every layer's final KV/conv/recurrent state, and every late layer's prefix state at N=700/T=200 over batch/ubatch pairs 700/512, 128/128 and 50/50. Prefix dumps are necessary to expose a wrong alpha correction before tail decay. Alpha is 0 and 1. Controls require byte-identical mode-off and T>=N outputs, a wrong map that fails by over 100x tolerance, a wrong correction orientation at alpha 1, and refusal of a filled row asking for logits. The tolerance is 1e-4 relative F32.

Quality accepts hand-picked rows through the RF27 API. The rho-scaled alpha-1 reference and the quality+qattn sweep integration are pending; those cases refuse by name and do not return a passing gate. Split 4 is already available for L-C's mass comparison.
