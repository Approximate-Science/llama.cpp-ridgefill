# RidgeFill conversion checks

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
