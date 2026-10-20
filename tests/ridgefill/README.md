# RidgeFill conversion checks

Run from the repository root with NumPy and safetensors installed. The converter uses the repository's `gguf-py` and needs no torch.

```sh
python3 tests/ridgefill/test_converter.py -v
python3 convert_ridgefill_to_gguf.py /path/to/int8-package --outfile out/projector.gguf
python3 tests/ridgefill/check_converter.py /path/to/int8-package out/projector.gguf
```

The package check verifies all SHA256SUMS entries and manifest file hashes, metadata preservation, every map's error bound, exact biases and row scores, and the correction head permutation. It must reject a different layer's map and an unpermuted correction. Packages without nontrivial controls produce SKIPPED with a nonzero exit code.

Q8_0 maps are compared with `codes * scale` using each 32-element block's rounding bound, including its fp16 scale error. F32 maps must be exact. F16 and BF16 maps must equal the corresponding rounding of the source values. These are conversion checks; they do not claim oracle agreement of the inference engine.
