#!/usr/bin/env python3
"""Generate a CPU-only Qwen3.5 fixture and format-1 random projectors at splits 3 and 4.

The gates feed token IDs, so the vocabulary is synthetic. Conversion uses the upstream
convert_hf_to_gguf entry point with only its vocabulary method replaced, as in the Oct-8 gates.
Projector maps are tested at their dequantized release values, not at the pre-quantization fit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import convert_ridgefill_to_gguf as converter

gguf = converter.gguf


def synthetic_vocab(n):
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)]
    types = [gguf.TokenType.UNKNOWN, gguf.TokenType.CONTROL, gguf.TokenType.CONTROL] + [gguf.TokenType.BYTE] * 256
    tokens += ["word" + chr(65 + i // 26) + chr(65 + i % 26) for i in range(n - len(tokens))]
    types += [gguf.TokenType.NORMAL] * (n - len(types))
    tokens[269] = "Digit7"
    return tokens, types


def set_vocab(self):
    n = self.hparams["vocab_size"]
    tokens, types = synthetic_vocab(n)
    w = self.gguf_writer
    w.add_tokenizer_model("llama")
    w.add_tokenizer_pre("default")
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * n)
    w.add_token_types(types)
    w.add_unk_token_id(0)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)


def seal(folder: Path, manifest: dict) -> None:
    manifest["files"] = {p.name: converter.file_sha256(p) for p in sorted(folder.glob("*.safetensors"))}
    (folder / "ridgefill.json").write_text(json.dumps(manifest, indent=2) + "\n")
    paths = sorted(folder.glob("*.safetensors")) + [folder / "ridgefill.json"]
    (folder / "SHA256SUMS").write_text("".join(f"{converter.file_sha256(p)}  {p.name}\n" for p in paths))


def make_projector(output: Path, hf: Path, config: dict, model_gguf: Path, split: int, wrong: bool = False, wrong_state: bool = False) -> Path:
    name = f"projector-s{split}" + ("-wrong" if wrong else "") + ("-wrong-st" if wrong_state else "")
    folder = output / name
    folder.mkdir(exist_ok=True)
    layers = range(split, 8)
    kinds = config["layer_types"]
    generator = torch.Generator().manual_seed(1000 + split)
    weights = {i: (torch.eye(64) + 0.3 * torch.randn(64, 64, generator=generator) / 8).float() for i in layers}
    biases = {i: (0.1 * torch.randn(64, generator=generator)).to(torch.bfloat16) for i in layers}
    recurrent = [i for i in layers if kinds[i] == "linear_attention"]
    corrections = {i: 0.05 * torch.randn(4, 16, 16, generator=generator) for i in recurrent}
    for st in corrections.values():
        if torch.equal(st, st.transpose(-1, -2)):
            raise AssertionError("fixture correction must be non-symmetric")
    for il in layers:
        source = min(il + 1, 7) if wrong else il
        weight = weights[source]
        # The tiny width is one partial 128-column group.
        scale = (weight.abs().amax(-1, keepdim=True) / 127).to(torch.bfloat16)
        scale[scale == 0] = 1
        codes = torch.round(weight / scale.float()).clamp(-127, 127).to(torch.int8)
        save_file({f"proj.{il}.codes": codes, f"proj.{il}.scale": scale, f"proj.{il}.bias": biases[source]},
                  str(folder / f"proj8.L{il}.safetensors"))
    # The oracle adds [H, K, V]; the release package stores [H, V, K].
    save_file({f"st.{i}": (st if wrong_state else st.transpose(-1, -2)).contiguous() for i, st in corrections.items()},
              str(folder / "correction.safetensors"))
    score = torch.full((300,), -torch.inf)
    score[259::2] = torch.arange(len(score[259::2]), dtype=torch.float32)
    score[299] = score[297]  # exercise ties by position
    save_file({"score": score}, str(folder / "rowsel.safetensors"))
    keys = ("num_hidden_layers", "hidden_size", "intermediate_size", "vocab_size", "num_attention_heads", "num_key_value_heads",
            "head_dim", "full_attention_interval", "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim")
    manifest = dict(format=1, name=name, authors=["RidgeFill test fixture"], license="MIT", doi="test-fixture",
                    model=dict(model_type="qwen3_5_text", meta={k: config[k] for k in keys}, layer_types=" ".join(kinds),
                               tokenizer_json_sha256=converter.file_sha256(hf / "tokenizer.json")),
                    split=split, stream_width=64, block_in_width=64,
                    layers={str(i): "recurrent" if kinds[i] == "linear_attention" else "full_attn" for i in layers},
                    tail=dict(default=200, min=1), correction=dict(alpha_default=0.0, layers=recurrent, layout="[heads, V, K]"),
                    rowsel=dict(share_default=0.5, qattn_default=0.2, digits_default=True),
                    projector=dict(encoding="i8*bf16[1x128]", files={str(i): f"proj8.L{i}.safetensors" for i in layers}))
    seal(folder, manifest)
    path = output / (name + "-f32.gguf")
    converter.convert_package(argparse.Namespace(projector=folder, model_gguf=model_gguf, base=hf, outtype="f32", outfile=path))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "out/gate1-qwen35")
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if torch.version.cuda is not None or torch.version.hip is not None:
        raise ValueError("use the CPU-only oracle venv for fixture generation")
    torch.manual_seed(0)
    kinds = (["linear_attention"] * 3 + ["full_attention"]) * 2
    rope_parameters: dict = dict(rope_type="default", rope_theta=10000000, partial_rotary_factor=0.25,
                                 mrope_interleaved=True, mrope_section=[1, 1, 0])
    cfg = Qwen3_5TextConfig(vocab_size=300, hidden_size=64, intermediate_size=160, num_hidden_layers=8,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, layer_types=kinds,
                          linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                          linear_value_head_dim=16, linear_conv_kernel_dim=4, rms_norm_eps=1e-6,
                          max_position_embeddings=4096, tie_word_embeddings=False, bos_token_id=1, eos_token_id=2, pad_token_id=0,
                          rope_parameters=rope_parameters)
    model = Qwen3_5ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.dim() == 1 and "A_log" not in name and "dt_bias" not in name:
                parameter.add_(0.1 * torch.randn_like(parameter))
    hf = output / "tiny-qwen35-hf"
    model.save_pretrained(str(hf), safe_serialization=True)
    config = json.loads((hf / "config.json").read_text())
    config["full_attention_interval"] = 4
    (hf / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    from tokenizers import Tokenizer, models
    tokens, _ = synthetic_vocab(300)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(vocab={token: i for i, token in enumerate(tokens)}, unk_token="<unk>")),
                                       unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<unk>")
    tokenizer.save_pretrained(hf)
    from conversion.qwen import Qwen3_5TextModel
    import convert_hf_to_gguf
    Qwen3_5TextModel.set_vocab = set_vocab
    model_gguf = output / "tiny-qwen35-f32.gguf"
    sys.argv = ["convert_hf_to_gguf.py", str(hf), "--outfile", str(model_gguf), "--outtype", "f32", "--no-mtp"]
    convert_hf_to_gguf.main()
    ids = np.random.default_rng(17).integers(259, 300, size=700, dtype=np.int32)
    ids.tofile(output / "tokens.bin")
    projectors = []
    for split in (3, 4):
        for wrong in (False, True):
            projectors.append(str(make_projector(output, hf, config, model_gguf, split, wrong)))
        projectors.append(str(make_projector(output, hf, config, model_gguf, split, wrong_state=True)))
    result = dict(model=str(model_gguf), hf=str(hf), projectors=projectors, tokens=700,
                  torch=torch.__version__, transformers=__import__("transformers").__version__, seed=0,
                  head_dim=16, gdn_key_heads=2, gdn_value_heads=4, non_symmetric_st=True)
    (output / "fixture.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
