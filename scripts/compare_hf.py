"""在同一 FP32 输入上比较 PlainInfer 与 Hugging Face 中间结果。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as apply_reference_rope

from plaininfer.comparison import error_stats
from plaininfer.layers import DecoderLayer
from plaininfer.model import PlainInferModel
from plaininfer.rope import RotaryEmbedding, apply_rotary_pos_emb


def tensor_output(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> torch.Tensor:
    """从普通 tensor 或 Hugging Face tuple 输出中取得 hidden state。"""
    value = output[0] if isinstance(output, tuple) else output
    if not isinstance(value, torch.Tensor):
        raise TypeError("hook 输出不是 tensor")
    return value.detach()


def capture(module: nn.Module, values: dict[str, torch.Tensor], name: str) -> torch.utils.hooks.RemovableHandle:
    """注册一次 forward hook。"""

    def hook(target: nn.Module, inputs: tuple[object, ...], output: object) -> None:
        values[name] = tensor_output(target, inputs, output)

    return module.register_forward_hook(hook)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--prompt", default="Hello, world!")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    plain = PlainInferModel.from_pretrained(args.model_dir, dtype=torch.float32)
    reference: Any = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float32, local_files_only=True, attn_implementation="eager"
    )
    reference.eval()

    ours: dict[str, torch.Tensor] = {}
    theirs: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    first_layer = cast(DecoderLayer, plain.layers[0])
    pairs: list[tuple[str, nn.Module, nn.Module]] = [
        ("embedding", plain.embed_tokens, reference.model.embed_tokens),
        ("layer0.input_norm", first_layer.input_layernorm, reference.model.layers[0].input_layernorm),
        ("layer0.q_proj", first_layer.self_attn.q_proj, reference.model.layers[0].self_attn.q_proj),
        ("layer0.k_proj", first_layer.self_attn.k_proj, reference.model.layers[0].self_attn.k_proj),
        ("layer0.v_proj", first_layer.self_attn.v_proj, reference.model.layers[0].self_attn.v_proj),
        ("layer0.attention", first_layer.self_attn, reference.model.layers[0].self_attn),
        ("layer0.mlp", first_layer.mlp, reference.model.layers[0].mlp),
        ("final_norm", plain.norm, reference.model.norm),
    ]
    pairs.extend(
        (f"layer{index}.output", layer, reference.model.layers[index]) for index, layer in enumerate(plain.layers)
    )
    for name, own_module, reference_module in pairs:
        handles.extend((capture(own_module, ours, name), capture(reference_module, theirs, name)))
    with torch.no_grad():
        own_logits = plain(input_ids)
        reference_logits = reference(input_ids).logits
    for handle in handles:
        handle.remove()

    # PlainInfer 和 Hugging Face 分别使用自己的 RoPE 实现, 确保这一层比较彼此独立。
    batch, sequence = input_ids.shape
    positions = torch.arange(sequence).unsqueeze(0)
    own_query = ours["layer0.q_proj"].view(batch, sequence, plain.config.num_attention_heads, -1).transpose(1, 2)
    own_key = ours["layer0.k_proj"].view(batch, sequence, plain.config.num_key_value_heads, -1).transpose(1, 2)
    own_rope = RotaryEmbedding(plain.config.head_dim, plain.config.rope_theta)
    own_cos, own_sin = own_rope(positions, torch.float32)
    ours["layer0.q_rope"], ours["layer0.k_rope"] = apply_rotary_pos_emb(own_query, own_key, own_cos, own_sin)

    reference_query = (
        theirs["layer0.q_proj"].view(batch, sequence, plain.config.num_attention_heads, -1).transpose(1, 2)
    )
    reference_key = theirs["layer0.k_proj"].view(batch, sequence, plain.config.num_key_value_heads, -1).transpose(1, 2)
    reference_cos, reference_sin = reference.model.rotary_emb(theirs["embedding"], positions)
    theirs["layer0.q_rope"], theirs["layer0.k_rope"] = apply_reference_rope(
        reference_query,
        reference_key,
        reference_cos,
        reference_sin,
    )
    ours["logits"] = own_logits
    theirs["logits"] = reference_logits
    for name in ours:
        stats = error_stats(ours[name], theirs[name])
        print(
            f"{name:24} max_abs={stats.max_absolute:.8g} mean_abs={stats.mean_absolute:.8g} max_rel={stats.max_relative:.8g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
