"""Hugging Face safetensors 到 PlainInfer 参数的显式加载。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from plaininfer.inspection import find_checkpoint_files


def parameter_name_map(num_layers: int, tie_word_embeddings: bool, attention_bias: bool = True) -> dict[str, str]:
    """列出 checkpoint 名称到 PlainInfer state_dict 名称的全部映射。"""
    mapping = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.norm.weight": "norm.weight",
    }
    if not tie_word_embeddings:
        mapping["lm_head.weight"] = "lm_head.weight"
    for layer_index in range(num_layers):
        source = f"model.layers.{layer_index}"
        target = f"layers.{layer_index}"
        for norm_name in ("input_layernorm", "post_attention_layernorm"):
            mapping[f"{source}.{norm_name}.weight"] = f"{target}.{norm_name}.weight"
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mapping[f"{source}.self_attn.{projection}.weight"] = f"{target}.self_attn.{projection}.weight"
        if attention_bias:
            for projection in ("q_proj", "k_proj", "v_proj"):
                mapping[f"{source}.self_attn.{projection}.bias"] = f"{target}.self_attn.{projection}.bias"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            mapping[f"{source}.mlp.{projection}.weight"] = f"{target}.mlp.{projection}.weight"
    return mapping


def read_checkpoint(model_dir: str | Path) -> dict[str, torch.Tensor]:
    """逐分片读取 checkpoint。第一版模型可直接放入 CPU 内存。"""
    tensors: dict[str, torch.Tensor] = {}
    for path in find_checkpoint_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            names = checkpoint.keys()
            for name in names:
                if name in tensors:
                    raise ValueError(f"checkpoint 中存在重复 tensor: {name}")
                tensors[name] = checkpoint.get_tensor(name)
    return tensors


def load_checkpoint(module: nn.Module, tensors: Mapping[str, torch.Tensor], mapping: Mapping[str, str]) -> None:
    """严格检查名称和 shape 后复制权重。"""
    parameters = _validate_mapping(module, {name: tensor.shape for name, tensor in tensors.items()}, mapping)
    with torch.no_grad():
        for source_name, target_name in mapping.items():
            parameters[target_name].copy_(tensors[source_name])


def load_checkpoint_from_dir(module: nn.Module, model_dir: str | Path, mapping: Mapping[str, str]) -> None:
    """先校验全部元数据, 再逐个 tensor 复制, 避免同时保留整份转换后权重。"""
    locations: dict[str, Path] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    for path in find_checkpoint_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            names = checkpoint.keys()
            for name in names:
                if name in locations:
                    raise ValueError(f"checkpoint 中存在重复 tensor: {name}")
                locations[name] = path
                shapes[name] = tuple(checkpoint.get_slice(name).get_shape())

    parameters = _validate_mapping(module, shapes, mapping)
    sources_by_file: dict[Path, list[str]] = {}
    for source_name in mapping:
        sources_by_file.setdefault(locations[source_name], []).append(source_name)

    with torch.no_grad():
        for path, source_names in sources_by_file.items():
            with safe_open(path, framework="pt", device="cpu") as checkpoint:
                for source_name in source_names:
                    parameters[mapping[source_name]].copy_(checkpoint.get_tensor(source_name))


def _validate_mapping(
    module: nn.Module,
    source_shapes: Mapping[str, torch.Size | tuple[int, ...]],
    mapping: Mapping[str, str],
) -> dict[str, torch.Tensor]:
    """统一校验 checkpoint key、module key 和每个 tensor shape。"""
    expected_sources = set(mapping)
    actual_sources = set(source_shapes)
    missing = sorted(expected_sources - actual_sources)
    unexpected = sorted(actual_sources - expected_sources)
    if missing or unexpected:
        raise KeyError(f"missing keys: {missing}; unexpected keys: {unexpected}")

    parameters = module.state_dict()
    expected_targets = set(mapping.values())
    module_keys = set(parameters)
    if expected_targets != module_keys:
        raise KeyError(
            f"module missing keys: {sorted(expected_targets - module_keys)}; "
            f"module unexpected keys: {sorted(module_keys - expected_targets)}"
        )

    for source_name, target_name in mapping.items():
        target = parameters[target_name]
        source_shape = source_shapes[source_name]
        if tuple(source_shape) != tuple(target.shape):
            raise ValueError(
                f"shape mismatch: {source_name} {tuple(source_shape)} -> {target_name} {tuple(target.shape)}"
            )
    return parameters
