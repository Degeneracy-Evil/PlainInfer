"""只读取配置和 checkpoint 元数据, 不加载完整模型。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from safetensors import safe_open

from plaininfer.config import ModelConfig


def find_checkpoint_files(model_dir: str | Path) -> list[Path]:
    """查找 Hugging Face 模型目录中的 safetensors 分片。"""
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError("模型目录中没有找到 *.safetensors")
    return files


def inspect_checkpoint(model_dir: str | Path) -> dict[str, tuple[int, ...]]:
    """读取所有 tensor 名称和 shape, 不把权重整体载入内存。"""
    tensors: dict[str, tuple[int, ...]] = {}
    for checkpoint_path in find_checkpoint_files(model_dir):
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
            tensor_names = checkpoint.keys()
            for name in tensor_names:
                if name in tensors:
                    raise ValueError(f"checkpoint 中存在重复 tensor: {name}")
                tensors[name] = tuple(checkpoint.get_slice(name).get_shape())
    return tensors


def format_report(config: ModelConfig, tensors: dict[str, tuple[int, ...]]) -> str:
    """生成便于人工检查的配置与 tensor shape 报告。"""
    config_lines = (
        f"hidden_size: {config.hidden_size}",
        f"intermediate_size: {config.intermediate_size}",
        f"num_hidden_layers: {config.num_hidden_layers}",
        f"num_attention_heads: {config.num_attention_heads}",
        f"num_key_value_heads: {config.num_key_value_heads}",
        f"num_key_value_groups: {config.num_key_value_groups}",
        f"head_dim: {config.head_dim}",
        f"vocab_size: {config.vocab_size}",
        f"max_position_embeddings: {config.max_position_embeddings}",
        f"rope_theta: {config.rope_theta}",
        f"rms_norm_eps: {config.rms_norm_eps}",
        f"tensor_count: {len(tensors)}",
    )
    tensor_lines = tuple(f"{name}: {shape}" for name, shape in sorted(tensors.items()))
    return "\n".join((*config_lines, "", "checkpoint tensors:", *tensor_lines))


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="检查 Qwen2.5 配置与 checkpoint")
    parser.add_argument("model_dir", nargs="?", type=Path, help="Hugging Face/ModelScope 模型目录")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """运行模型检查。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.model_dir is None:
        parser.print_help()
        return 0
    config = ModelConfig.from_model_dir(args.model_dir)
    tensors = inspect_checkpoint(args.model_dir)
    print(format_report(config, tensors))
    return 0
