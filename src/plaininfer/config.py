"""模型配置读取。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self


@dataclass(frozen=True)
class ModelConfig:
    """PlainInfer 第一版计算所需的 Qwen2.5 配置。"""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool
    bos_token_id: int
    eos_token_id: int
    attention_bias: bool = True

    @property
    def head_dim(self) -> int:
        """每个 attention head 的维度。"""
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        """每个 KV head 被多少个 query heads 共享。"""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """从 Hugging Face config 字典读取并校验第一版所需字段。"""
        if data.get("model_type") != "qwen2":
            raise ValueError("PlainInfer v0 只支持 model_type=qwen2")

        config = cls(
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            num_key_value_heads=int(data["num_key_value_heads"]),
            vocab_size=int(data["vocab_size"]),
            max_position_embeddings=int(data["max_position_embeddings"]),
            rope_theta=float(data.get("rope_theta", 10000.0)),
            rms_norm_eps=float(data["rms_norm_eps"]),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", False)),
            bos_token_id=int(data["bos_token_id"]),
            eos_token_id=int(data["eos_token_id"]),
            attention_bias=bool(data.get("attention_bias", True)),
        )
        config.validate()
        return config

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> Self:
        """从模型目录中的 config.json 读取配置。"""
        config_path = Path(model_dir) / "config.json"
        with config_path.open(encoding="utf-8") as file:
            data: dict[str, Any] = json.load(file)
        return cls.from_dict(data)

    def validate(self) -> None:
        """检查 GQA reshape 所依赖的整除关系。"""
        positive_fields = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} 必须大于零")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size 必须能被 num_attention_heads 整除")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads 必须能被 num_key_value_heads 整除")
