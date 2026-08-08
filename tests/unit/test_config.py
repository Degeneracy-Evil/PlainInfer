"""模型配置测试。"""

import json
from pathlib import Path

import pytest

from plaininfer.config import ModelConfig


def config_data() -> dict[str, object]:
    return {
        "model_type": "qwen2",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "rope_theta": 1_000_000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }


def test_load_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(config_data()), encoding="utf-8")
    config = ModelConfig.from_model_dir(tmp_path)
    assert config.head_dim == 2
    assert config.num_key_value_groups == 2
    assert config.attention_bias is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hidden_size", 7, "hidden_size"),
        ("num_key_value_heads", 3, "num_attention_heads"),
    ],
)
def test_reject_invalid_head_layout(field: str, value: int, message: str) -> None:
    data = config_data()
    data[field] = value
    with pytest.raises(ValueError, match=message):
        ModelConfig.from_dict(data)


def test_reject_other_architecture() -> None:
    data = config_data()
    data["model_type"] = "llama"
    with pytest.raises(ValueError, match="qwen2"):
        ModelConfig.from_dict(data)


def test_reject_non_positive_dimensions() -> None:
    data = config_data()
    data["num_attention_heads"] = 0
    with pytest.raises(ValueError, match="大于零"):
        ModelConfig.from_dict(data)
