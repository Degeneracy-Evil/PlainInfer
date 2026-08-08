"""Checkpoint loader 测试。"""

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from plaininfer.checkpoint import load_checkpoint, load_checkpoint_from_dir, parameter_name_map, read_checkpoint


def test_parameter_name_map() -> None:
    mapping = parameter_name_map(1, tie_word_embeddings=True)
    assert mapping["model.layers.0.self_attn.q_proj.weight"] == "layers.0.self_attn.q_proj.weight"
    assert "lm_head.weight" not in mapping
    assert parameter_name_map(0, tie_word_embeddings=False)["lm_head.weight"] == "lm_head.weight"
    without_bias = parameter_name_map(1, tie_word_embeddings=True, attention_bias=False)
    assert "model.layers.0.self_attn.q_proj.bias" not in without_bias


def test_read_and_load_checkpoint(tmp_path: Path) -> None:
    save_file({"source.weight": torch.full((2, 3), 2.0)}, tmp_path / "model.safetensors")
    module = nn.Linear(3, 2, bias=False)
    load_checkpoint(module, read_checkpoint(tmp_path), {"source.weight": "weight"})
    assert torch.equal(module.weight, torch.full((2, 3), 2.0))


def test_stream_checkpoint_from_dir(tmp_path: Path) -> None:
    save_file({"source.weight": torch.full((2, 3), 2.0)}, tmp_path / "model.safetensors")
    module = nn.Linear(3, 2, bias=False)
    load_checkpoint_from_dir(module, tmp_path, {"source.weight": "weight"})
    assert torch.equal(module.weight, torch.full((2, 3), 2.0))


def test_reject_checkpoint_keys() -> None:
    module = nn.Linear(1, 1, bias=False)
    with pytest.raises(KeyError, match="missing keys"):
        load_checkpoint(module, {"other": torch.ones(1, 1)}, {"source": "weight"})


def test_reject_module_keys() -> None:
    module = nn.Linear(1, 1)
    with pytest.raises(KeyError, match="module unexpected keys"):
        load_checkpoint(module, {"source": torch.ones(1, 1)}, {"source": "weight"})


def test_reject_shape() -> None:
    module = nn.Linear(1, 1, bias=False)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_checkpoint(module, {"source": torch.ones(2, 1)}, {"source": "weight"})
