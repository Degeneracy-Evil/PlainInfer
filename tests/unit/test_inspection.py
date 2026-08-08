"""Checkpoint 元数据检查测试。"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from plaininfer.config import ModelConfig
from plaininfer.inspection import find_checkpoint_files, format_report, inspect_checkpoint, main
from tests.unit.test_config import config_data


def make_model_dir(path: Path) -> Path:
    (path / "config.json").write_text(json.dumps(config_data()), encoding="utf-8")
    save_file({"model.embed_tokens.weight": torch.zeros(32, 8)}, path / "model.safetensors")
    return path


def test_inspect_and_format(tmp_path: Path) -> None:
    model_dir = make_model_dir(tmp_path)
    tensors = inspect_checkpoint(model_dir)
    assert tensors == {"model.embed_tokens.weight": (32, 8)}
    report = format_report(ModelConfig.from_model_dir(model_dir), tensors)
    assert "head_dim: 2" in report
    assert "model.embed_tokens.weight: (32, 8)" in report


def test_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_dir = make_model_dir(tmp_path)
    assert main([str(model_dir)]) == 0
    assert "tensor_count: 1" in capsys.readouterr().out


def test_main_without_model_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "model_dir" in capsys.readouterr().out


def test_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="safetensors"):
        find_checkpoint_files(tmp_path)


def test_duplicate_tensor_across_shards(tmp_path: Path) -> None:
    tensor = {"duplicate": torch.zeros(1)}
    save_file(tensor, tmp_path / "a.safetensors")
    save_file(tensor, tmp_path / "b.safetensors")
    with pytest.raises(ValueError, match="duplicate"):
        inspect_checkpoint(tmp_path)
