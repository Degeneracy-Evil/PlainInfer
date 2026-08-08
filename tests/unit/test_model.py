"""完整模型 forward 测试。"""

import pytest
import torch

from plaininfer.model import PlainInferModel
from tests.unit.test_layers import tiny_config


def test_model_forward_shape_and_tied_head() -> None:
    model = PlainInferModel(tiny_config())
    logits = model(torch.tensor([[1, 2, 3]]))
    assert logits.shape == (1, 3, 32)
    assert model.lm_head is None


def test_model_with_untied_head() -> None:
    config = tiny_config()
    object.__setattr__(config, "tie_word_embeddings", False)
    model = PlainInferModel(config)
    assert model.lm_head is not None
    assert model(torch.tensor([[1]])).shape == (1, 1, 32)


@pytest.mark.parametrize("input_ids", [torch.tensor([1, 2]), torch.ones(2, 1, dtype=torch.long)])
def test_model_rejects_unsupported_batch(input_ids: torch.Tensor) -> None:
    with pytest.raises(ValueError, match=r"\[1, sequence\]"):
        PlainInferModel(tiny_config())(input_ids)


def test_model_rejects_empty_and_too_long_sequence() -> None:
    model = PlainInferModel(tiny_config())
    with pytest.raises(ValueError, match="非空"):
        model(torch.empty(1, 0, dtype=torch.long))
    with pytest.raises(ValueError, match="max_position_embeddings"):
        model(torch.ones(1, 129, dtype=torch.long))
