"""无 KV Cache generation 测试。"""

import pytest
import torch
from torch import nn

from plaininfer.generation import generate, generate_greedy_full, generate_ids, sample_next_token
from plaininfer.model import PlainInferModel
from tests.unit.test_layers import tiny_config


class IncrementModel(nn.Module):
    """令 next token 恒为当前 token + 1 的确定性测试模型。"""

    def __init__(self) -> None:
        super().__init__()
        self.seen_lengths: list[int] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.seen_lengths.append(input_ids.shape[1])
        logits = torch.zeros((*input_ids.shape, 8))
        next_ids = (input_ids + 1).clamp_max(7).unsqueeze(-1)
        return logits.scatter(-1, next_ids, 1.0)


def test_full_generation_reruns_growing_sequence() -> None:
    model = IncrementModel()
    result = generate_greedy_full(model, torch.tensor([[1, 2]]), 3)
    assert result.tolist() == [[1, 2, 3, 4, 5]]
    assert model.seen_lengths == [2, 3, 4]


def test_full_generation_stops_at_eos() -> None:
    result = generate_greedy_full(IncrementModel(), torch.tensor([[1]]), 5, eos_token_id=3)
    assert result.tolist() == [[1, 2, 3]]


def test_full_generation_zero_tokens() -> None:
    input_ids = torch.tensor([[1]])
    assert torch.equal(generate_greedy_full(IncrementModel(), input_ids, 0), input_ids)


@pytest.mark.parametrize(
    "input_ids",
    [torch.tensor([1]), torch.ones(2, 1, dtype=torch.long)],
)
def test_full_generation_rejects_batch(input_ids: torch.Tensor) -> None:
    with pytest.raises(ValueError, match=r"\[1, sequence\]"):
        generate_greedy_full(IncrementModel(), input_ids, 1)


def test_full_generation_rejects_negative_length() -> None:
    with pytest.raises(ValueError, match="负数"):
        generate_greedy_full(IncrementModel(), torch.tensor([[1]]), -1)


def test_sample_greedy_and_seeded_sampling() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    assert sample_next_token(logits).item() == 2
    first = sample_next_token(logits, temperature=1.0, generator=torch.Generator().manual_seed(7))
    second = sample_next_token(logits, temperature=1.0, generator=torch.Generator().manual_seed(7))
    assert torch.equal(first, second)


def test_top_k_and_top_p_filtering() -> None:
    logits = torch.tensor([[100.0, 10.0, 0.0]])
    for _ in range(5):
        assert sample_next_token(logits, temperature=1.0, top_k=1).item() == 0
        assert sample_next_token(logits, temperature=1.0, top_p=0.1).item() == 0


def test_top_k_keeps_exactly_k_candidates() -> None:
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    sampled = {
        sample_next_token(logits, temperature=10.0, top_k=2, generator=torch.Generator().manual_seed(seed)).item()
        for seed in range(20)
    }
    assert sampled <= {0, 1}


def test_sampling_rejects_invalid_options() -> None:
    logits = torch.ones(1, 3)
    with pytest.raises(ValueError, match="temperature"):
        sample_next_token(logits, temperature=-1)
    with pytest.raises(ValueError, match="top_k"):
        sample_next_token(logits, top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        sample_next_token(logits, top_k=4)
    with pytest.raises(ValueError, match="top_p"):
        sample_next_token(logits, top_p=0)
    with pytest.raises(ValueError, match="top_p"):
        sample_next_token(logits, top_p=1.1)


def test_sampling_rejects_shape() -> None:
    with pytest.raises(ValueError, match=r"\[1, vocab_size\]"):
        sample_next_token(torch.ones(3))


def test_cached_generate_ids() -> None:
    torch.manual_seed(3)
    model = PlainInferModel(tiny_config()).eval()
    prompt = torch.tensor([[1, 2]])
    assert torch.equal(generate_ids(model, prompt, 0), prompt)
    output = generate_ids(model, prompt, 3)
    assert output.shape == (1, 5)


def test_generate_ids_validates_even_when_no_output() -> None:
    model = PlainInferModel(tiny_config())
    with pytest.raises(ValueError, match="非空"):
        generate_ids(model, torch.empty(1, 0, dtype=torch.long), 0)
    with pytest.raises(ValueError, match="max_position_embeddings"):
        generate_ids(model, torch.ones(1, 127, dtype=torch.long), 2)


def test_generate_ids_rejects_negative_length() -> None:
    with pytest.raises(ValueError, match="负数"):
        generate_ids(PlainInferModel(tiny_config()), torch.tensor([[1]]), -1)


class FakeTokenizerOutput:
    input_ids = torch.tensor([[1, 2]])


class FakeTokenizer:
    eos_token_id: int | None = None

    def __call__(self, text: str, *, return_tensors: str) -> FakeTokenizerOutput:
        assert text == "prompt"
        assert return_tensors == "pt"
        return FakeTokenizerOutput()

    def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return " ".join(str(token) for token in token_ids.tolist())


def test_string_generation_api() -> None:
    model = PlainInferModel(tiny_config()).eval()
    text = generate(model, FakeTokenizer(), "prompt", max_new_tokens=1)
    assert len(text.split()) == 3
