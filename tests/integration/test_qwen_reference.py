"""Qwen2.5-0.5B 的可选端到端与 reference 对齐测试。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from plaininfer.generation import generate_greedy_full, generate_ids
from plaininfer.model import PlainInferModel

MODEL_DIR_TEXT = os.environ.get("PLAININFER_MODEL_DIR")
pytestmark = pytest.mark.skipif(MODEL_DIR_TEXT is None, reason="未设置 PLAININFER_MODEL_DIR")


@pytest.fixture(scope="module")
def real_models() -> tuple[PlainInferModel, Any, Any]:
    if MODEL_DIR_TEXT is None:
        pytest.skip("未设置 PLAININFER_MODEL_DIR")
    model_dir = Path(MODEL_DIR_TEXT)
    tokenizer: Any = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    plain = PlainInferModel.from_pretrained(model_dir, dtype=torch.float32)
    reference: Any = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, local_files_only=True, attn_implementation="eager"
    )
    reference.eval()
    return plain, reference, tokenizer


@pytest.mark.parametrize("prompt", ["Hello, world!", "你好世界", "Transformer inference from scratch."])
def test_real_logits_match_hugging_face(real_models: tuple[PlainInferModel, Any, Any], prompt: str) -> None:
    plain, reference, tokenizer = real_models
    input_ids = cast(torch.Tensor, tokenizer(prompt, return_tensors="pt").input_ids)
    with torch.no_grad():
        actual = plain(input_ids)
        expected = cast(torch.Tensor, reference(input_ids).logits)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_real_longer_positions_match_hugging_face(real_models: tuple[PlainInferModel, Any, Any]) -> None:
    plain, reference, _tokenizer = real_models
    input_ids = (torch.arange(64) % plain.config.vocab_size).unsqueeze(0)
    with torch.no_grad():
        actual = plain(input_ids)
        expected = cast(torch.Tensor, reference(input_ids).logits)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)


def test_real_cached_decode_matches_full_forward(real_models: tuple[PlainInferModel, Any, Any]) -> None:
    plain, _reference, tokenizer = real_models
    input_ids = cast(torch.Tensor, tokenizer("Hello, world!", return_tensors="pt").input_ids)
    with torch.no_grad():
        _, cache = plain.prefill(input_ids)
        sequence = input_ids
        logits = plain(input_ids)
        for _ in range(4):
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            sequence = torch.cat((sequence, next_token), dim=1)
            cached = plain.decode(next_token, cache)
            full = plain(sequence)
            torch.testing.assert_close(cached[:, -1], full[:, -1], atol=3e-5, rtol=1e-5)
            logits = cached


def test_real_cached_generation_matches_baseline(real_models: tuple[PlainInferModel, Any, Any]) -> None:
    plain, _reference, tokenizer = real_models
    input_ids = cast(torch.Tensor, tokenizer("Hello, world!", return_tensors="pt").input_ids)
    full = generate_greedy_full(plain, input_ids, 3)
    cached = generate_ids(plain, input_ids, 3)
    assert torch.equal(cached, full)


def test_real_sampling_is_reproducible(real_models: tuple[PlainInferModel, Any, Any]) -> None:
    plain, _reference, tokenizer = real_models
    input_ids = cast(torch.Tensor, tokenizer("你好", return_tensors="pt").input_ids)
    first = generate_ids(
        plain,
        input_ids,
        max_new_tokens=4,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        generator=torch.Generator().manual_seed(42),
    )
    second = generate_ids(
        plain,
        input_ids,
        max_new_tokens=4,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        generator=torch.Generator().manual_seed(42),
    )
    assert torch.equal(first, second)
