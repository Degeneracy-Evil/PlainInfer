"""PlainInfer 的自回归生成循环。"""

from __future__ import annotations

from typing import Protocol, cast

import torch
from torch import nn

from plaininfer.model import PlainInferModel


class TokenizerOutput(Protocol):
    """Tokenizer 返回值中 PlainInfer 使用的最小部分。"""

    input_ids: torch.Tensor


class Tokenizer(Protocol):
    """与 Hugging Face tokenizer 兼容的最小接口。"""

    eos_token_id: int | None

    def __call__(self, text: str, *, return_tensors: str) -> TokenizerOutput: ...

    def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str: ...


def generate_greedy_full(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """每生成一个 token 都重新计算完整 sequence, 作为 KV Cache 的 baseline。"""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids 必须是 [1, sequence]")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens 不能为负数")

    generated = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # logits: [1, current_sequence, vocab_size]
            logits = cast(torch.Tensor, model(generated))
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
    return generated


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """从单个 token 的 [1, vocab_size] logits 透明地执行 greedy 或 sampling。"""
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("logits 必须是 [1, vocab_size]")
    if temperature < 0:
        raise ValueError("temperature 不能为负数")
    if top_k is not None and not 0 < top_k <= logits.shape[-1]:
        raise ValueError("top_k 必须在 [1, vocab_size] 内")
    if not 0 < top_p <= 1:
        raise ValueError("top_p 必须在 (0, 1] 内")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    scores = logits.to(torch.float32) / temperature
    if top_k is not None:
        top_values, top_indices = torch.topk(scores, top_k, dim=-1)
        filtered = torch.full_like(scores, -torch.inf)
        scores = filtered.scatter(1, top_indices, top_values)
    if top_p < 1:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        sorted_probabilities = torch.softmax(sorted_scores, dim=-1)
        cumulative = sorted_probabilities.cumsum(dim=-1)
        # 当前 token 之前的概率质量已经达到 top_p 时, 移除当前及之后的 token。
        remove_sorted = cumulative - sorted_probabilities > top_p
        remove = torch.zeros_like(remove_sorted).scatter(1, sorted_indices, remove_sorted)
        scores = scores.masked_fill(remove, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


def generate_ids(
    model: PlainInferModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """使用 prefill + KV Cache decode 生成 token ids。"""
    model.validate_input_ids(input_ids)
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens 不能为负数")
    if input_ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt + max_new_tokens 超过 max_position_embeddings")
    if max_new_tokens == 0:
        return input_ids.clone()

    generated = input_ids.clone()
    with torch.no_grad():
        logits, cache = model.prefill(input_ids)
        for index in range(max_new_tokens):
            next_token = sample_next_token(logits[:, -1, :], temperature, top_k, top_p, generator)
            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            if index + 1 < max_new_tokens:
                logits = model.decode(next_token, cache)
    return generated


def generate(
    model: PlainInferModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> str:
    """字符串输入、字符串输出的最小 Python generation API。"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.embed_tokens.weight.device)
    output_ids = generate_ids(
        model,
        input_ids,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        tokenizer.eos_token_id,
        generator,
    )
    return tokenizer.decode(output_ids[0].cpu(), skip_special_tokens=False)
