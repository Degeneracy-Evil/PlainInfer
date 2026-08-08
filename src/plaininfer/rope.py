"""Rotary Position Embedding。"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """根据绝对 position 生成 RoPE 的 cos 和 sin。"""

    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        # inv_freq: [head_dim / 2]
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [B, S], frequencies: [B, S, head_dim / 2]
        inv_freq = cast(torch.Tensor, self.inv_freq)
        frequencies = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos().to(dtype), embeddings.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把向量两半从 [x1, x2] 旋转成 [-x2, x1]。"""
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """将 [B, S, D] 的 cos/sin 广播到 attention heads。"""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)
