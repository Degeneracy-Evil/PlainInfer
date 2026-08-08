"""Qwen2.5 的基础计算组件。"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

from plaininfer.cache import KVCache
from plaininfer.config import ModelConfig
from plaininfer.rope import RotaryEmbedding, apply_rotary_pos_emb


class RMSNorm(nn.Module):
    """RMSNorm(x) = weight * x / sqrt(mean(x²) + eps)。"""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.to(torch.float32)
        variance = values.square().mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.eps)
        return (self.weight * normalized).to(input_dtype)


def repeat_kv(hidden_states: torch.Tensor, num_groups: int) -> torch.Tensor:
    """将每个 KV head 显式共享给 num_groups 个 query heads。"""
    if num_groups == 1:
        return hidden_states
    # [B, H_kv, S, D] -> [B, H_kv, G, S, D] -> [B, H_q, S, D]
    batch, kv_heads, sequence, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(batch, kv_heads, num_groups, sequence, head_dim)
    return expanded.reshape(batch, kv_heads * num_groups, sequence, head_dim)


def causal_mask(query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
    """创建 bool mask; True 表示该 query 可以看到该 key。"""
    query_positions = torch.arange(key_length - query_length, key_length, device=device).unsqueeze(-1)
    key_positions = torch.arange(key_length, device=device).unsqueeze(0)
    return key_positions <= query_positions


class SelfAttention(nn.Module):
    """显式实现 Qwen2.5 Grouped Query Attention。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_key_value_groups
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """不使用 cache 的 full-sequence attention。"""
        return self._forward(hidden_states, position_ids, None, None)

    def forward_cached(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor, cache: KVCache, layer_index: int
    ) -> torch.Tensor:
        """只计算新 token 的 Q/K/V, 并将 K/V 追加到 cache。"""
        return self._forward(hidden_states, position_ids, cache, layer_index)

    def _forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache: KVCache | None,
        layer_index: int | None,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        # projection 后先拆 head, 再转为 [B, H, S, D]。
        query = self.q_proj(hidden_states).view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(position_ids, query.dtype)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if cache is not None:
            if layer_index is None:
                raise ValueError("cached attention 需要 layer_index")
            layer_cache = cache.update(layer_index, key, value)
            key, value = layer_cache.key, layer_cache.value
        key = repeat_kv(key, self.num_kv_groups)
        value = repeat_kv(value, self.num_kv_groups)

        # scores: [B, H_q, S_q, S_k]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        key_length = key.shape[2]
        mask = causal_mask(sequence, key_length, hidden_states.device)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(probabilities, value)
        # [B, H, S, D] -> [B, S, H * D]
        output = output.transpose(1, 2).contiguous().view(batch, sequence, self.num_heads * self.head_dim)
        return cast(torch.Tensor, self.o_proj(output))


class MLP(nn.Module):
    """Qwen2.5 SwiGLU: down(SiLU(gate(x)) * up(x))。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return cast(torch.Tensor, self.down_proj(gate * up))


class DecoderLayer(nn.Module):
    """Pre-norm Qwen2.5 decoder layer, 包含两条 residual path。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = SelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, S, hidden_size]
        return self._forward(hidden_states, position_ids, None, None)

    def forward_cached(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache: KVCache,
        layer_index: int,
    ) -> torch.Tensor:
        """执行使用该层 KV Cache 的 decoder forward。"""
        return self._forward(hidden_states, position_ids, cache, layer_index)

    def _forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache: KVCache | None,
        layer_index: int | None,
    ) -> torch.Tensor:
        """共享 full forward 与 cached forward 的 norm 和 residual 路径。"""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if cache is None:
            attention_output = self.self_attn(hidden_states, position_ids)
        else:
            if layer_index is None:
                raise ValueError("cached decoder layer 需要 layer_index")
            attention_output = self.self_attn.forward_cached(hidden_states, position_ids, cache, layer_index)
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return cast(torch.Tensor, residual + self.mlp(hidden_states))
