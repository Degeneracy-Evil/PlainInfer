"""PlainInfer 的完整 Qwen2.5 causal language model。"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from torch import nn

from plaininfer.cache import KVCache
from plaininfer.checkpoint import load_checkpoint_from_dir, parameter_name_map
from plaininfer.config import ModelConfig
from plaininfer.layers import DecoderLayer, RMSNorm


class PlainInferModel(nn.Module):
    """第一版只支持 batch_size=1 的 Qwen2.5 模型。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = (
            None if config.tie_word_embeddings else nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        )

    @classmethod
    def from_pretrained(
        cls, model_dir: str | Path, *, dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu"
    ) -> PlainInferModel:
        """创建自有模型并从 safetensors 显式加载权重。"""
        config = ModelConfig.from_model_dir(model_dir)
        model = cls(config).to(device=device, dtype=dtype)
        mapping = parameter_name_map(
            config.num_hidden_layers,
            config.tie_word_embeddings,
            config.attention_bias,
        )
        load_checkpoint_from_dir(model, model_dir, mapping)
        model.eval()
        return model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """执行不使用 KV Cache 的 full-sequence forward。"""
        self.validate_input_ids(input_ids)
        _, sequence = input_ids.shape
        position_ids = torch.arange(sequence, device=input_ids.device).unsqueeze(0)
        # hidden_states: [1, S, hidden_size]
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)
        hidden_states = self.norm(hidden_states)
        return self._compute_logits(hidden_states)

    def prefill(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, KVCache]:
        """一次处理整个 prompt, 并为每层保存全部 K/V。"""
        self.validate_input_ids(input_ids)
        cache = KVCache(self.config.num_hidden_layers)
        return self._forward_cached(input_ids, cache, position_start=0, last_token_only=True), cache

    def decode(self, input_ids: torch.Tensor, cache: KVCache) -> torch.Tensor:
        """只输入一个新 token, 并 attention 到 cache 中的所有历史 K/V。"""
        self.validate_input_ids(input_ids)
        if input_ids.shape[1] != 1:
            raise ValueError("decode 每次只接受一个 token")
        position_start = cache.validate_for_decode(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            device=self.embed_tokens.weight.device,
            dtype=self.embed_tokens.weight.dtype,
        )
        if position_start >= self.config.max_position_embeddings:
            raise ValueError("decode 将超过 max_position_embeddings")
        return self._forward_cached(input_ids, cache, position_start=position_start, last_token_only=False)

    def validate_input_ids(self, input_ids: torch.Tensor) -> None:
        """检查第一版 batch、sequence、context 和 device 约束。"""
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("input_ids 必须是非空的 [1, sequence]")
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("sequence length 超过 max_position_embeddings")
        if input_ids.device != self.embed_tokens.weight.device:
            raise ValueError("input_ids 和 model 必须位于同一 device")

    def _forward_cached(
        self,
        input_ids: torch.Tensor,
        cache: KVCache,
        position_start: int,
        *,
        last_token_only: bool,
    ) -> torch.Tensor:
        sequence = input_ids.shape[1]
        position_ids = torch.arange(position_start, position_start + sequence, device=input_ids.device).unsqueeze(0)
        hidden_states = self.embed_tokens(input_ids)
        for layer_index, layer in enumerate(self.layers):
            decoder_layer = cast(DecoderLayer, layer)
            hidden_states = decoder_layer.forward_cached(hidden_states, position_ids, cache, layer_index)
        hidden_states = self.norm(hidden_states)
        if last_token_only:
            # generation 的 prefill 只需要最后一个 prompt token 的 logits。
            hidden_states = hidden_states[:, -1:, :]
        return self._compute_logits(hidden_states)

    def _compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """应用 tied embedding 或独立 LM Head。"""
        if self.lm_head is None:
            return torch.nn.functional.linear(hidden_states, self.embed_tokens.weight)
        return cast(torch.Tensor, self.lm_head(hidden_states))
