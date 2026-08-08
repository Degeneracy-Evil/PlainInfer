"""每层 KV Cache 的直观数据结构。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LayerKVCache:
    """key/value shape 均为 [B, H_kv, cached_sequence, head_dim]。"""

    key: torch.Tensor
    value: torch.Tensor


class KVCache:
    """按 decoder layer index 保存 K/V。"""

    def __init__(self, num_layers: int) -> None:
        if num_layers <= 0:
            raise ValueError("num_layers 必须大于零")
        self.layers: list[LayerKVCache | None] = [None] * num_layers

    @property
    def sequence_length(self) -> int:
        """所有已写入层必须具有相同 cache 长度。"""
        filled_layers = [layer for layer in self.layers if layer is not None]
        if not filled_layers:
            return 0
        if len(filled_layers) != len(self.layers):
            raise RuntimeError("KV Cache 只填充了部分 decoder layers")
        lengths = {layer.key.shape[2] for layer in filled_layers}
        if len(lengths) != 1:
            raise RuntimeError(f"各层 KV Cache 长度不一致: {sorted(lengths)}")
        return lengths.pop()

    def update(self, layer_index: int, key: torch.Tensor, value: torch.Tensor) -> LayerKVCache:
        """把当前 token/chunk 的 K/V 追加到指定层。"""
        if not 0 <= layer_index < len(self.layers):
            raise IndexError(f"layer_index 越界: {layer_index}")
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("key 和 value 必须是 [B, H_kv, S, head_dim]")
        if key.shape != value.shape:
            raise ValueError("key 和 value shape 必须一致")
        if key.device != value.device or key.dtype != value.dtype:
            raise ValueError("key 和 value 的 device/dtype 必须一致")
        current = self.layers[layer_index]
        if current is None:
            updated = LayerKVCache(key, value)
        else:
            if current.key.shape[:2] + current.key.shape[3:] != key.shape[:2] + key.shape[3:]:
                raise ValueError("追加的 K/V 除 sequence 维外必须与 cache shape 一致")
            updated = LayerKVCache(torch.cat((current.key, key), dim=2), torch.cat((current.value, value), dim=2))
        self.layers[layer_index] = updated
        return updated

    def validate_for_decode(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> int:
        """在 decode 前检查 cache 是否完整且与当前模型一致。"""
        if len(self.layers) != num_layers:
            raise ValueError(f"KV Cache 层数错误: {len(self.layers)} != {num_layers}")
        sequence_length = self.sequence_length
        if sequence_length == 0:
            raise ValueError("decode 需要先执行 prefill")
        expected_prefix = (1, num_kv_heads)
        for layer_index, layer in enumerate(self.layers):
            if layer is None:  # sequence_length 已检查, 此分支只帮助类型收窄。
                raise RuntimeError("KV Cache 只填充了部分 decoder layers")
            expected_shape = (*expected_prefix, sequence_length, head_dim)
            if layer.key.shape != expected_shape or layer.value.shape != expected_shape:
                raise ValueError(f"第 {layer_index} 层 KV Cache shape 错误")
            if layer.key.device != device or layer.value.device != device:
                raise ValueError(f"第 {layer_index} 层 KV Cache device 错误")
            if layer.key.dtype != dtype or layer.value.dtype != dtype:
                raise ValueError(f"第 {layer_index} 层 KV Cache dtype 错误")
        return sequence_length
