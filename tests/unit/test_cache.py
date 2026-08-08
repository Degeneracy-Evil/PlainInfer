"""KV Cache 与 cached decode 测试。"""

import pytest
import torch

from plaininfer.cache import KVCache, LayerKVCache
from plaininfer.model import PlainInferModel
from tests.unit.test_layers import tiny_config


def test_cache_growth_and_shape() -> None:
    cache = KVCache(2)
    first = torch.zeros(1, 2, 3, 4)
    cache.update(0, first, first)
    cache.update(1, first, first)
    assert cache.sequence_length == 3
    new = torch.ones(1, 2, 1, 4)
    updated = cache.update(0, new, new)
    assert updated.key.shape == (1, 2, 4, 4)


def test_cache_validation() -> None:
    with pytest.raises(ValueError, match="大于零"):
        KVCache(0)
    cache = KVCache(1)
    with pytest.raises(ValueError, match="shape"):
        cache.update(0, torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 2, 2))
    with pytest.raises(ValueError, match=r"\[B, H_kv, S, head_dim\]"):
        cache.update(0, torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))
    with pytest.raises(ValueError, match="device/dtype"):
        cache.update(0, torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2, dtype=torch.int64))
    with pytest.raises(IndexError, match="越界"):
        cache.update(2, torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2))
    cache.update(0, torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2))
    with pytest.raises(ValueError, match="sequence"):
        cache.update(0, torch.zeros(1, 2, 1, 2), torch.zeros(1, 2, 1, 2))


def test_cache_rejects_inconsistent_layers() -> None:
    cache = KVCache(2)
    cache.layers[0] = LayerKVCache(torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2))
    cache.layers[1] = LayerKVCache(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2))
    with pytest.raises(RuntimeError, match="不一致"):
        _ = cache.sequence_length


def test_cache_rejects_partially_filled_layers() -> None:
    cache = KVCache(2)
    cache.update(0, torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2))
    with pytest.raises(RuntimeError, match="部分"):
        _ = cache.sequence_length


def test_prefill_matches_full_forward() -> None:
    torch.manual_seed(1)
    model = PlainInferModel(tiny_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    full = model(input_ids)
    prefill, cache = model.prefill(input_ids)
    assert prefill.shape == (1, 1, 32)
    torch.testing.assert_close(prefill[:, -1], full[:, -1])
    assert cache.sequence_length == 3
    for layer in cache.layers:
        assert layer is not None
        assert layer.key.shape == (1, 2, 3, 2)


def test_multiple_cached_decode_matches_full_forward() -> None:
    torch.manual_seed(2)
    model = PlainInferModel(tiny_config()).eval()
    sequence = torch.tensor([[1, 2]])
    _, cache = model.prefill(sequence)
    for token in (3, 4, 5):
        sequence = torch.cat((sequence, torch.tensor([[token]])), dim=1)
        cached_logits = model.decode(torch.tensor([[token]]), cache)
        full_logits = model(sequence)
        torch.testing.assert_close(cached_logits[:, -1], full_logits[:, -1], atol=1e-5, rtol=1e-5)
        assert cache.sequence_length == sequence.shape[1]


def test_decode_requires_one_token() -> None:
    model = PlainInferModel(tiny_config())
    _, cache = model.prefill(torch.tensor([[1]]))
    with pytest.raises(ValueError, match="一个 token"):
        model.decode(torch.tensor([[2, 3]]), cache)


def test_decode_validates_cache_and_context() -> None:
    model = PlainInferModel(tiny_config())
    with pytest.raises(ValueError, match="prefill"):
        model.decode(torch.tensor([[1]]), KVCache(2))
    with pytest.raises(ValueError, match="层数"):
        model.decode(torch.tensor([[1]]), KVCache(1))

    wrong_shape = KVCache(2)
    for layer_index in range(2):
        wrong_shape.update(layer_index, torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2))
    with pytest.raises(ValueError, match="shape"):
        model.decode(torch.tensor([[1]]), wrong_shape)

    _, cache = model.prefill(torch.ones(1, 128, dtype=torch.long))
    with pytest.raises(ValueError, match="max_position_embeddings"):
        model.decode(torch.tensor([[1]]), cache)
