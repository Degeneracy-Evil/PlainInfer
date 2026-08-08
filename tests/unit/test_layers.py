"""基础数学组件测试。"""

from dataclasses import replace

import torch

from plaininfer.config import ModelConfig
from plaininfer.layers import MLP, DecoderLayer, RMSNorm, SelfAttention, causal_mask, repeat_kv
from plaininfer.rope import RotaryEmbedding, apply_rotary_pos_emb, rotate_half
from tests.unit.test_config import config_data


def tiny_config() -> ModelConfig:
    return ModelConfig.from_dict(config_data())


def test_rms_norm_formula() -> None:
    layer = RMSNorm(2, 1e-6)
    x = torch.tensor([[[3.0, 4.0]]])
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(layer(x), expected)


def test_rope_and_rotate_half() -> None:
    rope = RotaryEmbedding(2, 10_000.0)
    cos, sin = rope(torch.tensor([[0, 1]]), torch.float32)
    assert cos.shape == (1, 2, 2)
    x = torch.tensor([1.0, 2.0])
    assert torch.equal(rotate_half(x), torch.tensor([-2.0, 1.0]))
    q = torch.ones(1, 1, 2, 2)
    rotated_q, rotated_k = apply_rotary_pos_emb(q, q, cos, sin)
    torch.testing.assert_close(rotated_q, rotated_k)


def test_repeat_kv() -> None:
    x = torch.arange(4).reshape(1, 2, 1, 2)
    repeated = repeat_kv(x, 2)
    assert repeated.shape == (1, 4, 1, 2)
    assert torch.equal(repeat_kv(x, 1), x)
    assert torch.equal(repeated[:, 0], repeated[:, 1])


def test_causal_mask_with_offset() -> None:
    expected = torch.tensor([[True, True, True], [True, True, True]])
    assert torch.equal(causal_mask(2, 3, torch.device("cpu")), expected.tril(diagonal=1))


def test_attention_is_causal() -> None:
    torch.manual_seed(0)
    attention = SelfAttention(tiny_config())
    x = torch.randn(1, 3, 8)
    positions = torch.arange(3).unsqueeze(0)
    original = attention(x, positions)
    changed = x.clone()
    changed[:, 2] += 100
    torch.testing.assert_close(original[:, :2], attention(changed, positions)[:, :2])


def test_attention_respects_bias_config() -> None:
    attention = SelfAttention(replace(tiny_config(), attention_bias=False))
    biases = (attention.q_proj.bias, attention.k_proj.bias, attention.v_proj.bias)
    assert all(bias is None for bias in biases)


def test_mlp_shape() -> None:
    assert MLP(tiny_config())(torch.randn(1, 3, 8)).shape == (1, 3, 8)


def test_decoder_layer_residual_path() -> None:
    layer = DecoderLayer(tiny_config())
    for name, parameter in layer.named_parameters():
        if "layernorm" not in name:
            parameter.data.zero_()
    hidden_states = torch.randn(1, 3, 8)
    positions = torch.arange(3).unsqueeze(0)
    torch.testing.assert_close(layer(hidden_states, positions), hidden_states)
