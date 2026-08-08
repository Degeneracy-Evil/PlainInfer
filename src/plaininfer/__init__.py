"""PlainInfer: 用于学习 decoder-only Transformer 推理的朴素实现。"""

from plaininfer.config import ModelConfig
from plaininfer.generation import generate, generate_ids
from plaininfer.model import PlainInferModel

__all__ = ["ModelConfig", "PlainInferModel", "generate", "generate_ids"]
