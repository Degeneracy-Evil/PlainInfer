"""与 reference tensor 比较时使用的小型数值工具。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ErrorStats:
    """两个 tensor 的绝对和相对误差摘要。"""

    max_absolute: float
    mean_absolute: float
    max_relative: float


def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> ErrorStats:
    """计算误差; relative error 的分母设置微小下限以避开除零。"""
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}")
    absolute = (actual.to(torch.float32) - expected.to(torch.float32)).abs()
    relative = absolute / expected.to(torch.float32).abs().clamp_min(1e-12)
    return ErrorStats(absolute.max().item(), absolute.mean().item(), relative.max().item())
