"""数值比较工具测试。"""

import pytest
import torch

from plaininfer.comparison import error_stats


def test_error_stats() -> None:
    stats = error_stats(torch.tensor([1.0, 3.0]), torch.tensor([1.0, 2.0]))
    assert stats.max_absolute == 1.0
    assert stats.mean_absolute == 0.5
    assert stats.max_relative == 0.5


def test_error_stats_rejects_shape() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        error_stats(torch.ones(1), torch.ones(2))
