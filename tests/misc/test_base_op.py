"""BaseOP must be callable (the nn.Module convention the model code uses:
``layer(x)``, not ``layer.forward(x)``).

Regression: the Qwen3.8 engine's first request crashed with
``TypeError: 'Q6KLinear' object is not callable`` because the model code
calls BaseOP layers as callables (GenInfer-Qwen3.8 boot gate, 2026-09-24).
"""
import torch

from minisgl.layers.base import BaseOP


class _DummyOP(BaseOP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


def test_base_op_is_callable():
    op = _DummyOP()
    x = torch.zeros(2, 3)
    assert torch.equal(op(x), op.forward(x))
    assert torch.equal(op(x), x + 1)
