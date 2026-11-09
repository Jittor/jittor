"""Typed tensor constructors return non-trainable leaves, as torch does."""

import pytest
import torch


@pytest.mark.parametrize("ctor", [torch.FloatTensor, torch.DoubleTensor, torch.IntTensor, torch.LongTensor])
def test_typed_constructor_returns_leaf_without_grad(ctor):
    tensor = ctor(2, 3)
    assert tensor.is_leaf
    assert not tensor.requires_grad
    # A no-op requires_grad_(False) on a leaf must not raise.
    tensor.requires_grad_(False)


def test_typed_constructor_with_shape_is_zero_initialized():
    tensor = torch.FloatTensor(4)
    assert tuple(tensor.shape) == (4,)
    assert float(tensor.abs().sum()) == 0.0
