"""Stacking helpers apply Torch's atleast_Nd rules to 0-dim tensors.

``torch.hstack([tensor(1.), tensor(2.)])`` is ``[1., 2.]``; it used to raise
because 0-dim operands reached a dim=1 concat. dstack and column_stack had the
same gap. Expected shapes were taken from PyTorch 2.9.0.
"""
import pytest
import torch

A, B = 1.0, 2.0


@pytest.mark.parametrize("name,shape", [("hstack", (2,)), ("vstack", (2, 1)), ("dstack", (1, 1, 2)),
                                        ("column_stack", (1, 2))])
def test_zero_dim_operands(name, shape):
    out = getattr(torch, name)([torch.tensor(A), torch.tensor(B)])
    assert tuple(out.shape) == shape
    assert out.reshape(-1).tolist() == [A, B]


def test_hstack_mixes_scalars_and_vectors():
    assert tuple(torch.hstack([torch.tensor(A), torch.tensor([B, 3.0])]).shape) == (3,)


def test_column_stack_mixes_scalars_and_vectors():
    assert tuple(torch.column_stack([torch.tensor(A), torch.tensor([B])]).shape) == (1, 2)
