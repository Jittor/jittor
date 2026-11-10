"""A Python complex operand is a weak 0-dim scalar, as in Torch.

It used to become a one-element complex64 vector: ``1j * tensor(0.3)`` had
shape ``[1]``, a float64 operand produced complex64, the scalar value was
rounded to float32 and the result required grad even for a constant tensor,
so ``1j * sin(theta / 2)`` became a batch of one and lost double precision.
Expected shapes, dtypes and flags were taken
from PyTorch 2.9.0.
"""
import numpy as np
import pytest
import torch

S = 0.3

CASES = [
    ("1j*s", lambda s: 1j * s, 0.3j),
    ("s*1j", lambda s: s * 1j, 0.3j),
    ("s+1j", lambda s: s + 1j, 0.3 + 1j),
    ("s-1j", lambda s: s - 1j, 0.3 - 1j),
    ("1j-s", lambda s: 1j - s, -0.3 + 1j),
    ("s/1j", lambda s: s / 1j, -0.3j),
    ("1j/s", lambda s: 1j / s, 1j / 0.3),
    ("numpy", lambda s: np.complex128(1j) * s, 0.3j),
]


@pytest.mark.parametrize("name,fn,expected", CASES, ids=[c[0] for c in CASES])
def test_float64_scalar_tensor(name, fn, expected):
    out = fn(torch.tensor(S, dtype=torch.float64))
    assert tuple(out.shape) == ()
    assert out.dtype == torch.complex128
    assert out.requires_grad is False
    assert complex(out.item()) == pytest.approx(expected, abs=1e-15)


def test_scalar_value_keeps_double_precision():
    out = torch.tensor(S, dtype=torch.float64) * 0.1j
    assert complex(out.item()).imag == 0.3 * 0.1


@pytest.mark.parametrize("dtype,expected", [(torch.float32, torch.complex64), (torch.int64, torch.complex64),
                                            (torch.complex128, torch.complex128)])
def test_weak_scalar_result_dtype(dtype, expected):
    out = torch.ones(2, dtype=dtype) * 1j
    assert tuple(out.shape) == (2,) and out.dtype == expected


def test_gradient_flows_only_through_the_tensor():
    x = torch.tensor(S, dtype=torch.float64, requires_grad=True)
    out = x * 1j
    assert out.requires_grad is True
    out.imag.backward()
    assert float(x.grad) == 1.0
