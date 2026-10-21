"""Unit-width chunks retain their axis and share storage with the input."""

import numpy as np
import pytest
import torch
import jittor as jt


@pytest.fixture(autouse=True)
def cpu_device():
    with jt.flag_scope(use_cuda=0, use_acl=0):
        yield


_CASES = [(3, 1), (7, 1), (4, -1), (8, -1), (2, 0)]


def _chunk(x, chunks, dim, native):
    return jt.chunk(x, chunks, dim) if native else x.chunk(chunks, dim=dim)


@pytest.mark.parametrize("chunks,dim", _CASES)
@pytest.mark.parametrize("native", [False, True])
def test_unit_chunks_forward_and_storage_alias(chunks, dim, native):
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    axis = dim % values.ndim
    with torch.no_grad():
        x = torch.tensor(values, device="cpu")
        parts = _chunk(x, chunks, dim, native)
        assert len(parts) == values.shape[axis]
        for index, part in enumerate(parts):
            selection = [slice(None)] * values.ndim
            selection[axis] = slice(index, index + 1)
            np.testing.assert_array_equal(part.numpy(), values[tuple(selection)])
            assert part.shape[axis] == 1
        # A write to a returned chunk must affect the input as Torch's view
        # contract requires. Advanced [index] selection instead returns a copy.
        parts[1].fill_(-19.0)
        expected = values.copy()
        selection = [slice(None)] * values.ndim
        selection[axis] = slice(1, 2)
        expected[tuple(selection)] = -19.0
        np.testing.assert_array_equal(x.numpy(), expected)


@pytest.mark.parametrize("chunks,dim", _CASES)
@pytest.mark.parametrize("native", [False, True])
def test_unit_chunks_weighted_input_gradient(chunks, dim, native):
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 11
    axis = dim % values.ndim
    rng = np.random.RandomState(20261020)
    expected = np.zeros_like(values)
    x = torch.tensor(values, device="cpu", requires_grad=True)
    parts = _chunk(x, chunks, dim, native)
    loss = None
    for index, part in enumerate(parts):
        cotangent = rng.uniform(-0.4, 0.4, tuple(part.shape)).astype(np.float32)
        term = (part * torch.tensor(cotangent, device="cpu")).sum() * (index + 1)
        loss = term if loss is None else loss + term
        selection = [slice(None)] * values.ndim
        selection[axis] = slice(index, index + 1)
        expected[tuple(selection)] = cotangent * (index + 1)
    gradient, = torch.autograd.grad(loss, (x,))
    np.testing.assert_allclose(gradient.numpy(), expected, rtol=1e-6, atol=1e-6)
