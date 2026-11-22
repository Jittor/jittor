"""Device and autograd regression for the public torch.fill APIs."""
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from _device_assertions import require_residency


@pytest.fixture(params=['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


@pytest.mark.parametrize('dtype', [torch.float32, torch.int64, torch.bool])
def test_fill_does_not_mutate_and_keeps_shape_dtype(device, dtype):
    x = torch.tensor([0, 1, 0], dtype=dtype, device=device)
    before = x.detach().cpu().numpy().copy()
    actual = torch.fill(x, 1)
    require_residency(actual, device)
    assert actual.shape == x.shape and actual.dtype == dtype
    np.testing.assert_array_equal(x.detach().cpu().numpy(), before)
    np.testing.assert_array_equal(actual.detach().cpu().numpy(), np.ones(3))


def test_fill_gradients_and_ignores_nan_inputs(device):
    x = torch.tensor([float('nan'), float('inf')], dtype=torch.float32, device=device, requires_grad=True)
    value = torch.tensor(2.5, dtype=torch.float32, device=device, requires_grad=True)
    result = torch.fill(x, value)
    require_residency(result, device)
    np.testing.assert_array_equal(result.detach().cpu().numpy(), [2.5, 2.5])
    result.sum().backward()
    require_residency(x.grad, device)
    require_residency(value.grad, device)
    np.testing.assert_array_equal(x.grad.detach().cpu().numpy(), [0., 0.])
    assert value.grad.item() == 2.


def test_fill_scalar_and_empty_shapes(device):
    scalar = torch.fill(torch.tensor(3., device=device), 4.)
    empty = torch.fill(torch.empty((0, 2), device=device), 4.)
    require_residency(scalar, device)
    require_residency(empty, device)
    assert tuple(scalar.shape) == () and scalar.item() == 4.
    assert tuple(empty.shape) == (0, 2) and empty.numel() == 0


def test_fill_inplace_and_invalid_value(device):
    x = torch.zeros(2, device=device)
    result = torch.fill_(x, 3.)
    assert result is x
    np.testing.assert_array_equal(x.cpu().numpy(), [3., 3.])
    with pytest.raises(RuntimeError, match='0-dimension'):
        torch.fill(x, torch.ones(2, device=device))
    with pytest.raises(TypeError):
        torch.fill(x, 3, out=x)


@pytest.mark.parametrize('no_grad', [False, True])
def test_literal_bool_fill_preserves_view_readers_and_empty_write_version(device, no_grad):
    base = torch.tensor([False, True, False, True], device=device)
    view = base[1:3]
    empty = view[:0]
    require_residency(base, device)
    require_residency(view, device)
    version = base._version
    pointer = base.data_ptr()
    before = torch.logical_not(view)

    scope = nullcontext()
    if hasattr(torch, '_torch_compat_install_context'):
        import jittor as jt
        scope = jt.flag_scope(auto_graph_replay=0)
    with scope, torch.no_grad() if no_grad else nullcontext():
        assert view.fill_(True) is view
        assert empty.fill_(False) is empty

    assert base._version == version + 2
    assert base.data_ptr() == pointer
    assert view.dtype == empty.dtype == torch.bool
    np.testing.assert_array_equal(base.cpu().numpy(), [False, True, True, True])
    np.testing.assert_array_equal(before.cpu().numpy(), [False, True])
    assert empty.shape == (0,)
