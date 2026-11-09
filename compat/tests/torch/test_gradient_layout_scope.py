"""The scoped gradient-layout seam and ordinary Torch data-change behavior."""
import threading
import pytest
import torch

from jittor.compat.torch.gradient_task import leaf_gradient_layout_scope, _resolve_leaf_gradient_layout
from jittor.compat.torch.installers.tensor.autograd_api import _initial_leaf_gradient


@pytest.fixture(params=['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def device(request):
    if request.param=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


def test_resolver_is_nested_thread_local_and_restored():
    param=object(); grad=object()
    first=lambda p,g: 'first'
    second=lambda p,g: 'second'
    assert _resolve_leaf_gradient_layout(param,grad) is None
    with leaf_gradient_layout_scope(first):
        assert _resolve_leaf_gradient_layout(param,grad)=='first'
        observed=[]
        worker=threading.Thread(target=lambda: observed.append(_resolve_leaf_gradient_layout(param,grad)))
        worker.start(); worker.join()
        assert observed==[None]
        with pytest.raises(ValueError):
            with leaf_gradient_layout_scope(second):
                assert _resolve_leaf_gradient_layout(param,grad)=='second'
                raise ValueError('exercise cleanup')
        assert _resolve_leaf_gradient_layout(param,grad)=='first'
    assert _resolve_leaf_gradient_layout(param,grad) is None
    with pytest.raises(TypeError):
        with leaf_gradient_layout_scope(None):
            pass


def test_shape_resolver_is_explicit_and_validated(device):
    parameter=torch.nn.Parameter(torch.empty(0,device=device))
    gradient=torch.ones((2,3),device=device)
    with pytest.raises(RuntimeError):
        _initial_leaf_gradient(parameter,gradient)
    with leaf_gradient_layout_scope(lambda p,g: g.contiguous() if p is parameter else None):
        actual=_initial_leaf_gradient(parameter,gradient)
    assert tuple(actual.shape)==(2,3) and actual.device.type==device
    with leaf_gradient_layout_scope(lambda p,g: torch.ones(1,device=device)):
        with pytest.raises(RuntimeError,match='shape, dtype or device'):
            _initial_leaf_gradient(parameter,gradient)
    with leaf_gradient_layout_scope(lambda p,g: 3):
        with pytest.raises(TypeError,match='Tensor or None'):
            _initial_leaf_gradient(parameter,gradient)


def test_no_scope_preserves_shape_error_on_backward(device):
    p=torch.nn.Parameter(torch.arange(6.,device=device).reshape(2,3))
    loss=(p*2).sum()
    p.data=torch.empty(0,device=device)
    with pytest.raises(RuntimeError):
        loss.backward()
