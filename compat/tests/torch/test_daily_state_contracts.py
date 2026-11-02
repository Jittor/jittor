"""Daily state contracts, also executable unchanged with independent PyTorch."""

import copy
import operator
import os

import numpy as np
import pytest
import torch


_DEVICES = tuple(name for name in os.environ.get(
    "JITTOR_TEST_DEVICES", "cpu").split(",") if name in ("cpu", "cuda"))


@pytest.fixture(params=_DEVICES)
def device(request):
    name = request.param
    previous = torch.get_default_device()
    is_shim = hasattr(torch, "_torch_compat_install_context")
    if is_shim:
        import jittor as jt
        from jittor._runtime.fallback import forbid_backend_fallbacks
        scope = jt.flag_scope(use_cuda=int(name == "cuda"))
        scope.__enter__()
        guard = forbid_backend_fallbacks()
        guard.__enter__()
        before = jt.core.backend_fallback_count()
    try:
        torch.set_default_device(name)
        yield name
        if name == "cuda":
            torch.cuda.synchronize()
        if is_shim:
            assert jt.core.backend_fallback_count() == before
    finally:
        torch.set_default_device(previous)
        if is_shim:
            guard.__exit__(None, None, None)
            scope.__exit__(None, None, None)


def values(tensor):
    return tensor.detach().cpu().numpy().copy()


def on_device(tensor, device):
    assert tensor.device.type == device
    if hasattr(torch, "_torch_compat_install_context"):
        import jittor as jt
        assert jt.core.dispatch_context([tensor])[0] == device


def test_empty_fast_path_defaults_to_a_frozen_leaf(device):
    for shape in ((7,), ((2, 3),), (2, 3)):
        target = torch.empty(*shape)
        assert not target.requires_grad and target.is_leaf
        target.zero_()
        assert not target.requires_grad
        on_device(target, device)
    target = torch.empty(2, requires_grad=True)
    assert target.requires_grad and target.is_leaf


@pytest.mark.parametrize("source_requires_grad", (False, True))
@pytest.mark.parametrize("target_requires_grad", (False, True))
def test_no_grad_copy_preserves_leaf_and_gradient_isolation(
        device, source_requires_grad, target_requires_grad):
    source = torch.tensor([3., 4.], requires_grad=source_requires_grad)
    target = torch.nn.Parameter(torch.tensor([1., 2.]),
                                requires_grad=target_requires_grad)
    with torch.no_grad():
        assert target.copy_(source) is target
    assert target.requires_grad == target_requires_grad
    assert target.is_leaf
    np.testing.assert_array_equal(values(target), [3., 4.])
    on_device(target, device)
    if target_requires_grad:
        (target * target).sum().backward()
        assert target.grad is not None
        np.testing.assert_array_equal(values(target.grad), [6., 8.])
        on_device(target.grad, device)
    assert source.grad is None
    with torch.no_grad():
        source.add_(10.)
    np.testing.assert_array_equal(values(target), [3., 4.])


@pytest.mark.parametrize("fused", (False, True))
def test_load_state_dict_keeps_model_storage_independent(device, fused):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1., 2.]))
            self.register_buffer("offset", torch.tensor([3., 4.]))

        def forward(self):
            return (self.weight * self.weight).sum()

    source, target = Model(), Model()
    parameter, buffer = target.weight, target.offset
    checkpoint = source.state_dict()
    target.load_state_dict(checkpoint)
    optimizer = torch.optim.AdamW(source.parameters(), lr=.1, fused=fused)
    source().backward()
    optimizer.step()
    # The target is deliberately first read after the source's in-place step:
    # load_state_dict must capture values when called, even with lazy execution.
    np.testing.assert_array_equal(values(target.weight), [1., 2.])
    assert not np.array_equal(values(source.weight), [1., 2.])
    assert target.weight is parameter and target.offset is buffer
    assert target.weight.requires_grad
    with torch.no_grad():
        source.offset.add_(10.)
        target.weight.add_(20.)
    np.testing.assert_array_equal(values(target.offset), [3., 4.])
    assert not np.array_equal(values(source.weight), values(target.weight))
    on_device(source.weight, device)
    on_device(target.weight, device)
    on_device(target.offset, device)


@pytest.mark.parametrize("fused", (False, None, True))
def test_sgd_common_execution_flags_and_momentum(device, fused):
    weight = torch.nn.Parameter(torch.tensor([1., 2.]))
    optimizer = torch.optim.SGD([weight], lr=.1, momentum=.9,
                                foreach=False, fused=fused)
    for _ in range(2):
        optimizer.zero_grad()
        (weight * weight).sum().backward()
        optimizer.step()
    np.testing.assert_allclose(values(weight), [.46, .92], atol=1e-6, rtol=1e-6)
    on_device(weight, device)
    assert optimizer.param_groups[0]["foreach"] is False
    assert optimizer.param_groups[0]["fused"] is fused


def test_sgd_positional_dampening_weight_decay_and_default_lr(device):
    weight = torch.nn.Parameter(torch.tensor([1., 2.]))
    # Torch positions: momentum, dampening, weight_decay (native order differs).
    optimizer = torch.optim.SGD([weight], .1, .9, 0., .5, False)
    (weight * weight).sum().backward()
    optimizer.step()
    np.testing.assert_allclose(values(weight), [.75, 1.5], atol=1e-6, rtol=1e-6)
    default = torch.optim.SGD([weight], foreach=False, fused=False)
    assert default.param_groups[0]["lr"] == .001
    on_device(weight, device)


def test_index_write_connects_rhs_without_promoting_no_grad_or_integer_target(device):
    source = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    mask = torch.tensor([[True, False, True, False]])
    hidden = torch.zeros(1, 4, 3)
    hidden[mask] = source
    assert hidden.requires_grad and not hidden.is_leaf
    hidden.sum().backward()
    np.testing.assert_array_equal(values(source.grad), np.ones((2, 3)))
    on_device(source.grad, device)
    frozen = torch.zeros(1, 4, 3)
    with torch.no_grad():
        frozen[mask] = source
    assert not frozen.requires_grad
    integer = torch.zeros(2, 3, dtype=torch.int64)
    integer[:] = source
    assert not integer.requires_grad
    np.testing.assert_array_equal(values(integer), [[1, 2, 3], [4, 5, 6]])


def test_failed_index_write_does_not_enable_destination_grad(device):
    target = torch.zeros(2, 3)
    source = torch.ones(4, 3, requires_grad=True)
    with pytest.raises(RuntimeError):
        target[:] = source
    assert not target.requires_grad
    np.testing.assert_array_equal(values(target), np.zeros((2, 3)))


def test_grad_enabled_copy_connects_source_and_owns_storage(device):
    source = torch.tensor([3., 4.], requires_grad=True)
    target = torch.zeros(2)
    target.copy_(source)
    assert target.requires_grad and not target.is_leaf
    (target * target).sum().backward()
    np.testing.assert_array_equal(values(source.grad), [6., 8.])
    with torch.no_grad():
        source.add_(10.)
    np.testing.assert_array_equal(values(target), [3., 4.])
    on_device(target, device)


@pytest.mark.parametrize("operation", ("normal_", "uniform_", "zero_", "fill_"))
def test_initialization_preserves_frozen_tensor_and_view_grad_state(device, operation):
    target = torch.zeros(2, 3)
    args = (.5,) if operation == "fill_" else ()
    getattr(target, operation)(*args)
    target[0].add_(1.)
    assert not target.requires_grad
    assert target.is_leaf
    assert np.isfinite(values(target)).all()
    on_device(target, device)


@pytest.mark.parametrize("operation,expected,gradient", (
    (operator.iadd, [3., 4.], [1., 1.]),
    (operator.isub, [-1., -2.], [-1., -1.]),
    (operator.imul, [2., 3.], [1., 1.]),
    (operator.itruediv, [.5, 1./3.], [-.25, -1./9.]),
))
def test_augmented_assignment_on_deepcopy_tracks_rhs_and_keeps_storage_independent(
        device, operation, expected, gradient):
    base = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    target = copy.deepcopy(base.data)
    source = torch.nn.Parameter(torch.tensor([2., 3.]))
    assert operation(target, source) is target
    assert target.requires_grad and not target.is_leaf
    target.sum().backward()
    np.testing.assert_allclose(values(target), expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(values(source.grad), gradient, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(values(base), [1., 1.])
    on_device(target, device)
    on_device(source.grad, device)
    frozen = torch.ones(2)
    with torch.no_grad():
        operation(frozen, source)
    assert not frozen.requires_grad
    with pytest.raises(RuntimeError, match="leaf Variable.*in-place operation"):
        operation(source, torch.ones(2))


@pytest.mark.parametrize("operation,args", (("normal_", (.2, .5)), ("uniform_", (-2., 3.))))
def test_random_initializer_honors_explicit_generator_and_preserves_global_rng(
        device, operation, args):
    generator = torch.Generator(device=device).manual_seed(73)
    peer = torch.Generator(device=device).manual_seed(73)
    get_global_state = torch.get_rng_state if device == "cpu" else torch.cuda.get_rng_state
    global_state = values(get_global_state())
    if hasattr(torch, "_torch_compat_install_context") and (
            operation == "normal_" or device == "cuda"):
        # The supported explicit stream is CPU uniform. Other combinations
        # reject before either the destination or generator changes.
        target = torch.ones(7)
        state = values(generator.get_state())
        with pytest.raises(NotImplementedError, match="explicit Generator"):
            getattr(target, operation)(*args, generator=generator)
        np.testing.assert_array_equal(values(target), np.ones(7))
        np.testing.assert_array_equal(values(generator.get_state()), state)
        np.testing.assert_array_equal(values(get_global_state()), global_state)
        return
    first, second, advanced, replay = (torch.empty(7) for _ in range(4))
    getattr(first, operation)(*args, generator=generator)
    getattr(second, operation)(*args, generator=peer)
    checkpoint = generator.get_state()
    getattr(advanced, operation)(*args, generator=generator)
    generator.set_state(checkpoint)
    getattr(replay, operation)(*args, generator=generator)
    np.testing.assert_array_equal(values(first), values(second))
    np.testing.assert_array_equal(values(advanced), values(replay))
    assert not np.array_equal(values(first), values(advanced))
    np.testing.assert_array_equal(values(get_global_state()), global_state)
    assert not first.requires_grad
    on_device(first, device)
