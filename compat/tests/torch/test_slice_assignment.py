"""Slice assignment preserves independent Torch types and both input gradients."""

import numpy as np
import pytest
import jittor as jt
import torch

from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks


_CASES = [
    pytest.param((slice(0, 2), slice(1, 3)), 0.0, None, id="python_scalar"),
    pytest.param((slice(0, 2), slice(1, 3)), np.array(-2.0, dtype=np.float32),
                 np.array(18.0, dtype=np.float32), id="scalar_tensor"),
    pytest.param((slice(0, 2), slice(1, 3)), np.array([[-1.0, -2.0]], dtype=np.float32),
                 np.array([[8.0, 10.0]], dtype=np.float32), id="broadcast_row"),
    pytest.param((1, slice(1, 3)), np.array([-1.0, -2.0], dtype=np.float32),
                 np.array([6.0, 7.0], dtype=np.float32), id="integer_axis"),
]


def _check_slice_assignment(device, index, rhs_np, expected_rhs_gradient):
    assert torch is not jt
    assert torch.Tensor is not jt.Var
    assert torch.Tensor._frontend_backend is jt
    source_np = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    cotangent_np = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    expected = source_np.copy()
    expected[index] = rhs_np
    expected_base_gradient = cotangent_np.copy()
    expected_base_gradient[index] = 0.0

    with forbid_backend_fallbacks():
        base = torch.tensor(source_np, device=device, requires_grad=True)
        output = base * 1.0  # Assign to a non-leaf, retaining the original base.
        rhs = (float(rhs_np) if expected_rhs_gradient is None else
               torch.tensor(rhs_np, device=device, requires_grad=True))
        output[index] = rhs
        cotangent = torch.tensor(cotangent_np, device=device)
        inputs = (base,) if expected_rhs_gradient is None else (base, rhs)
        gradients = torch.autograd.grad((output * cotangent).sum(), inputs)
        tensors = (output,) + tuple(gradients)
        for value in tensors:
            assert type(value) is torch.Tensor
            value.sync()
            assert value.placement_backend == (2 if device == "npu" else 0)
            assert value.location() == ("device" if device == "npu" else "cpu")
            if device == "npu":
                assert value.device_id >= 0
        actual = [value.detach().cpu().numpy() for value in tensors]
        if expected_rhs_gradient is not None:
            assert tuple(gradients[1].shape) == tuple(rhs.shape) == rhs_np.shape

    np.testing.assert_array_equal(actual[0], expected)
    np.testing.assert_array_equal(actual[1], expected_base_gradient)
    if expected_rhs_gradient is not None:
        np.testing.assert_array_equal(actual[2], expected_rhs_gradient)


@pytest.mark.cpu
@pytest.mark.parametrize("index,rhs_np,expected_rhs_gradient", _CASES)
def test_slice_assignment_forward_backward_cpu(index, rhs_np, expected_rhs_gradient):
    with jt.flag_scope(use_cuda=0):
        _check_slice_assignment("cpu", index, rhs_np, expected_rhs_gradient)


@pytest.mark.npu
@pytest.mark.parametrize("index,rhs_np,expected_rhs_gradient", _CASES)
def test_slice_assignment_forward_backward_npu(index, rhs_np, expected_rhs_gradient):
    if not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("No ACL found")
    with jt.flag_scope(use_acl=1, use_cuda=1):
        _check_slice_assignment("npu", index, rhs_np, expected_rhs_gradient)


_MASK_CASES = [
    pytest.param(np.zeros((2, 3), dtype=bool), 0.0, id="empty_mask"),
    pytest.param(np.array([[True, False, True], [False, True, False]]),
                 9.0, id="partial_mask"),
    pytest.param(np.ones((2, 3), dtype=bool), 21.0, id="full_mask"),
]


def _check_mask_scalar_assignment(device, mask_np, expected_rhs_gradient):
    assert torch is not jt
    assert torch.Tensor is not jt.Var
    assert torch.Tensor._frontend_backend is jt
    source_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    cotangent_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    expected = source_np.copy()
    expected[mask_np] = -2.0
    expected_base_gradient = cotangent_np.copy()
    expected_base_gradient[mask_np] = 0.0

    with forbid_backend_fallbacks():
        base = torch.tensor(source_np, device=device, requires_grad=True)
        rhs = torch.tensor(-2.0, device=device, requires_grad=True)
        mask = torch.tensor(mask_np, device=device, dtype=torch.bool)
        output = base * 1.0
        output[mask] = rhs
        cotangent = torch.tensor(cotangent_np, device=device)
        base_gradient, rhs_gradient = torch.autograd.grad(
            (output * cotangent).sum(), (base, rhs))
        tensors = (output, base_gradient, rhs_gradient)
        for value in tensors:
            assert type(value) is torch.Tensor
            value.sync()
            assert value.placement_backend == (2 if device == "npu" else 0)
            assert value.location() == ("device" if device == "npu" else "cpu")
            if device == "npu":
                assert value.device_id >= 0
        assert tuple(rhs.shape) == tuple(rhs_gradient.shape) == ()
        actual = [value.detach().cpu().numpy() for value in tensors]

    np.testing.assert_array_equal(actual[0], expected)
    np.testing.assert_array_equal(actual[1], expected_base_gradient)
    np.testing.assert_array_equal(actual[2], np.array(expected_rhs_gradient, dtype=np.float32))


@pytest.mark.cpu
@pytest.mark.parametrize("mask_np,expected_rhs_gradient", _MASK_CASES)
def test_mask_scalar_assignment_forward_backward_cpu(mask_np, expected_rhs_gradient):
    with jt.flag_scope(use_cuda=0):
        _check_mask_scalar_assignment("cpu", mask_np, expected_rhs_gradient)


@pytest.mark.npu
@pytest.mark.parametrize("mask_np,expected_rhs_gradient", _MASK_CASES)
def test_mask_scalar_assignment_forward_backward_npu(mask_np, expected_rhs_gradient):
    if not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("No ACL found")
    with jt.flag_scope(use_acl=1, use_cuda=1):
        _check_mask_scalar_assignment("npu", mask_np, expected_rhs_gradient)
