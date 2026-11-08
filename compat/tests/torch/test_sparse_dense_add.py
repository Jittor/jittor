from _helpers import capability
import copy

import numpy as np
import jittor as jt
import pytest
import torch


def _cpu_test(function):
    def wrapped():
        with jt.runtime.scope(use_cuda=0):
            function()
    return wrapped


@_cpu_test
def test_dense_add_sparse_is_out_of_place():
    dense = torch.ones((2, 2))
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0], [1]], dtype=torch.long),
        torch.tensor([4.0]),
        size=(2, 2),
    )

    result = dense + sparse

    assert result is not dense
    np.testing.assert_allclose(dense.numpy(), np.ones((2, 2), dtype="float32"))
    np.testing.assert_allclose(
        result.numpy(), np.array([[1.0, 5.0], [1.0, 1.0]], dtype="float32")
    )


@_cpu_test
def test_iadd_promotes_constant_holder_and_preserves_rhs_graph():
    values = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    base = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    result = base.data.clone()
    alias = result

    result += values
    result += values * 2

    assert result is alias
    assert result.requires_grad and not result.is_leaf
    result.sum().backward()
    np.testing.assert_array_equal(values.grad.numpy(), [3.0, 3.0])

    leaf = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(RuntimeError, match="leaf Variable.*in-place operation"):
        leaf += torch.ones(2)
    frozen = torch.ones(2)
    with torch.no_grad():
        frozen += values
    assert not frozen.requires_grad and frozen.is_leaf


@_cpu_test
def test_iadd_preserves_inplace_shape_and_dtype_contract():
    with pytest.raises(RuntimeError, match="can't be cast"):
        integer = torch.ones(2, dtype=torch.int64)
        integer += torch.ones(2, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="doesn't match the broadcast shape"):
        narrow = torch.ones((1, 2))
        narrow += torch.ones((3, 2))
    value = torch.ones(2, dtype=torch.float32)
    identity = value
    value += torch.ones(2, dtype=torch.float64)
    assert value is identity
    assert value.dtype is torch.float32
    assert tuple(value.shape) == (2,)


@_cpu_test
def test_iadd_sparse_preserves_holder_and_sparse_value_gradient():
    values = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        values,
        size=(2, 2),
    )
    base = torch.nn.Parameter(torch.ones((2, 2)), requires_grad=False)
    result = base.data.clone()
    alias = result

    result += sparse

    assert result is alias
    assert result.requires_grad and not result.is_leaf
    torch.nn.functional.linear(torch.ones((1, 2)), result).sum().backward()
    np.testing.assert_array_equal(values.grad.numpy(), [1.0, 1.0])


@_cpu_test
def test_deepcopy_data_view_is_independent_and_keeps_rhs_graph():
    values = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    base = torch.nn.Parameter(torch.ones((2, 2)), requires_grad=False)
    original = base.detach().clone()
    result = copy.deepcopy(base.data[:, :])
    alias = result

    result += values.reshape(1, 2)

    assert result is alias
    assert result.requires_grad and not result.is_leaf
    result.sum().backward()
    np.testing.assert_array_equal(values.grad.numpy(), [2.0, 2.0])
    np.testing.assert_array_equal(base.numpy(), original.numpy())
    assert result._torch_data_owner is None
    assert result._torch_data_path == ()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("alpha", [1, 2.5, -3.0])
@pytest.mark.parametrize("operation", ["add", "add_", "sub_", "torch.add"])
@pytest.mark.parametrize("strided", [False, True])
def test_dense_sparse_alpha_storage_and_all_input_gradients(device, alpha, operation, strided):
    if device == "cuda" and not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("no usable CUDA in this build")
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_cuda=int(device == "cuda"), backend_fallback="error"):
        source = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64,
                              device=device, requires_grad=True)
        values = torch.tensor([2., 3., -4.], dtype=torch.float64,
                              device=device, requires_grad=True)
        indices = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.int64, device=device)
        sparse = torch.sparse_coo_tensor(indices, values, (2, 2), device=device)
        target = source * 1
        if strided:
            target = target.t()
        if operation == "torch.add":
            result = torch.add(target, sparse, alpha=alpha)
        else:
            result = getattr(target, operation)(sparse, alpha=alpha)
        sign = -1 if operation == "sub_" else 1
        dense = np.array([[0., 5.], [-4., 0.]])
        initial = np.array([[1., 2.], [3., 4.]])
        if strided:
            initial = initial.T
        expected = initial + sign * alpha * dense
        np.testing.assert_array_equal(result.detach().cpu().numpy(), expected)
        assert str(result.device).split(":")[0] == device
        if operation.endswith("_"):
            assert result is target
        else:
            np.testing.assert_array_equal(target.detach().cpu().numpy(), initial)
        seed = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64, device=device)
        (result * seed).sum().backward()
        gradient = np.array([[1., 2.], [3., 4.]])
        np.testing.assert_array_equal(source.grad.cpu().numpy(), gradient.T if strided else gradient)
        np.testing.assert_array_equal(values.grad.cpu().numpy(), sign * alpha * np.array([2., 2., 3.]))
    assert jt.core.backend_fallback_count() == before


@pytest.mark.parametrize("strided", [False, True])
def test_deepcopy_data_is_independent_and_retains_sparse_value_gradients(strided):
    parameter = torch.nn.Parameter(torch.tensor([[1., 2.], [3., 4.]]), requires_grad=False)
    data = parameter.data.t() if strided else parameter.data
    data.user_state = {"values": [7]}
    target = copy.deepcopy(data)
    target.user_state["values"].append(8)
    assert data.user_state == {"values": [7]}
    assert target.data_ptr() != parameter.data_ptr()
    values = torch.tensor([2., 3., -4.], requires_grad=True)
    sparse = torch.sparse_coo_tensor([[0, 0, 1], [1, 1, 0]], values, (2, 2))
    target.add_(sparse, alpha=2)
    np.testing.assert_array_equal(parameter.detach().numpy(), [[1., 2.], [3., 4.]])
    seed = torch.tensor([[1., 2.], [3., 4.]])
    (target * seed).sum().backward()
    assert values.grad is not None
    np.testing.assert_array_equal(values.grad.numpy(), [4., 4., 6.])


def test_deepcopy_parameter_preserves_memo_user_state_and_original_data_alias():
    parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
    shared = {"values": [7]}
    tensor = parameter.detach()
    tensor.first = shared
    tensor.second = shared
    copies = copy.deepcopy([parameter, parameter, tensor, tensor])
    assert copies[0] is copies[1]
    cloned = copies[0]
    assert isinstance(cloned, torch.nn.Parameter)
    assert cloned.requires_grad and cloned.is_leaf
    assert cloned.data_ptr() != parameter.data_ptr()
    assert copies[2] is copies[3]
    assert copies[2].first is copies[2].second
    assert copies[2].first is not shared
    copies[2].first["values"].append(8)
    assert shared == {"values": [7]}
    with torch.no_grad():
        parameter.data.add_(3)
    np.testing.assert_array_equal(parameter.detach().numpy(), [4., 5.])
    np.testing.assert_array_equal(cloned.detach().numpy(), [1., 2.])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("operation", ["add_", "sub_"])
@pytest.mark.parametrize("strided", [False, True])
def test_no_grad_sparse_write_preserves_allocation_and_alias(device, operation, strided):
    if device == "cuda" and not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("no usable CUDA in this build")
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_cuda=int(device == "cuda"), backend_fallback="error"), torch.no_grad():
        target = torch.tensor([[1., 2.], [3., 4.]], device=device)
        if strided:
            target = target.t()
        alias = target.detach()
        pointer = target.data_ptr()
        values = torch.tensor([2., 3., -4.], device=device)
        indices = torch.tensor([[0, 0, 1], [1, 1, 0]], device=device)
        sparse = torch.sparse_coo_tensor(indices, values, (2, 2), device=device)
        actual = getattr(target, operation)(sparse, alpha=-2.5)
        assert actual is target
        assert target.data_ptr() == pointer == alias.data_ptr()
        initial = np.array([[1., 2.], [3., 4.]])
        if strided:
            initial = initial.T
        sign = -1 if operation == "sub_" else 1
        expected = initial - sign * 2.5 * np.array([[0., 5.], [-4., 0.]])
        np.testing.assert_array_equal(actual.cpu().numpy(), expected)
        np.testing.assert_array_equal(alias.cpu().numpy(), expected)
    assert jt.core.backend_fallback_count() == before
