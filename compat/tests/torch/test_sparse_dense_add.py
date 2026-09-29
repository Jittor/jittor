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


@_cpu_test
def test_tensor_uniform_honors_generator_stream():
    first_generator = torch.Generator(device="cpu").manual_seed(73)
    second_generator = torch.Generator(device="cpu").manual_seed(73)
    first = torch.empty(7).uniform_(-2.0, 3.0, generator=first_generator)
    second = torch.empty(7).uniform_(-2.0, 3.0, generator=second_generator)
    advanced = torch.empty(7).uniform_(-2.0, 3.0, generator=first_generator)

    np.testing.assert_array_equal(first.numpy(), second.numpy())
    assert not np.array_equal(first.numpy(), advanced.numpy())
    assert first_generator.get_state().numpy().tolist() == [73, 14]
    assert second_generator.get_state().numpy().tolist() == [73, 7]
