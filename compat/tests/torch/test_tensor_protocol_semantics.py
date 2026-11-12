"""Tensor protocol details that array-dispatch libraries rely on.

Each expectation was taken from PyTorch 2.9.0 in a separate interpreter:
``bool`` of any one-element tensor, ``ndimension``, NumPy integers as sizes,
``input=`` keywords, refusing ``backward``/NumPy conversion where torch
refuses, nested ``torch.tensor`` construction from tensors, ``torch.Size``
normalization, ``tensor == None``, ``torch.vander``, ``torch.dtype`` pickling by identity,
constants staying constant through ``matrix_exp`` and a zero
Jacobian for a complex output that has no graph.
"""
import numpy as np
import pytest
import torch


def test_bool_of_one_element_tensors():
    assert bool(torch.tensor(1.5)) is True and bool(torch.tensor(0.0)) is False
    assert bool(torch.tensor([3])) is True and bool(torch.tensor(1 + 0j)) is True
    with pytest.raises(RuntimeError, match="more than one value"):
        bool(torch.tensor([1.0, 2.0]))
    with pytest.raises(RuntimeError, match="no values"):
        bool(torch.tensor([]))


def test_ndimension():
    assert torch.ones(2, 3).ndimension() == 2 and torch.tensor(1.0).ndimension() == 0


def test_numpy_integer_sizes():
    n = np.int64(2)
    assert tuple(torch.Tensor(n, 3).shape) == (2, 3)
    assert tuple(torch.Tensor(n).shape) == (2,)
    assert tuple(torch.nn.Linear(n, n).weight.shape) == (2, 2)
    assert tuple(torch.nn.Embedding(n, n).weight.shape) == (2, 2)
    assert tuple(torch.cat([torch.ones(2), torch.ones(3)], dim=np.int64(0)).shape) == (5,)


def test_input_keyword():
    assert torch.prod(input=torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=0).tolist() == [3.0, 8.0]


def test_backward_requires_a_graph():
    with pytest.raises(RuntimeError, match="does not require grad and does not have a grad_fn"):
        torch.tensor(1.0).backward()
    leaf = torch.tensor(2.0, requires_grad=True)
    (leaf * leaf).backward()
    leaf.backward()
    assert float(leaf.grad) == 5.0


def test_numpy_protocol_refuses_grad_tensors():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(RuntimeError, match="Can't call numpy\\(\\) on Tensor that requires grad"):
        np.asarray(x)
    assert np.asarray(x.detach()).tolist() == [1.0, 2.0]
    with torch.no_grad():
        assert np.asarray(x).tolist() == [1.0, 2.0]
    assert x.tolist() == [1.0, 2.0] and "1." in repr(x)


def test_tensor_from_nested_tensors():
    x = torch.tensor([0.5, 1.5], dtype=torch.float64, requires_grad=True)
    out = torch.tensor([[0, -torch.cos(x[0])], [x[1] * 2, 1]])
    assert tuple(out.shape) == (2, 2) and out.dtype == torch.float64 and not out.requires_grad
    assert tuple(torch.tensor([x[:1].detach(), x[1:].detach()]).shape) == (2,)


def test_size_normalizes_integers():
    size = torch.Size((2, np.int64(1)))
    assert size == torch.ones(2, 1).shape and type(size[1]) is int
    with pytest.raises(TypeError):
        torch.Size((2.0,))


def test_equality_with_none():
    t = torch.tensor([1.0, 2.0])
    assert (t == None) is False and (t != None) is True  # noqa: E711
    assert [t, None].count(None) == 1
    assert (t == torch.tensor([1.0, 0.0])).tolist() == [True, False]


def test_ndarray_operands_are_constants():
    t = torch.tensor(2.0, dtype=torch.float64)
    out = np.array([1.0, 4.0]) / t
    assert out.requires_grad is False and out.dtype == torch.float64
    assert out.tolist() == [0.5, 2.0]
    g = torch.tensor([1.0, 3.0], dtype=torch.float64, requires_grad=True)
    assert (g / np.array([1.0, 4.0])).requires_grad is True


def test_linalg_shapes_and_dtypes():
    u = torch.tensor([[1 + 1j, 2], [0.5j, 1 - 1j]], dtype=torch.complex128)
    r = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    assert tuple(torch.linalg.det(u).shape) == () and tuple(torch.det(r).shape) == ()
    assert tuple(torch.linalg.det(torch.stack([u, u])).shape) == (2,)
    sign, logabs = torch.linalg.slogdet(r)
    assert tuple(sign.shape) == () and float(logabs) == pytest.approx(np.log(2.0))
    h = torch.tensor([[2, 1j], [-1j, 3]], dtype=torch.complex128)
    assert torch.linalg.eigvalsh(h).dtype == torch.float64
    w, v = torch.linalg.eigh(h)
    assert w.dtype == torch.float64 and v.dtype == torch.complex128
    assert torch.linalg.eigvals(r).dtype == torch.complex128
    assert torch.linalg.eig(r)[0].dtype == torch.complex128


def test_new_functions():
    x = torch.tensor([0.0, 0.5, -1.0], dtype=torch.float64)
    entr = torch.special.entr(x).tolist()
    assert entr[0] == 0.0 and entr[1] == pytest.approx(0.5 * np.log(2.0)) and entr[2] == float("-inf")
    a = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64)
    expected = np.array([[np.cos(1.0), np.sin(1.0)], [-np.sin(1.0), np.cos(1.0)]])
    np.testing.assert_allclose(torch.linalg.matrix_exp(a).numpy(), expected, atol=1e-14)
    np.testing.assert_allclose(a.matrix_exp().numpy(), expected, atol=1e-14)


def test_complex_setitem_converts_precision():
    x = torch.zeros(3, dtype=torch.complex128)
    x[1] = torch.tensor(1 + 2j, dtype=torch.complex64)
    y = torch.zeros(2, dtype=torch.complex64)
    y[0] = torch.tensor(0.5 + 0.25j, dtype=torch.complex128)
    assert x.tolist()[1] == 1 + 2j and y.tolist()[0] == 0.5 + 0.25j


def test_matrix_norm_orders_return_tensors():
    a = torch.tensor([[1.0, -2.0], [3.0, 0.5]], dtype=torch.float64)
    inf = float("inf")
    assert float(torch.linalg.norm(a, ord=inf)) == 3.5
    assert float(torch.linalg.norm(a, ord=-inf)) == 3.0
    assert float(torch.linalg.matrix_norm(a, ord=1)) == 4.0
    assert float(torch.linalg.vector_norm(a, ord=inf)) == 3.0


def test_complex_classification():
    c = torch.tensor([complex("nan"), 1 + 1j, complex(1, float("nan")), complex(float("inf"), 0)],
                     dtype=torch.complex128)
    assert torch.isnan(c).tolist() == [True, False, True, False]
    assert torch.isinf(c).tolist() == [False, False, False, True]
    assert torch.isfinite(c).tolist() == [False, True, False, False]


def test_numpy_scalars_are_constants():
    x = torch.tensor(0.3, dtype=torch.float64)
    for value in (x - np.float64(5.1), x + np.float32(1.0), np.float64(2.0) * x, x * np.int64(3)):
        assert value.requires_grad is False and value.dtype == torch.float64
    half = torch.tensor([1.0], dtype=torch.float16)
    assert (half + np.float64(1.0)).dtype == torch.float16
    assert (torch.tensor([1], dtype=torch.int32) * np.float32(1.5)).dtype == torch.float32
    g = torch.tensor(1.0, requires_grad=True)
    assert (g - np.float64(2.0)).requires_grad is True


def test_vander():
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    v = torch.vander(x, 3)
    assert v.tolist() == [[1.0, 1.0, 1.0], [4.0, 2.0, 1.0], [9.0, 3.0, 1.0]]
    assert v.dtype == torch.float64
    assert torch.vander(torch.tensor([2, 3]), increasing=True).tolist() == [[1, 2], [1, 3]]
    assert tuple(torch.vander(torch.tensor([2.0]), 0).shape) == (1, 0)
    v.sum().backward()
    assert x.grad.tolist() == [3.0, 5.0, 7.0]


def test_dtype_pickles_by_identity():
    import pickle
    assert pickle.loads(pickle.dumps(torch.float64)) is torch.float64
    assert pickle.loads(pickle.dumps(torch.dtype)) is torch.dtype


def test_matrix_exp_of_constant_is_constant():
    a = torch.tensor([[0.0, 1j], [1j, 0.0]], dtype=torch.complex128)
    assert torch.matrix_exp(a).requires_grad is False
    x = a.clone().requires_grad_()
    y = torch.matrix_exp(x)
    assert y.requires_grad is True
    y.real.sum().backward()
    assert np.allclose(x.grad.numpy(), np.full((2, 2), np.cos(1) - 1j * np.sin(1)))


def test_jacobian_of_complex_output_without_graph_is_zero():
    x = torch.tensor([0.2, 0.3], dtype=torch.float64, requires_grad=True)
    jac = torch.autograd.functional.jacobian(lambda t: (t * 1j).detach(), x)
    assert tuple(jac.shape) == (2, 2) and float(jac.abs().max()) == 0.0
    with pytest.raises(RuntimeError, match="real scalar outputs"):
        torch.autograd.functional.jacobian(lambda t: t * 1j, x)
