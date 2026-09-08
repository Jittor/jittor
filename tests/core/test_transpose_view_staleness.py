"""Transpose views follow source updates and support inverse write-through."""

import numpy as np
import pytest

import jittor as jt


@pytest.fixture(params=[0, 1], ids=["cpu", "cuda"])
def transpose_device(request):
    if request.param and not jt.has_cuda:
        pytest.skip("CUDA unavailable")
    with jt.flag_scope(use_cuda=request.param):
        yield
        jt.sync_all(True)


def test_a_transposed_view_sees_a_later_assign_to_its_source(transpose_device):
    a = jt.array(np.arange(12, dtype="float32").reshape(3, 4))
    b = jt.array(np.ones((3, 5), dtype="float32"))
    at = a.transpose()
    a.assign(jt.zeros((3, 4), "float32"))
    np.testing.assert_array_equal(jt.matmul(at, b).numpy(), np.zeros((4, 5), "float32"))
    np.testing.assert_array_equal(at.numpy(), np.zeros((4, 3), "float32"))


def test_materialized_transpose_refreshes_after_repeated_source_assign(transpose_device):
    source = np.arange(12, dtype="float32").reshape(3, 4)
    a = jt.array(source)
    at = a.transpose()
    at.sync()
    for offset in (3, 7):
        current = source + offset
        a.assign(jt.array(current))
        np.testing.assert_array_equal(at.numpy(), current.T)
        np.testing.assert_allclose(
            jt.matmul(at, jt.ones((3, 2))).numpy(),
            current.T @ np.ones((3, 2), "float32"), rtol=1e-6, atol=1e-6)


def test_transpose_then_matmul_without_an_assign_is_correct(transpose_device):
    source = np.arange(12, dtype="float32").reshape(3, 4)
    a = jt.array(source)
    b = jt.array(np.ones((3, 5), dtype="float32"))
    got = jt.matmul(a.transpose(), b).numpy()
    np.testing.assert_allclose(got, source.T @ np.ones((3, 5), "float32"),
                               rtol=1e-6, atol=1e-6)


def test_assignment_through_transpose_and_slice_updates_source(transpose_device):
    source = np.arange(12, dtype="float32").reshape(3, 4)
    a = jt.array(source)
    at = a.transpose()
    section = at[1:3]
    section.assign(jt.full((2, 3), 9.0))
    expected = source.copy()
    expected[:, 1:3] = 9
    np.testing.assert_array_equal(a.numpy(), expected)
    np.testing.assert_array_equal(at.numpy(), expected.T)
    np.testing.assert_array_equal(section.numpy(), expected.T[1:3])
    at.assign(jt.full((4, 3), 2.0))
    np.testing.assert_array_equal(a.numpy(), np.full((3, 4), 2.0, "float32"))


def test_batched_last_axes_transpose_is_an_explicit_view(transpose_device):
    source = np.arange(24, dtype="float32").reshape(2, 3, 4)
    a = jt.array(source)
    at = a.transpose(-1, -2)
    assert at._is_last2_transpose_view()
    for name in ("_jittor_transpose_base", "_jittor_transpose_axes", "_jittor_transpose_last2"):
        assert name not in vars(at)
    a.assign(jt.array(source + 1))
    b = jt.ones((2, 3, 2))
    expected = (source + 1).swapaxes(-1, -2) @ np.ones((2, 3, 2), "float32")
    np.testing.assert_allclose(jt.matmul(at, b).numpy(), expected, rtol=1e-6, atol=1e-6)


def test_general_permutation_and_its_gradient_follow_updated_base(transpose_device):
    source = np.arange(24, dtype="float32").reshape(2, 3, 4)
    a = jt.array(source)
    transposed = a.transpose((1, 2, 0))
    assert not transposed._is_last2_transpose_view()
    a.assign(jt.array(source + 1))
    np.testing.assert_array_equal(transposed.numpy(), (source + 1).transpose(1, 2, 0))
    np.testing.assert_array_equal(jt.grad(transposed.sum(), a).numpy(),
                                  np.ones_like(source))
