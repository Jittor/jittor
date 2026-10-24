"""Transpose views follow source updates and support inverse write-through."""

from _helpers import capability as _test_capability

import numpy as np
import pytest

import jittor as jt


@pytest.fixture(params=[0, 1], ids=["cpu", "cuda"])
def transpose_device(request):
    if request.param and not _test_capability.check_accelerator('cuda', backend=jt).enabled:
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


def test_a_size_changing_rebind_stales_a_lazy_view_instead_of_aborting(transpose_device):
    """A rebind that changes the element count cannot re-derive a lazy view.

    ``x.data = <different shape>`` is legal in torch, and vLLM-Omni's
    layerwise offload uses it to swap a parameter for a zero-element
    placeholder. ``flatten()`` records a storage view on that parameter, and
    re-deriving the reshape against the emptied holder aborted with
    "reshape shape is invalid for input of size [x_items(0) == y_items(1152)]".
    A view that can no longer be derived keeps the data it was taken from,
    which is what torch's views do after ``x.data = y`` replaces the storage
    under them.
    """
    x = jt.array(np.arange(12, dtype="float32").reshape(3, 4))
    flat = x.view(-1)
    x._update(jt.zeros((0,), "float32"))
    assert x.numel() == 0
    np.testing.assert_array_equal(flat.numpy(), np.arange(12, dtype="float32"))


@pytest.mark.parametrize("shape", [(1, 2), (2, 3)], ids=["unit_axis", "matrix"])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("materialized", [False, True], ids=["lazy", "synced"])
def test_matmul_keeps_exact_pretranspose_reshape_gradient_targets(
        transpose_device, shape, side, materialized):
    """The relay must preserve graph identity, not merely matching storage.

    Two rebinds construct new forward intermediates, as repeated checkpoint
    loads do. This is native math/target ownership, not an FSDP/PPO test.
    """
    fallback_before = jt.core.backend_fallback_count()
    backend = "cuda" if jt.flags.use_cuda else "cpu"
    root = jt.array(np.zeros(int(np.prod(shape)), dtype="float32"))
    expected_ordinal = int(jt.core.dispatch_context([root])[1])
    root_holder = id(root)
    for offset in (0, 1):
        weights = (np.arange(2, 2 + int(np.prod(shape)), dtype="float32") + offset).reshape(shape)
        root.assign(jt.array(weights.reshape(-1)))
        root.start_grad()
        assert id(root) == root_holder
        full = root.reshape(shape)
        full_holder, full_node = id(full), full.id
        transposed = full.transpose(-1, -2)
        assert transposed._is_last2_transpose_view()
        if materialized:
            transposed.sync()
        if side == "right":
            inputs = np.arange(1, 1 + 2 * shape[1], dtype="float32").reshape(2, shape[1])
            x = jt.array(inputs)
            output = jt.matmul(x, transposed)
            expected_output = inputs.astype("float64") @ weights.astype("float64").T
            expected_full = 2 * expected_output.T @ inputs.astype("float64")
            expected_input = 2 * expected_output @ weights.astype("float64")
        else:
            inputs = np.arange(1, 1 + 2 * shape[0], dtype="float32").reshape(shape[0], 2)
            x = jt.array(inputs)
            output = jt.matmul(transposed, x)
            expected_output = weights.astype("float64").T @ inputs.astype("float64")
            expected_full = 2 * inputs.astype("float64") @ expected_output.T
            expected_input = 2 * weights.astype("float64") @ expected_output
        if backend == "cuda":
            assert output._producer_name() == "cublas_matmul", "the transpose relay must remain active"
        loss = output.sqr().sum()
        gradients = jt.core.grad_optional(loss, [full, root, x], False)
        assert len(gradients) == 3 and all(gradient is not None for gradient in gradients)
        assert id(full) == full_holder and full.id == full_node
        expected_gradients = [expected_full, expected_full.reshape(-1), expected_input]
        values = [output, full, root, x] + list(gradients)
        # Check every live result before reading any tensor on the host.
        for value in values:
            value.sync()
            actual_backend, ordinal = jt.core.dispatch_context([value])
            assert actual_backend == backend
            assert value.location() == ("device" if backend == "cuda" else "cpu")
            if backend == "cuda":
                assert int(ordinal) == int(value.device_id) == expected_ordinal
        assert tuple(output.shape) == expected_output.shape
        np.testing.assert_allclose(output.cpu().numpy(), expected_output, rtol=1e-5, atol=1e-5)
        for gradient, expected in zip(gradients, expected_gradients):
            assert tuple(gradient.shape) == expected.shape
            actual = np.array(gradient.cpu().numpy(), copy=True)
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        assert jt.core.backend_fallback_count() == fallback_before == 0
