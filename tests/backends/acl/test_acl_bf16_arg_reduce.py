"""Native-mode arg reductions: dtype, first-index semantics, and full gradients."""

import numpy as np
import pytest
import jittor as jt

from _helpers import capability as _test_capability


_CASES = [
    (backend, dtype, kind, operation, dimension, keepdims)
    for backend in ("cpu", "acl")
    for dtype in (("float16", "float32") if backend == "cpu"
                  else ("bfloat16", "float16", "float32"))
    for kind in (("finite-ties", "nan-inf") if dtype == "bfloat16"
                 else ("finite-ties",))
    for operation in ("max", "min")
    for dimension in (-1, 0)
    for keepdims in (False, True)
]


def _reference(kind, operation, dimension, keepdims):
    source = np.array([[1, 4, 4, -2], [3, 1, -2, 3], [-1, 4, 0, 3]],
                      dtype=np.float32)
    if kind == "nan-inf":
        source = np.array([[np.nan, 2, -np.inf, 4],
                           [3, np.nan, np.inf, -3],
                           [np.nan, np.nan, -np.inf, 4]], dtype=np.float32)
    # First-index tie/NaN semantics were independently verified on native
    # TorchNPU 2.7.1.post2/CANN8.5. All finite values are BF16-exact.
    indices = getattr(np, "arg" + operation)(source, axis=dimension)
    expanded = np.expand_dims(indices, axis=dimension)
    values = np.take_along_axis(source, expanded, axis=dimension)
    if keepdims:
        indices = expanded
    else:
        values = np.squeeze(values, axis=dimension)
    weights = (np.arange(values.size, dtype=np.float32).reshape(values.shape)
               + 1) / 8
    gradient = np.zeros_like(source)
    np.put_along_axis(
        gradient, expanded,
        weights if keepdims else np.expand_dims(weights, axis=dimension),
        axis=dimension,
    )
    return source, indices, values, weights, gradient


@pytest.mark.parametrize(
    "backend,dtype,kind,operation,dimension,keepdims", _CASES,
    ids=[f"{backend}-{dtype}-{kind}-{operation}-dim{dimension}-keep{int(keepdims)}"
         for backend, dtype, kind, operation, dimension, keepdims in _CASES],
)
def test_arg_reduce_values_indices_full_gradient(
        backend, dtype, kind, operation, dimension, keepdims):
    if backend == "acl" and not _test_capability.check_accelerator(
            "acl", backend=jt).enabled:
        pytest.skip("ACL backend is unavailable")
    from jittor._runtime.fallback import forbid_backend_fallbacks

    source, expected_index, expected_value, weights, expected_gradient = (
        _reference(kind, operation, dimension, keepdims)
    )
    before = jt.core.backend_fallback_count()
    enabled = int(backend == "acl")
    with jt.flag_scope(use_acl=enabled, use_cuda=enabled), jt.runtime.scope(
            backend_fallback="error"), forbid_backend_fallbacks():
        x = jt.array(source, dtype=dtype).start_grad()
        indices, values = getattr(jt, "arg" + operation)(
            x, dim=dimension, keepdims=keepdims)
        gradient = jt.grad((values.float32() * jt.array(weights)).sum(), x)
        assert str(values.dtype) == dtype
        assert str(gradient.dtype) == dtype
        assert str(indices.dtype) == "int32"

        # Materialize first, then use native Var.location() before any host
        # fetch. dispatch_context alone describes a route, not actual storage.
        jt.sync_all(True)
        expected_location = "device" if backend == "acl" else "cpu"
        context = tuple(jt.core.dispatch_context([x]))
        assert context[0] == backend
        for tensor in (x, indices, values, gradient):
            assert tensor.location() == expected_location
            assert tuple(jt.core.dispatch_context([tensor])) == context
            if backend == "acl":
                assert tensor.device_id == context[1]
                assert tensor.device_id >= 0
        actual_index, actual_value, actual_gradient = jt.fetch_sync(
            [indices, values.float32(), gradient.float32()])

    assert jt.core.backend_fallback_count() == before
    np.testing.assert_array_equal(actual_index, expected_index)
    np.testing.assert_array_equal(actual_value, expected_value)
    np.testing.assert_array_equal(actual_gradient, expected_gradient)
