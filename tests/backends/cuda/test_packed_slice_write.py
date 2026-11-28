import numpy as np
import pytest

from _helpers.capability import require_accelerator
import jittor as jt


@pytest.mark.cuda
@pytest.mark.parametrize(("start", "stop", "step", "dtype"), [
    (3, 13, 1, "float32"),
    (3, 13, 1, "bfloat16"),
    (1, 15, 2, "float32"),
    (-14, -1, 1, "float16"),
])
def test_assign_dense_interval(start, stop, step, dtype):
    require_accelerator("cuda")
    shape = (2, 4, 16, 68)
    key = (slice(None), slice(None), slice(start, stop, step))
    original = np.full(shape, -1, dtype=np.float32)
    expected = original.copy()
    values = np.arange(expected[key].size, dtype=np.float32).reshape(expected[key].shape) % 67
    expected[key] = values
    with jt.flag_scope(use_cuda=1, backend_fallback="error"), jt.no_grad():
        before = jt.array(original).cast(dtype)
        source = jt.array(values).cast(dtype)
        out = before.setitem(key, source)
        out.sync()
        assert out.location() == "device"
        np.testing.assert_array_equal(out.float32().numpy(), expected)
        np.testing.assert_array_equal(before.float32().numpy(), original)


@pytest.mark.cuda
def test_assign_dense_interval_gradient():
    require_accelerator("cuda")
    with jt.flag_scope(use_cuda=1, backend_fallback="error"):
        before = jt.ones((4, 16, 64)).start_grad()
        source = jt.ones((4, 8, 64)).start_grad()
        result = before.setitem((slice(None), slice(4, 12)), source)
        grad_before, grad_source = jt.grad(result.sum(), [before, source])
        expected = np.ones((4, 16, 64), dtype=np.float32)
        expected[:, 4:12] = 0
        np.testing.assert_array_equal(grad_before.numpy(), expected)
        np.testing.assert_array_equal(grad_source.numpy(), np.ones((4, 8, 64)))
