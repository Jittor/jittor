"""Execute CUDA implementations moved out of the shared operator sources."""

from _helpers import capability as _test_capability

import numpy as np
import pytest


@pytest.fixture
def cuda_runtime():
    import jittor as jt

    if not _test_capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip("CUDA runtime required")
    with jt.flag_scope(use_cuda=1, backend_fallback="error"):
        yield jt


def _device_value(value):
    value.sync()
    assert value.location() == "device"
    return value.numpy().copy()


def test_small_where_candidate_and_transpose_use_cuda_sources(cuda_runtime):
    jt = cuda_runtime
    condition = np.array([False, True, False, True, True, False])
    indices, = jt.where(jt.array(condition))
    np.testing.assert_array_equal(_device_value(indices), np.where(condition)[0])
    values = jt.array(np.array([[3], [2], [4], [1], [5]], dtype=np.float32))
    chosen = jt.candidate(values, "@x(j,0)>@x(i,0)")
    np.testing.assert_array_equal(_device_value(chosen), [0, 2, 4])
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    result = jt.ops.transpose(jt.array(source), (2, 0, 1))
    np.testing.assert_array_equal(_device_value(result), source.transpose(2, 0, 1))


def test_shared_indexing_gpu_fragments_preserve_values_and_gradients(cuda_runtime):
    jt = cuda_runtime
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    value = jt.array(source)
    selected = value[:, ::-1, 1::2]
    np.testing.assert_array_equal(_device_value(selected), source[:, ::-1, 1::2])
    expected_grad = np.zeros_like(source)
    expected_grad[:, ::-1, 1::2] = 1
    np.testing.assert_array_equal(_device_value(jt.grad(selected.sum(), value)), expected_grad)

    base = jt.array(np.ones((1, 4), dtype=np.float32))
    updates = jt.array(np.array([[1, 2, 3, 4]], dtype=np.float32))
    indices = jt.array(np.zeros((1, 4), dtype=np.int64))
    result = base.scatter_add(1, indices, updates)
    np.testing.assert_array_equal(_device_value(result), [[11, 1, 1, 1]])
    for derivative in jt.grad(result.sum(), [base, updates]):
        np.testing.assert_array_equal(_device_value(derivative), np.ones((1, 4)))


def test_legacy_complex_cuda_builders_execute_without_reinterpret_view(cuda_runtime, monkeypatch):
    jt = cuda_runtime
    monkeypatch.setattr(jt, "reinterpret_view", None)
    pairs = np.arange(12, dtype=np.float32).reshape(6, 2)
    value = jt.nn.view_as_complex(jt.array(pairs))
    np.testing.assert_array_equal(_device_value(value), pairs[:, 0] + 1j * pairs[:, 1])
    restored = jt.nn.view_as_real(value)
    np.testing.assert_array_equal(_device_value(restored), pairs)


def test_moved_connected_component_kernel_labels_distinct_regions(cuda_runtime):
    jt = cuda_runtime
    from jittor.backends.cuda.kernels.ccl.ccl_2d import label_image

    mask = np.zeros((32, 32), dtype=np.uint32)
    mask[2:4, 2:4] = 1
    mask[20:22, 20:22] = 1
    labels = label_image(jt.array(mask.reshape(-1)), jt.ones((1,), dtype="uint32"), 32, 32)
    actual = _device_value(labels).reshape(32, 32)
    first = np.unique(actual[2:4, 2:4])
    second = np.unique(actual[20:22, 20:22])
    assert first.size == second.size == 1
    assert first[0] != second[0]
