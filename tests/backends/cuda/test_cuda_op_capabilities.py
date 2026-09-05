"""Native semantic capabilities select real CUDA implementations and graph relays."""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def cuda_runtime():
    import jittor as jt

    if not jt.has_cuda or "cuda" not in jt.core.registered_backends():
        pytest.skip("CUDA runtime required")
    with jt.flag_scope(use_cuda=1, auto_mixed_precision_level=0, enable_tuner=1,
                       profiler_hide_relay=0, profiler_record_shape=0):
        yield jt


def _device_array(value):
    value.sync()
    assert value.location() == "device"
    return value.numpy().copy()


def _assert_executed(report, implementation):
    count_column = report[0].index("Count")
    matching = [row for row in report[1:] if implementation in row[0]]
    assert matching, (implementation, report)
    assert sum(float(row[count_column]) for row in matching) >= 1, matching


def _assert_relay_executed(report, implementation):
    _assert_executed(report, implementation)
    filename_column = report[0].index("FileName")
    sources = [Path(row[filename_column]) for row in report[1:]
               if "opkey0:" in row[0]]
    assert any(path.is_file() and "relay_groups[" in path.read_text(encoding="utf-8")
               for path in sources), report


def test_cuda_library_capabilities_are_queryable(cuda_runtime):
    jt = cuda_runtime
    capabilities = set(jt.core.backend_supported_capabilities("cuda"))
    assert {
        "segmented_arg_reduce", "segmented_argsort", "where", "random",
        "matmul", "conv2d", "conv2d_backward_input", "conv2d_backward_weight",
    } <= capabilities
    implementations = set(jt.core.backend_supported_ops("cuda"))
    assert {
        "cub_arg_reduce", "cub_argsort", "cub_where", "curand_random",
        "cublas_matmul", "cudnn_conv", "cudnn_conv_backward_x", "cudnn_conv_backward_w",
    } <= implementations


def test_cuda_random_capability_runs_generator(cuda_runtime):
    jt = cuda_runtime
    jt.set_seed(417)
    with jt.profile_scope() as report:
        samples = _device_array(jt.random((8193,), "float32", "uniform"))
    _assert_executed(report, "curand_random")
    assert np.isfinite(samples).all()
    assert np.all((samples >= 0) & (samples <= 1))
    assert abs(float(np.mean(samples)) - 0.5) < 0.03
    assert abs(float(np.var(samples)) - 1.0 / 12.0) < 0.01
    # NumPy and cuRAND do not share an RNG algorithm. Seed replay checks the
    # exact stream; NumPy checks distribution invariants above, not fake parity.
    jt.set_seed(417)
    np.testing.assert_array_equal(samples, _device_array(jt.random((8193,), "float32")))


def test_cuda_segmented_and_where_capabilities_match_numpy(cuda_runtime):
    jt = cuda_runtime
    data = np.array([[4.5, -2, 0.25, 1.5], [-3.5, 2.5, 0, 7]], dtype=np.float32)
    mask = data > 0
    x = jt.array(data)
    condition = jt.array(mask)
    jt.sync([x, condition])
    with jt.profile_scope() as report:
        indices, values = jt.ops.argsort(x, 1, False, "int32")
        max_indices, max_values = jt.ops.arg_reduce(x, "max", 1, False)
        coordinates = jt.ops.where(condition, "int64")
        actual_indices, actual_values = _device_array(indices), _device_array(values)
        actual_max_indices, actual_max_values = _device_array(max_indices), _device_array(max_values)
        actual_coordinates = tuple(_device_array(value) for value in coordinates)
    for implementation in ("cub_argsort", "cub_arg_reduce", "cub_where"):
        _assert_executed(report, implementation)
    np.testing.assert_array_equal(actual_indices, np.argsort(data, axis=1))
    np.testing.assert_array_equal(actual_values, np.sort(data, axis=1))
    np.testing.assert_array_equal(actual_max_indices, np.argmax(data, axis=1))
    np.testing.assert_array_equal(actual_max_values, np.max(data, axis=1))
    for actual, expected in zip(actual_coordinates, np.where(mask)):
        np.testing.assert_array_equal(actual, expected)
    assert len(actual_coordinates) == mask.ndim


def test_cuda_matmul_capability_relays_meta_operator_graph(cuda_runtime):
    jt = cuda_runtime
    a_data = (np.arange(24, dtype=np.float32).reshape(4, 6) - 9) / 16
    b_data = (np.arange(30, dtype=np.float32).reshape(6, 5) - 12) / 32
    a, b = jt.array(a_data), jt.array(b_data)
    jt.sync([a, b])
    with jt.profile_scope(compile_options={"_capability_matmul": 1}) as report:
        products = a.broadcast([4, 6, 5], [2]) * b.broadcast([4, 6, 5], [0])
        actual = _device_array(products.sum(1))
    np.testing.assert_allclose(actual, a_data @ b_data, rtol=1e-5, atol=1e-6)
    _assert_relay_executed(report, "cublas_matmul")


def test_cuda_conv_capability_relays_meta_operator_graph(cuda_runtime):
    jt = cuda_runtime
    x_data = (np.arange(60, dtype=np.float32).reshape(1, 2, 5, 6) - 25) / 32
    w_data = (np.arange(54, dtype=np.float32).reshape(3, 2, 3, 3) - 23) / 64
    expected = np.zeros((1, 3, 3, 4), dtype=np.float32)
    for kh in range(3):
        for kw in range(3):
            expected += np.einsum("nchw,oc->nohw", x_data[:, :, kh:kh + 3, kw:kw + 4],
                                  w_data[:, :, kh, kw])
    x, weight = jt.array(x_data), jt.array(w_data)
    jt.sync([x, weight])
    with jt.profile_scope(compile_options={"_capability_conv": 1}) as report:
        patches = x.reindex([1, 3, 2, 3, 4, 3, 3],
                            ["i0", "i2", "i3*1-0+i5*1", "i4*1-0+i6*1"])
        products = patches * weight.broadcast(patches.shape, [0, 3, 4])
        actual = _device_array(products.sum([2, 5, 6]))
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    _assert_relay_executed(report, "cudnn_conv")
