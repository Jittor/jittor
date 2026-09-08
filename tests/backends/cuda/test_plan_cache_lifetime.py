"""Real two-device cache reuse, explicit cleanup and reconstruction contracts."""
import ctypes

import numpy as np
import pytest

import jittor as jt
from jittor.nn.legacy_complex import _fft2


def _cuda_device():
    value = ctypes.c_int()
    function = ctypes.CDLL(None).cudaGetDevice
    function.argtypes = [ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_int
    assert function(ctypes.byref(value)) == 0
    return value.value


@pytest.fixture
def cache_libraries():
    if not jt.has_cuda or jt.get_device_count() < 2:
        pytest.skip("two CUDA devices are required for plan ownership")
    cufft, cudnn = jt.cufft, jt.cudnn
    assert cufft is not None and cudnn is not None
    benchmark = cudnn.get_benchmark()
    cufft.cufft_clear_plan_cache()
    cudnn.cudnn_clear_plan_cache()
    cudnn.cudnn_clear_algorithm_cache()
    cudnn.set_benchmark(0)
    try:
        yield cufft, cudnn
    finally:
        jt.sync_all(True)
        cufft.cufft_clear_plan_cache()
        cufft.cufft_set_plan_cache_size(32)
        cudnn.cudnn_clear_plan_cache()
        cudnn.cudnn_clear_algorithm_cache()
        cudnn.set_algorithm_cache_size(100)
        cudnn.set_benchmark(benchmark)


def _fft(device, size, seed=0, clear_while_pending=False):
    real = np.random.RandomState(seed).normal(size=(1, size, size)).astype("float32")
    source = np.stack((real, np.zeros_like(real)), axis=-1)
    with jt.flag_scope(use_cuda=1, device_id=device):
        value = jt.array(source)
        output = _fft2(value)
        output.sync()  # Submission does not complete the queued device work.
        if clear_while_pending:
            jt.cufft.cufft_clear_plan_cache(device)
        actual = output.numpy()
    reference = np.fft.fft2(real, axes=(1, 2))
    expected = np.stack((reference.real, reference.imag), axis=-1)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_cufft_same_geometry_has_one_reusable_plan_per_device(cache_libraries):
    cufft, _ = cache_libraries
    before = [cufft.cufft_plan_build_count(device) for device in (0, 1)]
    for index, device in enumerate((0, 1, 0, 1)):
        _fft(device, 11, index)
    assert [cufft.cufft_plan_cache_size(device) for device in (0, 1)] == [1, 1]
    assert [cufft.cufft_plan_build_count(device) for device in (0, 1)] == [n + 1 for n in before]
    with jt.flag_scope(device_id=1):
        destroyed = cufft.cufft_plan_destroy_count(0)
        cufft.cufft_clear_plan_cache(0)
        assert _cuda_device() == jt.current_device() == 1
        assert cufft.cufft_plan_cache_size(0) == 0
        assert cufft.cufft_plan_cache_size(1) == 1
        assert cufft.cufft_plan_destroy_count(0) == destroyed + 1
        cufft.cufft_clear_plan_cache(0)
        assert cufft.cufft_plan_destroy_count(0) == destroyed + 1
    _fft(0, 11, 8)
    assert cufft.cufft_plan_build_count(0) == before[0] + 2
    assert cufft.cufft_plan_destroy_failures() == 0


def test_cufft_eviction_budget_is_local_and_clear_waits_for_plan_consumers(cache_libraries):
    cufft, _ = cache_libraries
    cufft.cufft_set_plan_cache_size(1)
    before = [cufft.cufft_plan_destroy_count(device) for device in (0, 1)]
    _fft(0, 7)
    _fft(1, 7)
    _fft(0, 9)
    assert [cufft.cufft_plan_cache_size(device) for device in (0, 1)] == [1, 1]
    assert cufft.cufft_plan_destroy_count(0) == before[0] + 1
    assert cufft.cufft_plan_destroy_count(1) == before[1]
    _fft(1, 19, clear_while_pending=True)
    assert cufft.cufft_plan_cache_size(1) == 0
    with jt.flag_scope(device_id=1):
        cufft.cufft_clear_plan_cache()
        assert _cuda_device() == jt.current_device() == 1
    assert cufft.cufft_plan_cache_size() == 0
    assert cufft.cufft_plan_destroy_failures() == 0


def _convolution_reference(x, w):
    spatial = tuple(a - b + 1 for a, b in zip(x.shape[2:], w.shape[2:]))
    output = np.zeros((x.shape[0], w.shape[0], *spatial), dtype=x.dtype)
    dx, dw = np.zeros_like(x), np.zeros_like(w)
    for batch in range(x.shape[0]):
        for channel in range(w.shape[0]):
            for position in np.ndindex(spatial):
                region = (batch, slice(None)) + tuple(slice(p, p + k) for p, k in zip(position, w.shape[2:]))
                output[(batch, channel, *position)] = np.sum(x[region] * w[channel])
                dx[region] += w[channel]
                dw[channel] += x[region]
    return output, dx, dw


def _convolution(device, dimensions, dtype, gradients):
    x_shape = (1, 1) + (4,) * dimensions
    w_shape = (2, 1) + (2,) * dimensions
    x_np = (np.arange(np.prod(x_shape)).reshape(x_shape) / 17).astype(dtype)
    w_np = (np.arange(np.prod(w_shape)).reshape(w_shape) / 23).astype(dtype)
    with jt.flag_scope(use_cuda=1, device_id=device):
        x, w = jt.array(x_np, dtype=dtype), jt.array(w_np, dtype=dtype)
        if dimensions == 2:
            y = jt.cudnn.ops.cudnn_conv(x, w, 1, 1, 0, 0)
        else:
            y = jt.cudnn.ops.cudnn_conv3d(x, w, 1, 1, 1, 0, 0, 0)
        results = (y,) + tuple(jt.grad(y.sum(), [x, w])) if gradients else (y,)
        actual = [value.numpy() for value in results]
    for got, expected in zip(actual, _convolution_reference(x_np, w_np)):
        tolerance = 2e-5 if dtype == "float32" else 1e-10
        np.testing.assert_allclose(got, expected, rtol=tolerance, atol=tolerance)


def test_cudnn_six_legacy_paths_use_device_owned_algorithm_tables(cache_libraries):
    _, cudnn = cache_libraries
    cudnn.set_algorithm_cache_size(8)
    for device in (0, 1):
        for dimensions in (2, 3):
            # Double explicitly selects the legacy route in the 2-D backend.
            _convolution(device, dimensions, "float64", gradients=True)
        assert cudnn.cudnn_algorithm_cache_size(device) == 6
    with jt.flag_scope(device_id=1):
        cudnn.cudnn_clear_algorithm_cache(0)
        assert _cuda_device() == jt.current_device() == 1
    assert cudnn.cudnn_algorithm_cache_size(0) == 0
    assert cudnn.cudnn_algorithm_cache_size(1) == 6
    _convolution(0, 2, "float64", gradients=True)
    assert cudnn.cudnn_algorithm_cache_size(0) == 3


def test_cudnn_backend_plans_clear_one_device_and_rebuild(cache_libraries):
    _, cudnn = cache_libraries
    for device in (0, 1):
        _convolution(device, 2, "float32", gradients=True)
        assert cudnn.cudnn_plan_cache_size(device) == 3
    with jt.flag_scope(device_id=1):
        destroyed = cudnn.cudnn_plan_destroy_count(0)
        cudnn.cudnn_clear_plan_cache(0)
        assert _cuda_device() == jt.current_device() == 1
        assert cudnn.cudnn_plan_cache_size(0) == 0
        assert cudnn.cudnn_plan_cache_size(1) == 3
        assert cudnn.cudnn_plan_destroy_count(0) == destroyed + 3
        cudnn.cudnn_clear_plan_cache(0)
        assert cudnn.cudnn_plan_destroy_count(0) == destroyed + 3
    _convolution(0, 2, "float32", gradients=True)
    assert cudnn.cudnn_plan_cache_size(0) == 3
    with jt.flag_scope(device_id=0):
        cudnn.cudnn_clear_plan_cache()
        assert _cuda_device() == jt.current_device() == 0
    assert cudnn.cudnn_plan_cache_size() == 0


def test_cutt_plan_banks_own_devices_and_survive_targeted_clear(cache_libraries):
    from _helpers.cutt import require_cutt_ops
    operations = require_cutt_ops()
    cutt = jt.compile_extern.cutt
    cutt.cutt_clear_plan_cache()
    cutt.cutt_set_plan_cache_size(1)
    before = [cutt.cutt_plan_build_count(device) for device in (0, 1)]
    try:
        for device, size in ((0, 5), (1, 5), (0, 5), (1, 5)):
            data = np.arange(size * 7 * 3, dtype="float32").reshape(size, 7, 3)
            with jt.flag_scope(use_cuda=1, device_id=device):
                actual = operations.cutt_transpose(jt.array(data), (2, 0, 1)).numpy()
                np.testing.assert_array_equal(actual, data.transpose(2, 0, 1))
        assert [cutt.cutt_plan_cache_size(device) for device in (0, 1)] == [1, 1]
        assert [cutt.cutt_plan_build_count(device) for device in (0, 1)] == [n + 1 for n in before]
        with jt.flag_scope(device_id=1):
            destroyed = cutt.cutt_plan_destroy_count(0)
            cutt.cutt_clear_plan_cache(0)
            assert _cuda_device() == jt.current_device() == 1
            assert cutt.cutt_plan_cache_size(0) == 0
            assert cutt.cutt_plan_cache_size(1) == 1
            assert cutt.cutt_plan_destroy_count(0) == destroyed + 1
            cutt.cutt_clear_plan_cache(0)
            assert cutt.cutt_plan_destroy_count(0) == destroyed + 1
        with jt.flag_scope(device_id=0):
            output = operations.cutt_transpose(jt.array(data), (2, 0, 1))
            output.sync()
            cutt.cutt_clear_plan_cache(0)
            np.testing.assert_array_equal(output.numpy(), data.transpose(2, 0, 1))
        assert cutt.cutt_plan_build_count(0) == before[0] + 2
        assert cutt.cutt_plan_destroy_failures() == 0
    finally:
        cutt.cutt_clear_plan_cache()
        cutt.cutt_set_plan_cache_size(64)
