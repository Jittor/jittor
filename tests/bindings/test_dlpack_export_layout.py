"""DLPack export of every storage layout, checked by an external consumer.

A strided Var -- a step slice, a broadcast -- exports its own element strides
over the same bytes (copy=None/False), or a dense independent buffer
(copy=True); it used to be refused. CUDA external-consumer checks require
CuPy. The fixtures keep the whole base allocation alive.
"""

import gc

import jittor as jt
import numpy as np
import pytest

from _helpers import capability


@pytest.fixture(params=["cpu", "cuda"])
def export_device(request):
    device = request.param
    if (
        device == "cuda"
        and not capability.check_accelerator("cuda", backend=jt).enabled
    ):
        pytest.skip("CUDA is unavailable in this build")
    with jt.flag_scope(use_cuda=int(device == "cuda"), backend_fallback="error"):
        yield device


def _check_source_device(value, device):
    value.sync(device_sync=True, weak_sync=False)
    assert value.location() == ("device" if device == "cuda" else "cpu")
    expected = (2, value.device_id) if device == "cuda" else (1, 0)
    assert value.__dlpack_device__() == expected
    return expected


def _external_module(device):
    xp = pytest.importorskip("cupy") if device == "cuda" else np
    if not hasattr(xp, "from_dlpack"):
        pytest.skip(f"{xp.__name__} does not provide a from_dlpack consumer")
    return xp


def _check_external_device(value, device, device_id):
    if device == "cuda":
        assert value.device.id == device_id
    else:
        assert isinstance(value, np.ndarray)


_EXPORT_CASES = [
    pytest.param("function", False, None, id="function-classic"),
    pytest.param("method", False, None, id="method-classic-copy-none"),
    pytest.param("method", False, False, id="method-classic-copy-false"),
    pytest.param("method", False, True, id="method-classic-copy-true"),
    pytest.param("dunder", True, None, id="dunder-versioned-copy-none"),
    pytest.param("dunder", True, False, id="dunder-versioned-copy-false"),
    pytest.param("dunder", True, True, id="dunder-versioned-copy-true"),
    pytest.param("dunder", False, False, id="dunder-classic-copy-false"),
    pytest.param("method", True, True, id="method-versioned-copy-true"),
]


def _export(value, entrypoint, versioned, copy):
    if entrypoint == "function":
        return jt.to_dlpack(value)
    method = value.dlpack if entrypoint == "method" else value.__dlpack__
    if versioned:
        return method(max_version=(1, 0), copy=copy)
    return method(copy=copy)


class _Capsule:
    """Hand a ready capsule to a consumer that only takes __dlpack__ objects."""

    def __init__(self, capsule, device):
        self._capsule, self._device = capsule, device

    def __dlpack__(self, *args, **kwargs):
        capsule, self._capsule = self._capsule, None
        return capsule

    def __dlpack_device__(self):
        return self._device


@pytest.mark.parametrize("entrypoint,versioned,copy", _EXPORT_CASES)
@pytest.mark.parametrize("layout", ["slice", "broadcast"])
def test_strided_export_keeps_its_strides_or_copies(
    export_device, layout, entrypoint, versioned, copy
):
    dtype = "float64" if layout == "slice" else "complex128"
    data = np.arange(12).reshape(3, 4).astype(dtype)
    base = jt.array(data, dtype=dtype)
    view = base[:, ::2] if layout == "slice" else base[:, :1].broadcast((3, 4))
    expected = data[:, ::2] if layout == "slice" else np.broadcast_to(data[:, :1], (3, 4))
    device_type, device_id = _check_source_device(view, export_device)
    assert not view._storage_is_contiguous()
    strides = (4, 2) if layout == "slice" else (4, 0)
    assert tuple(view._storage_strides()) == strides
    address = view._storage_address

    capsule = _export(view, entrypoint, versioned, copy)
    itemsize = np.dtype(dtype).itemsize
    copied = copy is True and entrypoint != "function"
    if versioned:
        # NumPy before 2.1 reads only the classic capsule; Jittor's own
        # import aliases the exported layout and shows its strides.
        result = jt.from_dlpack(capsule)
        np.testing.assert_array_equal(result.numpy(), expected)
        if copied:
            assert result._storage_address != address
        else:
            assert result._storage_address == address
            assert tuple(result._storage_strides()) == strides
    else:
        xp = _external_module(export_device)
        result = xp.from_dlpack(_Capsule(capsule, (device_type, device_id)))
        _check_external_device(result, export_device, device_id)
        host = xp.asnumpy(result) if export_device == "cuda" else result
        np.testing.assert_array_equal(host, expected)
        pointer = result.data.ptr if device_type == 2 else result.__array_interface__["data"][0]
        if copied:
            assert pointer != address
        else:
            # The Var's own bytes, read through its own strides.
            assert pointer == address
            assert tuple(result.strides) == tuple(st * itemsize for st in strides)
    if copied:
        # An independent buffer: a write to the Var does not reach it.
        base[0, 0] = 99
        base.sync()
        got = result.numpy() if versioned else (
            xp.asnumpy(result) if export_device == "cuda" else result)
        np.testing.assert_array_equal(got, expected)
    _check_source_device(view, export_device)


@pytest.mark.parametrize("stream", [0, -2, 1.5, "1"])
def test_invalid_export_stream_is_rejected(stream):
    x = jt.array(np.arange(4, dtype="float32"))
    with pytest.raises(RuntimeError, match="stream"):
        x.__dlpack__(stream=stream)


def test_cpu_export_refuses_a_named_stream():
    with jt.flag_scope(use_cuda=0):
        x = jt.array(np.arange(4, dtype="float32"))
        x.sync()
        with pytest.raises(RuntimeError, match="on the CPU"):
            x.__dlpack__(stream=1)
        # None and -1 (no synchronization) are both fine on the CPU.
        for stream in (None, -1):
            np.testing.assert_array_equal(
                np.from_dlpack(_Capsule(x.__dlpack__(stream=stream), (1, 0))), [0, 1, 2, 3])


def test_contiguous_offset_export_keeps_external_alias(export_device):
    xp = _external_module(export_device)
    data = np.arange(12, dtype=np.float64).reshape(3, 4)
    source = xp.asarray(data)
    base = jt.from_dlpack(source)
    view = base[1:]
    device_type, device_id = _check_source_device(view, export_device)
    assert view._storage_is_contiguous()
    assert view._storage_offset() == 4
    address = view._storage_address

    result = xp.from_dlpack(view)
    _check_external_device(result, export_device, device_id)
    assert tuple(result.shape) == (2, 4)
    assert str(result.dtype) == "float64"
    pointer = (
        result.data.ptr if device_type == 2 else result.__array_interface__["data"][0]
    )
    assert pointer == address

    expected = data[1:].copy()
    source[1, 0] = 37
    expected[0, 0] = 37
    if export_device == "cuda":
        xp.cuda.get_current_stream().synchronize()
    del view, base, source
    gc.collect()
    _check_external_device(result, export_device, device_id)
    actual = xp.asnumpy(result) if export_device == "cuda" else result
    np.testing.assert_array_equal(actual, expected)


def test_explicit_contiguous_export_reaches_external_consumer(export_device):
    xp = _external_module(export_device)
    data = np.arange(12, dtype=np.float64).reshape(3, 4).astype(np.complex128)
    data += 2**-40 + (0.25 + 2**-39) * 1j
    base = jt.array(data, dtype="complex128")
    view = base[:, ::2]
    _check_source_device(view, export_device)
    assert not view._storage_is_contiguous()
    dense = view.contiguous()
    _, device_id = _check_source_device(dense, export_device)
    assert dense._storage_is_contiguous()
    assert dense._storage_address != view._storage_address

    result = xp.from_dlpack(dense)
    _check_external_device(result, export_device, device_id)
    assert tuple(result.shape) == (3, 2)
    assert str(result.dtype) == "complex128"
    del dense, view, base
    gc.collect()
    _check_external_device(result, export_device, device_id)
    actual = xp.asnumpy(result) if export_device == "cuda" else result
    np.testing.assert_array_equal(actual, data[:, ::2])


# The stream contract on a real device. CuPy hands its current stream to
# `__dlpack__(stream=)`: an export makes that stream wait for Jittor's work
# without blocking the host, and an import hands the producer Jittor's
# compute stream, so neither side waits for the whole device.
def _cupy_or_skip():
    if not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("CUDA is unavailable in this build")
    return pytest.importorskip("cupy")


def test_export_orders_a_consumer_stream_without_a_host_wait():
    cp = _cupy_or_skip()
    with jt.flag_scope(use_cuda=1, backend_fallback="error"):
        a = jt.array(np.random.default_rng(0).normal(size=(512, 512)).astype("float32"))
        expected = (a.numpy() @ a.numpy()).sum(axis=0)
        product = jt.matmul(a, a)          # pending until the export runs it
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            consumed = cp.from_dlpack(product)
            column_sums = consumed.sum(axis=0)
        stream.synchronize()
        np.testing.assert_allclose(cp.asnumpy(column_sums), expected, rtol=1e-3, atol=1e-2)


def test_export_stream_minus_one_skips_synchronization():
    cp = _cupy_or_skip()
    with jt.flag_scope(use_cuda=1, backend_fallback="error"):
        x = jt.array(np.arange(6, dtype="float32")) * 2
        capsule = x.__dlpack__(stream=-1)
        jt.sync_all(True)                  # the caller's own synchronization
        got = cp.from_dlpack(_Capsule(capsule, x.__dlpack_device__()))
        np.testing.assert_array_equal(cp.asnumpy(got), np.arange(6) * 2)


def test_import_hands_the_producer_jittors_stream():
    cp = _cupy_or_skip()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        source = cp.arange(1 << 20, dtype=cp.float32)
        for _ in range(8):                 # pending work on the producer's stream
            source = source * 1.0001 + 1
    with jt.flag_scope(use_cuda=1, backend_fallback="error"):
        imported = jt.from_dlpack(source)
        total = float(imported.sum().numpy())
    stream.synchronize()
    np.testing.assert_allclose(total, float(cp.asnumpy(source).sum()), rtol=1e-5)
    assert jt.core.dlpack_consumer_stream(source.device.id) not in (0, -1)


def test_second_gpu_round_trip_keeps_the_callers_device():
    # A CuPy caller inside `with cupy.cuda.Device(1)`: Jittor running a batch,
    # exporting or importing used to leave device 0 current, and CuPy then
    # refused its own device-1 stream. Imported data stays on device 1.
    cp = _cupy_or_skip()
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("needs two visible GPUs")
    with jt.flag_scope(use_cuda=1, backend_fallback="error"), cp.cuda.Device(1):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            source = cp.arange(12, dtype=cp.float32).reshape(3, 4) * 3
        imported = jt.from_dlpack(source)
        assert cp.cuda.runtime.getDevice() == 1
        assert imported.__dlpack_device__() == (2, 1)
        assert imported._storage_address == source.data.ptr
        shifted = imported + 1
        np.testing.assert_array_equal(shifted.numpy(), cp.asnumpy(source) + 1)
        assert cp.cuda.runtime.getDevice() == 1
        with stream:
            back = cp.from_dlpack(shifted)
            doubled = back * 2
        stream.synchronize()
        assert back.device.id == 1 and cp.cuda.runtime.getDevice() == 1
        np.testing.assert_array_equal(cp.asnumpy(doubled), (cp.asnumpy(source) + 1) * 2)


# A strided import is permuted back to its dense order for an elementwise op
# (layout propagation), which rebuilds the `1` as a broadcast of its own. That
# constant was made on the current device and stayed there next to an operand
# with an explicit placement on device 1, so the kernel on device 1 read a
# device-0 pointer and faulted with an illegal address.
_STRIDED_SECOND_GPU = """
import numpy as np, jittor as jt, cupy as cp
jt.flags.use_cuda = 1
a = jt.array(np.random.rand(64, 64).astype("float32"))
float(jt.matmul(a, a).sum().numpy())          # some work on device 0 first
b = jt.array(np.arange(12, dtype="float32").reshape(3, 4)).to_device(1)
with jt.flag_scope(transpose_storage_view=1):
    view = b.transpose((1, 0))
view.sync()
imported = jt.from_dlpack(view.__dlpack__())   # strided foreign memory on device 1
assert np.array_equal((imported + 1).numpy(), b.numpy().T + 1)
with cp.cuda.Device(1):
    source = cp.arange(12, dtype=cp.float32).reshape(3, 4)
    transposed = jt.from_dlpack(source.T)
assert np.array_equal((transposed * 2 + 1).numpy(), cp.asnumpy(source).T * 2 + 1)
print("STRIDED-SECOND-GPU-OK", flush=True)
"""


def test_strided_import_on_a_second_gpu_takes_python_scalars():
    from _helpers.child_process import run_child_script
    if not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("CUDA is unavailable in this build")
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("needs two visible GPUs")
    # In a child: a fault would poison the CUDA context of the process it hits.
    result = run_child_script(_STRIDED_SECOND_GPU, text=True, timeout=600,
                              merge_stderr=True, name="strided_second_gpu")
    assert "STRIDED-SECOND-GPU-OK" in result.stdout, result.stdout[-2000:]
