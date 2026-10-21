"""DLPack export rejects unsupported layouts instead of mislabelling storage.

CUDA external-consumer checks require CuPy; rejection checks do not.  The
noncontiguous fixtures deliberately retain a backing allocation large enough
for the old dense memcpy, so the pre-fix rejection tests cannot overread it.
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


@pytest.mark.parametrize("entrypoint,versioned,copy", _EXPORT_CASES)
@pytest.mark.parametrize("layout", ["slice", "broadcast"])
def test_noncontiguous_export_is_rejected(
    export_device, layout, entrypoint, versioned, copy
):
    dtype = "float64" if layout == "slice" else "complex128"
    base = jt.array(np.arange(12).reshape(3, 4), dtype=dtype)
    # The broadcast only aliases the first column, but base keeps all twelve
    # elements alive. Its logical twelve-element size therefore also fits in
    # the underlying allocation if the old copy=True path is reached.
    view = base[:, ::2] if layout == "slice" else base[:, :1].broadcast((3, 4))
    _check_source_device(view, export_device)
    assert not view._storage_is_contiguous()
    assert tuple(view._storage_strides()) == ((4, 2) if layout == "slice" else (4, 0))
    assert view._storage_address == base._storage_address
    assert tuple(base.shape) == (3, 4)

    # The guard belongs after sync in shared resolve_export(), before
    # make_materialized_copy() and capsule allocation. Never consume an
    # incorrectly accepted capsule: rejection itself is the contract here.
    with pytest.raises(RuntimeError, match=r"non.contiguous.*\.contiguous\(\)"):
        _export(view, entrypoint, versioned, copy)
    _check_source_device(view, export_device)


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
