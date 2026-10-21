"""Complex DLPack widths, ownership and residency without strided buffers."""

import gc

import numpy as np
import pytest
import jittor as jt

from _helpers import capability


@pytest.fixture(params=["cpu", "cuda"])
def complex_device(request):
    device = request.param
    if (
        device == "cuda"
        and not capability.check_accelerator("cuda", backend=jt).enabled
    ):
        pytest.skip("CUDA is unavailable in this build")
    with jt.flag_scope(use_cuda=int(device == "cuda")):
        yield device


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("versioned", [False, True])
@pytest.mark.parametrize("scalar", [False, True])
def test_complex_capsule_roundtrip(complex_device, dtype, versioned, scalar):
    expected = np.asarray(1 + 2**-40 + (2 + 2**-39) * 1j, dtype=dtype)
    if not scalar:
        expected = np.asarray([expected, -expected], dtype=dtype)
    source = jt.array(expected)
    source.sync()
    location = "device" if complex_device == "cuda" else "cpu"
    assert source.location() == location
    capsule = source.dlpack(max_version=(1, 0)) if versioned else source.dlpack()
    result = jt.from_dlpack(capsule)
    with pytest.raises(Exception, match="consumed|used|capsule"):
        jt.from_dlpack(capsule)
    del source, capsule
    gc.collect()
    result.sync()
    assert result.location() == location
    assert str(result.dtype) == dtype
    assert tuple(result.shape) == expected.shape
    np.testing.assert_array_equal(result.numpy(), expected)


def test_numpy_complex128_producer():
    with jt.flag_scope(use_cuda=0):
        expected = np.asarray([1 + 2**-40 + (2 + 2**-39) * 1j], dtype=np.complex128)
        result = jt.from_dlpack(expected)
        result.sync()
        assert result.location() == "cpu"
        assert str(result.dtype) == "complex128"
        np.testing.assert_array_equal(result.numpy(), expected)
