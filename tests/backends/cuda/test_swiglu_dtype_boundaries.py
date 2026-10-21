"""CUDA SwiGLU preserves the intermediate public SiLU dtype boundary."""
import jittor as jt
import numpy as np
import pytest

from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.backends.cuda.kernels.nn.swiglu_cuda import _silu_and_mul_cuda


def _round(value, dtype):
    if dtype == "float32":
        return np.asarray(value, dtype=np.float32).astype(np.float64)
    if dtype == "float16":
        return np.asarray(value, dtype=np.float16).astype(np.float64)
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32).astype(np.float64)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_fused_swiglu_keeps_silu_rounding_and_autograd_admission(dtype):
    available = capability.check_accelerator("cuda", backend=jt)
    if not available.enabled:
        pytest.skip(available.reason)
    gate = np.array([-2.75, -1.25, -.5, -.125, .125, .5, 1.25, 2.75])
    up = np.array([.3, -1.7, .9, 2.4, -2.4, .9, 1.7, -.3])
    data = np.stack((np.concatenate((gate, up)), np.concatenate((-gate, up[::-1]))))
    quantized = _round(data, dtype)
    gate, up = np.split(quantized, 2, axis=-1)
    activated = gate / (1. + np.exp(-gate))
    expected = _round(_round(activated, dtype) * up, dtype)
    if dtype != "float32":
        # This fixture must distinguish a single final cast from SiLU's
        # intermediate cast, rather than accepting both implementations.
        assert np.any(expected != _round(activated * up, dtype))
    with jt.flag_scope(use_cuda=1, amp_reg=0, cuda_kernel_math="strict"), forbid_backend_fallbacks():
        trainable = jt.array(data.astype("float32")).cast(dtype).start_grad()
        assert _silu_and_mul_cuda(trainable) is None
        frozen = jt.array(data.astype("float32")).cast(dtype).stop_grad()
        outputs = [_silu_and_mul_cuda(frozen)]
        with jt.no_grad():
            outputs.append(_silu_and_mul_cuda(trainable))
        for output in outputs:
            assert output is not None and output.is_stop_grad()
            assert str(output.dtype) == dtype
            output.sync()
            assert output.location() == "device"
            observed = output.float32().numpy()
            if dtype == "float32":
                np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-7)
            else:
                np.testing.assert_array_equal(observed, expected)
