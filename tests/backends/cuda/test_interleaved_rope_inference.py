import numpy as np
import pytest

from _helpers.capability import require_accelerator
import jittor as jt
from jittor._runtime.dispatch import try_dispatch


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "sliced"])
def test_interleaved_rope_exact(dtype, layout):
    require_accelerator("cuda")
    rng = np.random.default_rng(12)
    shape = (2, 3, 29, 14)
    storage_shape = {"contiguous": shape, "transposed": (2, 29, 3, 14),
                     "sliced": (2, 3, 29, 15)}[layout]
    values = rng.normal(size=storage_shape).astype(np.float32)
    cos = rng.normal(size=(1, 29, 14)).astype(np.float32)
    sin = rng.normal(size=(2, 1, 1, 14)).astype(np.float32)
    with jt.flag_scope(use_cuda=1, backend_fallback="error", cuda_kernel_math="strict"), jt.no_grad():
        base = jt.array(values).cast(dtype)
        if layout == "transposed":
            x = base.transpose(0, 2, 1, 3)
        elif layout == "sliced":
            x = base[..., 1:]
        else:
            x = base
        out = try_dispatch("nn.interleaved_rope.inference", x, jt.array(cos), jt.array(sin))
        assert out is not None
        out.sync()
        assert out.location() == "device" and not out.requires_grad

        widened = base.float32().numpy().reshape(storage_shape)
        if layout == "transposed":
            widened = widened.transpose(0, 2, 1, 3)
        elif layout == "sliced":
            widened = widened[..., 1:]
        rotated = np.empty_like(widened)
        rotated[..., ::2] = -widened[..., 1::2]
        rotated[..., 1::2] = widened[..., ::2]
        expected = jt.array(widened * cos + rotated * sin).cast(dtype)
        np.testing.assert_array_equal(out.float32().numpy(), expected.float32().numpy())


@pytest.mark.cuda
def test_interleaved_rope_declines_gradient_and_unsupported_broadcast():
    require_accelerator("cuda")
    with jt.flag_scope(use_cuda=1):
        x = jt.ones((2, 3, 14)).start_grad()
        cos, sin = (jt.ones((3, 14)).stop_grad() for _ in range(2))
        op = "nn.interleaved_rope.inference"
        assert try_dispatch(op, x, cos, sin) is None
        with jt.no_grad():
            assert try_dispatch(op, x, cos, sin[:, :-2]) is None
