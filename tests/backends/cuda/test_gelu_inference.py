import numpy as np
import pytest

from _helpers.capability import require_accelerator
import jittor as jt
from jittor._runtime.dispatch import override_kernel, select_kernel


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_vectorized_gelu_matches_elementwise(dtype, approximate):
    require_accelerator("cuda")
    size = 8198 if dtype == "float32" else (1 << 20) + 6
    data = np.random.default_rng(83).normal(size=size).astype(np.float32)
    with jt.flag_scope(use_cuda=1, backend_fallback="error", cuda_kernel_math="strict"), jt.no_grad():
        base = jt.array(data).cast(dtype)
        base.sync()
        # The aligned slice has a tail. The offset slice takes the scalar load.
        for x in (base[:-1], base[1:]):
            x.sync()
            op = "nn.gelu.inference"
            assert select_kernel(op, x, approximate=approximate) is not None
            out = jt.nn.gelu(x, approximate=approximate)
            assert out._producer_op() == "code"
            out.sync()
            assert out.location() == "device" and not out.requires_grad
            with override_kernel(op, "cuda", None):
                expected = jt.nn.gelu(x, approximate=approximate)
            np.testing.assert_array_equal(out.float32().numpy(), expected.float32().numpy())


@pytest.mark.cuda
def test_gelu_selects_matmul_boundary_and_leaves_elementwise_chain_fusable():
    require_accelerator("cuda")
    with jt.flag_scope(use_cuda=1, backend_fallback="error"), jt.no_grad():
        x = jt.ones((64, 37))
        y = jt.ones((37, 128))
        product = jt.nn.matmul(x, y)
        elementwise = (product + 0.25).reshape(2, 32, 128)
        op = "nn.gelu.inference"
        assert not product.is_finished
        assert select_kernel(op, product) is not None
        assert select_kernel(op, elementwise) is None
        assert select_kernel(op, product.transpose(0, 1)) is None
        narrow = jt.nn.matmul(x.bfloat16(), y.bfloat16())
        assert not narrow.is_finished
        assert select_kernel(op, narrow, approximate="tanh") is None
    assert select_kernel(op, jt.ones((8193,)).start_grad()) is None
