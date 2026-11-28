import numpy as np
import pytest

from _helpers.capability import require_accelerator
import jittor as jt
from jittor._runtime.dispatch import override_kernel, select_kernel


@pytest.mark.cuda
@pytest.mark.parametrize(("shape", "size", "align_corners"), [
    ((1, 9, 7, 7), (96, 96), False),
    ((2, 5, 6, 13), (81, 83), True),
    ((1, 16, 1, 1), (64, 64), False),
])
def test_separable_bicubic_preserves_fp32_order(shape, size, align_corners):
    require_accelerator("cuda")
    data = np.random.default_rng(83).normal(size=shape).astype(np.float32)
    data.reshape(-1)[:3] = (np.nan, np.inf, -np.inf)
    with jt.flag_scope(use_cuda=1, cuda_kernel_math="strict", backend_fallback="error"), jt.no_grad():
        source = jt.array(data)
        source.sync()
        params = (source, size, "bicubic", align_corners)
        assert select_kernel("nn.resize", *params, False) is not None
        actual = jt.nn.resize(*params)
        assert actual._producer_op() == "code"
        actual.sync()
        assert actual.location() == "device" and not actual.requires_grad
        with override_kernel("nn.resize", "cuda", None):
            reference = jt.nn.resize(*params)
        np.testing.assert_array_equal(actual.numpy(), reference.numpy())


@pytest.mark.cuda
def test_bicubic_declines_training_and_strided_input():
    require_accelerator("cuda")
    with jt.flag_scope(use_cuda=1, transpose_storage_view=1, backend_fallback="error"):
        source = jt.ones((1, 32, 5, 7)).start_grad()
        assert select_kernel("nn.resize", source, (64, 64), "bicubic", False, False) is None
        with jt.no_grad():
            view = source.transpose(2, 3)
            assert select_kernel("nn.resize", view, (64, 64), "bicubic", False, False) is None
