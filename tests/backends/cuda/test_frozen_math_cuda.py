"""Frozen FP32 features retain accurate values and real CUDA placement."""

import numpy as np
import pytest

import jittor as jt
from _helpers.capability import require_accelerator


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(64, 32, 112, 112), (64, 16, 56, 56),
                                   (64, 24, 28, 28), (64, 40, 14, 14),
                                   (64, 1280, 7, 7), (2, 16, 7, 7)])
def test_spatial_mean_method_and_root_match_fp64(shape):
    require_accelerator("cuda")
    data = np.random.default_rng(193).normal(0, .3, shape).astype("float32")
    expected = data.astype("float64").mean((2, 3), keepdims=True)
    with jt.flag_scope(use_cuda=1, backend_fallback="error"):
        x = jt.array(data).stop_grad()
        method = x.mean([-2, -1], True)
        root = jt.mean(x, dims=[2, 3], keepdims=True)
        for result in (method, root):
            result.sync()
            assert result.location() == "device"
            assert not result.requires_grad
            np.testing.assert_allclose(result.numpy(), expected, atol=1e-7, rtol=1e-5)
        np.testing.assert_array_equal(method.numpy(), root.numpy())


@pytest.mark.cuda
def test_spatial_mean_preserves_trainable_and_float64_paths():
    require_accelerator("cuda")
    data = np.random.default_rng(194).normal(size=(2, 16, 7, 7)).astype("float32")
    with jt.flag_scope(use_cuda=1, backend_fallback="error"):
        x = jt.array(data).start_grad()
        out = x.mean([2, 3])
        gradient = jt.grad(out.sum(), x)
        gradient.sync()
        assert gradient.location() == "device"
        np.testing.assert_allclose(gradient.numpy(), np.full(data.shape, 1 / 49),
                                   atol=1e-7, rtol=1e-6)
        wide = jt.array(data, dtype="float64").stop_grad().mean([2, 3])
        wide.sync()
        assert wide.location() == "device" and str(wide.dtype) == "float64"
        np.testing.assert_allclose(wide.numpy(), data.astype("float64").mean((2, 3)),
                                   atol=1e-12, rtol=1e-12)


@pytest.mark.cuda
def test_silu_frozen_forward_and_differentiable_fallback():
    require_accelerator("cuda")
    data = np.linspace(-12, 12, 1024, dtype="float32").reshape(4, 16, 4, 4)
    sigmoid = 1 / (1 + np.exp(-data.astype("float64")))
    with jt.flag_scope(use_cuda=1, backend_fallback="error"):
        frozen = jt.nn.silu(jt.array(data).stop_grad())
        frozen.sync()
        assert frozen.location() == "device" and not frozen.requires_grad
        np.testing.assert_allclose(frozen.numpy(), data * sigmoid, atol=1e-6, rtol=2e-6)
        x = jt.array(data).start_grad()
        gradient = jt.grad(jt.nn.silu(x).sum(), x)
        gradient.sync()
        assert gradient.location() == "device"
        np.testing.assert_allclose(gradient.numpy(), sigmoid * (1 + data * (1 - sigmoid)),
                                   atol=1e-6, rtol=1e-5)


@pytest.mark.cuda
def test_frozen_depthwise_selection_keeps_full_precision():
    require_accelerator("cuda")
    from jittor._runtime.dispatch import select_kernel
    from jittor.nn.modules import depthwise
    rng = np.random.default_rng(195)
    data = rng.normal(0, .1, (2, 4, 5, 5)).astype("float32")
    weight = rng.normal(0, .1, (4, 1, 3, 3)).astype("float32")
    padded = np.pad(data.astype("float64"), ((0, 0), (0, 0), (1, 1), (1, 1)))
    expected = np.empty(data.shape, dtype="float64")
    for i in range(5):
        for j in range(5):
            expected[:, :, i, j] = (padded[:, :, i:i+3, j:j+3]
                                    * weight[:, 0].astype("float64")).sum((-1, -2))
    with jt.flag_scope(use_cuda=1, backend_fallback="error",
                       float32_cudnn_precision="highest", cuda_allow_cudnn_tf32=0):
        x, w = jt.array(data).stop_grad(), jt.array(weight).stop_grad()
        selected = select_kernel("conv2d", x, w, None, (1, 1), (1, 1), (1, 1), 4,
                                 _depthwise_fast_path=True)
        assert selected is depthwise._depthwise_conv2d
        output = jt.nn.conv2d(x, w, padding=1, groups=4)
        output.sync()
        assert output.location() == "device"
        np.testing.assert_allclose(output.numpy(), expected, atol=1e-7, rtol=1e-5)
        x.start_grad()
        selected = select_kernel("conv2d", x, w, None, (1, 1), (1, 1), (1, 1), 4,
                                 _depthwise_fast_path=True)
        assert selected is depthwise._depthwise_conv2d
