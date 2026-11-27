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
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("shape", [(2, 7, 7, 11), (1, 19, 17, 19)])
def test_channels_last_spatial_mean_and_global_pool(dtype, shape):
    require_accelerator("cuda")
    n, c, h, w = shape
    features = np.random.default_rng(196).integers(-32, 33, (n, h, w, c)).astype("float32") / 16
    expected = features.astype("float64").mean((1, 2))
    with jt.flag_scope(use_cuda=1, backend_fallback="error", amp_reg=0,
                       auto_flush_ops=0), jt.no_grad():
        x = jt.array(features).cast(dtype)._storage_permute((0, 3, 1, 2))
        results = (x.mean((-1, -2)), jt.mean(x, [2, 3], keepdims=True),
                   jt.nn.adaptive_avg_pool2d(x, 1))
        assert not x._storage_is_contiguous()
        tolerance = {"float32": 2e-7, "float16": 5e-4, "bfloat16": 4e-3}[dtype]
        assert all(result._producer_op() == "code" for result in results)
        jt.sync(results)
        for result, result_shape in zip(results, ((n, c), (n, c, 1, 1), (n, c, 1, 1))):
            assert tuple(result.shape) == result_shape
            assert str(result.dtype) == dtype
            assert result.location() == "device"
            np.testing.assert_allclose(result.float32().numpy().reshape(n, c), expected,
                                       atol=1e-7, rtol=tolerance)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_dense_low_precision_spatial_mean(dtype):
    require_accelerator("cuda")
    features = np.arange(2 * 16 * 12 * 20, dtype=np.int32).reshape(2, 16, 12, 20) % 37 - 18
    expected = features.mean((2, 3))
    with jt.flag_scope(use_cuda=1, backend_fallback="error", amp_reg=0,
                       auto_flush_ops=0), jt.no_grad():
        x = jt.array(features.astype("float32")).cast(dtype)
        result = x.mean((2, 3))
        assert result._producer_op() == "code"
        result.sync()
        assert result.location() == "device" and str(result.dtype) == dtype
        np.testing.assert_allclose(result.float32().numpy(), expected, atol=1e-7,
                                   rtol=4e-3 if dtype == "bfloat16" else 5e-4)


@pytest.mark.cuda
def test_channels_last_spatial_mean_training_and_sliced_fallbacks():
    require_accelerator("cuda")
    features = np.arange(2 * 7 * 19 * 11, dtype=np.float32).reshape(2, 7, 19, 11) / 32
    with jt.flag_scope(use_cuda=1, backend_fallback="error", amp_reg=0):
        source = jt.array(features).float16().start_grad()
        x = source._storage_permute((0, 3, 1, 2))
        result = jt.nn.adaptive_avg_pool2d(x, (1, 1))
        assert result.requires_grad and result._producer_op() != "code"
        gradient = jt.grad(result.sum(), source)
        gradient.sync()
        assert gradient.location() == "device"
        np.testing.assert_array_equal(gradient.numpy(), np.full(source.shape, 1 / 133, dtype="float16"))

        sliced = x[:, :, :, 1::2].detach()
        mean = sliced.mean((2, 3))
        assert mean._producer_op() != "code"
        mean.sync()
        assert mean.location() == "device"
        expected = features[:, :, 1::2, :].mean((1, 2)).astype("float16")
        np.testing.assert_allclose(mean.numpy(), expected, atol=0, rtol=5e-4)


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
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("size", [256, 2051, 7])
def test_silu_vectorized_frozen_dtypes_and_tails(dtype, size):
    require_accelerator("cuda")
    raw = np.linspace(-6.0, 6.0, size, dtype="float32")
    with jt.flag_scope(use_cuda=1, backend_fallback="error", amp_reg=0):
        x = jt.array(raw).cast(dtype).stop_grad()
        x.sync()
        xf = x.float32().numpy().astype("float64")
        expected = xf / (1.0 + np.exp(-xf))
        out = jt.nn.silu(x)
        out.sync()
        assert out.location() == "device" and str(out.dtype) == dtype
        tol = 1e-6 if dtype == "float32" else (5e-3 if dtype == "float16" else 2e-2)
        np.testing.assert_allclose(out.float32().numpy(), expected, atol=tol, rtol=tol)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_silu_keeps_pending_elementwise_and_channels_last_paths_fusible(dtype):
    require_accelerator("cuda")
    with jt.flag_scope(use_cuda=1, backend_fallback="error", amp_reg=0,
                       auto_flush_ops=0), jt.no_grad():
        x = jt.full((2, 17, 17, 16), 1, dtype=dtype)
        x.sync()
        materialized = jt.nn.silu(x)
        pending = jt.nn.silu(x + 0.5)
        nhwc = jt.nn.silu(x._storage_permute((0, 3, 1, 2)))
        assert materialized._producer_op() == "code"
        assert pending._producer_op() != "code"
        assert not nhwc._storage_is_contiguous()
        assert tuple(nhwc._storage_strides()) == (4624, 1, 272, 16)
        jt.sync([materialized, pending, nhwc])
        for output, value in ((materialized, 1), (pending, 1.5), (nhwc, 1)):
            assert output.location() == "device" and str(output.dtype) == dtype
            np.testing.assert_allclose(output.float32().numpy(), value / (1 + np.exp(-value)),
                                       atol=0, rtol=5e-3 if dtype == "bfloat16" else 5e-4)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_training_vectorized_forward_and_backward(dtype, approximate):
    require_accelerator("cuda")
    import math as _m
    raw = np.linspace(-3.5, 3.5, 313, dtype="float32")
    dy_raw = np.linspace(-1.2, 1.4, 313, dtype="float32")
    with jt.flag_scope(use_cuda=1, backend_fallback="error", amp_reg=0):
        x = jt.array(raw).cast(dtype).start_grad()
        dy = jt.array(dy_raw).cast(dtype)
        jt.sync([x, dy])
        xf = x.float32().numpy().astype("float64")
        dyf = dy.float32().numpy().astype("float64")
        y = jt.nn.gelu(x, approximate=approximate)
        gx = jt.grad((y * dy).sum(), x)
        jt.sync([y, gx])
        assert y.location() == "device" and str(y.dtype) == dtype
        assert gx.location() == "device" and str(gx.dtype) == dtype
        if approximate == "none":
            erf_v = np.vectorize(_m.erf)(xf * 0.7071067811865476)
            ref_y = 0.5 * xf * (1.0 + erf_v)
            ref_gx = dyf * (0.5 * (1.0 + erf_v) + xf * 0.3989422804014327 * np.exp(-0.5 * xf * xf))
        else:
            c = 0.7978845608028654
            t = np.tanh(c * (xf + 0.044715 * xf**3))
            ref_y = 0.5 * xf * (1.0 + t)
            ref_gx = dyf * (0.5 * (1.0 + t) + 0.5 * xf * (1.0 - t * t) * c * (1.0 + 0.134145 * xf * xf))
        tol = 2e-5 if dtype == "float32" else (1e-2 if dtype == "float16" else 4e-2)
        np.testing.assert_allclose(y.float32().numpy(), ref_y, atol=tol, rtol=tol)
        np.testing.assert_allclose(gx.float32().numpy(), ref_gx, atol=tol, rtol=tol)


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
