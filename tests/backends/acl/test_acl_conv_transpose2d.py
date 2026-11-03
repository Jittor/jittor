"""310P inference transpose convolution and unchanged ordinary-conv gradients."""

import numpy as np
import pytest

import jittor as jt

from _helpers import capability
from jittor._runtime.dispatch import select_kernel
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.backends.acl.kernels import neural
from jittor.backends.acl.kernels._soc import soc_name
from jittor.backends.acl.kernels.ops.conv_op import _conv_program, _conv_transpose_program


@pytest.fixture(autouse=True)
def acl_device():
    if not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required")
    if not soc_name().startswith("Ascend310P"):
        pytest.skip("transpose inference admission is validated on Ascend310P only")
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=1, use_acl=1, acl_allow_hf32=1,
                          transpose_storage_view=1), forbid_backend_fallbacks():
        yield
    assert jt.core.backend_fallback_count() == before


GEOMETRIES = (
    ((1, 4, 3, 4), (4, 3, 3, 3), (1, 1), (0, 0), (0, 0), (1, 1), 1, False),
    ((2, 4, 3, 4), (4, 3, 3, 3), (2, 2), (1, 1), (1, 1), (1, 1), 1, True),
    ((1, 4, 3, 4), (4, 3, 2, 3), (2, 3), (0, 1), (1, 2), (1, 1), 1, True),
    ((1, 4, 3, 4), (4, 3, 3, 2), (2, 1), (2, 1), (1, 0), (2, 2), 1, False),
    ((1, 4, 3, 4), (4, 3, 3, 3), (2, 2), (1, 1), (1, 0), (1, 1), 2, True),
    ((1, 4, 3, 4), (4, 1, 3, 3), (2, 2), (1, 1), (1, 1), (1, 1), 4, True),
    ((1, 8, 7, 7), (8, 8, 4, 4), (4, 4), (0, 0), (0, 0), (1, 1), 1, True),
)


def reference(source, weight, bias, stride, padding, output_padding, dilation, groups):
    spatial = tuple((size - 1) * step - 2 * pad + dil * (kernel - 1) + op + 1
                    for size, kernel, step, pad, op, dil in zip(
                        source.shape[2:], weight.shape[2:], stride, padding, output_padding, dilation))
    result = np.zeros((source.shape[0], weight.shape[1] * groups) + spatial, dtype=np.float64)
    in_group, out_group = source.shape[1] // groups, weight.shape[1]
    for row, col, kh, kw in np.ndindex(source.shape[2], source.shape[3], weight.shape[2], weight.shape[3]):
        oy, ox = row * stride[0] - padding[0] + kh * dilation[0], col * stride[1] - padding[1] + kw * dilation[1]
        if not (0 <= oy < spatial[0] and 0 <= ox < spatial[1]):
            continue
        for group in range(groups):
            ci, co = slice(group * in_group, (group + 1) * in_group), slice(group * out_group, (group + 1) * out_group)
            result[:, co, oy, ox] += source[:, ci, row, col].astype(np.float64) @ weight[ci, :, kh, kw].astype(np.float64)
    if bias is not None:
        result += bias.astype(np.float64)[None, :, None, None]
    return result


def check(actual, expected, dtype):
    actual.sync(True)
    assert actual.location() == "device" and str(actual.dtype) == dtype
    np.testing.assert_allclose(actual.numpy(), expected, rtol=4e-3, atol=4e-4)


@pytest.mark.parametrize("case", GEOMETRIES)
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("module", [False, True])
def test_public_function_and_module(case, dtype, module):
    shape, weight_shape, stride, padding, output_padding, dilation, groups, biased = case
    rng = np.random.RandomState(20261029)
    source, weight = rng.uniform(-.2, .2, shape).astype(dtype), rng.uniform(-.2, .2, weight_shape).astype(dtype)
    bias = rng.uniform(-.1, .1, weight_shape[1] * groups).astype(dtype) if biased else None
    expected = reference(source, weight, bias, stride, padding, output_padding, dilation, groups)
    with jt.no_grad():
        x, w, b = jt.array(source), jt.array(weight), None if bias is None else jt.array(bias)
        if module:
            layer = jt.nn.ConvTranspose2d(shape[1], weight_shape[1] * groups,
                                         weight_shape[2:], stride, padding, output_padding,
                                         groups, biased, dilation)
            layer.weight, layer.bias = w, b
            actual = layer(x)
        else:
            actual = jt.nn.conv_transpose2d(x, w, b, stride, padding, output_padding, groups, dilation)
        check(actual, expected, dtype)


def test_actual_depth_anything_resize_geometry():
    rng = np.random.RandomState(48)
    source = rng.randint(-4, 5, (1, 48, 37, 37)).astype(np.float16) * np.float16(.125)
    # Non-overlapping 4x4/stride4 gives an exact independent channel-sum oracle.
    expected = (source.astype(np.float64).sum(1, keepdims=True) * .125 + .0625)
    expected = np.repeat(np.repeat(np.repeat(expected, 48, axis=1), 4, axis=2), 4, axis=3)
    with jt.no_grad():
        actual = jt.nn.conv_transpose2d(jt.array(source), jt.full((48, 48, 4, 4), .125).float16(),
                                       jt.full((48,), .0625).float16(), stride=4)
        check(actual, expected, "float16")


def test_admission_declines_grad_other_soc_mixed_dtypes_and_noncontiguous(monkeypatch):
    x, w = jt.ones((1, 4, 3, 4)).float16(), jt.ones((4, 3, 3, 3)).float16()
    args = (x, w, None, (2, 2), (1, 1), (1, 1), (1, 1), 1)
    with jt.no_grad():
        assert select_kernel("conv_transpose2d", *args) is neural.conv_transpose2d_acl
        for soc in ("Ascend910B3", "Ascend910", ""):
            monkeypatch.setattr(neural, "soc_name", lambda: soc)
            assert select_kernel("conv_transpose2d", *args) is None
        monkeypatch.setattr(neural, "soc_name", lambda: "Ascend310P3")
        assert select_kernel("conv_transpose2d", x, w.float32(), *args[2:]) is None
        view = jt.core.ops.transpose(jt.ones((1, 4, 4, 3)).float16(), (0, 1, 3, 2))
        assert not view._storage_is_contiguous()
        assert select_kernel("conv_transpose2d", view, *args[1:]) is None
        with jt.runtime.scope(acl_allow_hf32=0):
            monkeypatch.setattr(neural, "allow_reduced_precision", lambda: False)
            assert select_kernel("conv_transpose2d", x.float32(), w.float32(), *args[2:]) is None
    with jt.enable_grad():
        assert select_kernel("conv_transpose2d", *args) is None


def test_programs_separate_transpose_geometry_and_normal_conv_backward():
    normal = _conv_program(False, (1, 1), (0, 0), (1, 1), 1, 1)
    transposed = _conv_transpose_program(False, (1, 1), (0, 0), (0, 0), (1, 1), 1, 1)
    other_padding = _conv_transpose_program(False, (2, 2), (0, 0), (1, 0), (1, 1), 1, 1)
    assert normal.data != transposed.data != other_padding.data
    assert transposed is _conv_transpose_program(False, (1, 1), (0, 0), (0, 0), (1, 1), 1, 1)
    rng = np.random.RandomState(19)
    source, weight, seed = [rng.uniform(-.2, .2, shape).astype(np.float32)
                            for shape in ((1, 2, 3, 4), (3, 2, 2, 2), (1, 3, 2, 3))]
    expected, gx, gw = np.zeros_like(seed), np.zeros_like(source), np.zeros_like(weight)
    for row, col, kh, kw in np.ndindex(2, 3, 2, 2):
        patch, taps, cot = source[:, :, row + kh, col + kw], weight[:, :, kh, kw], seed[:, :, row, col]
        expected[:, :, row, col] += patch @ taps.T
        gx[:, :, row + kh, col + kw] += cot @ taps
        gw[:, :, kh, kw] += cot.T @ patch
    with jt.enable_grad():
        x, w = jt.array(source).start_grad(), jt.array(weight).start_grad()
        actual = jt.nn.conv2d(x, w)
        dx, dw = jt.grad((actual * jt.array(seed)).sum(), [x, w])
        for tensor, oracle in ((actual, expected), (dx, gx), (dw, gw)):
            check(tensor, oracle, "float32")
    with jt.no_grad():
        # The same ordinary geometry still runs correctly after the transpose
        # program was materialized; attribute program caches cannot leak modes.
        check(jt.nn.conv_transpose2d(jt.array(seed), jt.array(weight)),
              reference(seed, weight, None, (1, 1), (0, 0), (0, 0), (1, 1), 1), "float32")
        check(jt.nn.conv2d(jt.array(source), jt.array(weight)), expected, "float32")
