"""ACL Conv3d inference, supported gradients, and explicit 310P rejection."""

import numpy as np
import pytest

import jittor as jt

from _helpers import capability as _test_capability
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.backends.acl.kernels._soc import soc_name


@pytest.fixture(autouse=True)
def acl_device():
    if not _test_capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required for Conv3d")
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_acl=1, use_cuda=1, acl_allow_hf32=1), forbid_backend_fallbacks():
        yield
    assert jt.core.backend_fallback_count() == before


# Small geometries exercise the production convolution and its gradients;
# the patch case matches Wan's temporal/spatial kernel and stride topology.
_GEOMETRIES = {
    "patch": ((1, 3, 3, 6, 8), (4, 3, 1, 2, 2), (1, 2, 2), (0, 0, 0), (1, 1, 1), 1),
    "volume": ((1, 2, 4, 5, 6), (3, 2, 2, 3, 2), (1, 1, 1), (1, 1, 0), (1, 1, 1), 1),
    "dilated": ((1, 2, 5, 7, 6), (4, 2, 2, 2, 2), (2, 1, 2), (1, 1, 1), (1, 2, 1), 1),
    "grouped": ((1, 4, 4, 5, 6), (6, 2, 2, 3, 2), (1, 1, 1), (0, 1, 1), (1, 1, 1), 2),
}


def _reference(source, weight, stride, padding, dilation, groups, cotangent=None):
    """NCDHW cross correlation and VJP; no framework convolution is used."""
    source, weight = source.astype(np.float64), weight.astype(np.float64)
    spatial = tuple(
        (size + 2 * pad - dil * (kernel - 1) - 1) // step + 1
        for size, kernel, step, pad, dil in zip(
            source.shape[2:], weight.shape[2:], stride, padding, dilation)
    )
    padded = np.pad(source, [(0, 0), (0, 0)] + [(pad, pad) for pad in padding])
    output = np.zeros((source.shape[0], weight.shape[0]) + spatial, dtype=np.float64)
    grad_padded, grad_weight = np.zeros_like(padded), np.zeros_like(weight)
    in_group, out_group = source.shape[1] // groups, weight.shape[0] // groups
    if cotangent is not None:
        cotangent = cotangent.astype(np.float64)
    for group in range(groups):
        input_channels = slice(group * in_group, (group + 1) * in_group)
        output_channels = slice(group * out_group, (group + 1) * out_group)
        for offset in np.ndindex(*weight.shape[2:]):
            window = tuple(
                slice(index * dil, index * dil + size * step, step)
                for index, dil, size, step in zip(offset, dilation, spatial, stride)
            )
            input_index = (slice(None), input_channels) + window
            weight_index = (output_channels, slice(None)) + offset
            patch, taps = padded[input_index], weight[weight_index]
            output[:, output_channels] += np.einsum("ncdhw,oc->nodhw", patch, taps)
            if cotangent is not None:
                cot = cotangent[:, output_channels]
                grad_padded[input_index] += np.einsum("nodhw,oc->ncdhw", cot, taps)
                grad_weight[weight_index] = np.einsum("nodhw,ncdhw->oc", cot, patch)
    if cotangent is None:
        return output
    original_window = tuple(slice(pad, pad + size) for pad, size in zip(padding, source.shape[2:]))
    grad_source = grad_padded[(slice(None), slice(None)) + original_window]
    return output, grad_source, grad_weight, cotangent.sum(axis=(0, 2, 3, 4))


def _arrays(name, dtype, with_bias):
    shape, weight_shape, stride, padding, dilation, groups = _GEOMETRIES[name]
    rng = np.random.RandomState(20261020)
    source = rng.uniform(-0.2, 0.2, shape).astype(dtype)
    weight = rng.uniform(-0.2, 0.2, weight_shape).astype(dtype)
    bias = rng.uniform(-0.1, 0.1, weight_shape[0]).astype(dtype) if with_bias else None
    return source, weight, bias, stride, padding, dilation, groups


def _check(actual, expected, dtype):
    actual.sync()
    assert actual.location() == "device"
    assert str(actual.dtype).replace("torch.", "") == dtype
    np.testing.assert_allclose(actual.numpy(), expected, rtol=5e-3, atol=4e-4)


@pytest.mark.parametrize("name", tuple(_GEOMETRIES))
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_conv3d_forward(name, dtype, with_bias):
    source, weight, bias, stride, padding, dilation, groups = _arrays(name, dtype, with_bias)
    expected = _reference(source, weight, stride, padding, dilation, groups).astype(dtype)
    # The public Conv3d adds its bias after the convolution, preserving this
    # separately rounded FP16 addition instead of silently fusing it.
    if bias is not None:
        expected = (expected.astype(np.float64) + bias.reshape(1, -1, 1, 1, 1)).astype(dtype)
    with jt.no_grad():
        actual = jt.nn.conv3d(jt.array(source), jt.array(weight),
                              None if bias is None else jt.array(bias),
                              stride, padding, dilation, groups)
        _check(actual, expected, dtype)


@pytest.mark.parametrize("name", ["patch", "grouped"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_conv3d_backward(name, dtype, with_bias):
    source, weight, bias, stride, padding, dilation, groups = _arrays(name, dtype, with_bias)
    expected = _reference(source, weight, stride, padding, dilation, groups)
    cotangent = np.random.RandomState(20261021).uniform(-0.2, 0.2, expected.shape).astype(dtype)
    _, grad_source, grad_weight, grad_bias = _reference(
        source, weight, stride, padding, dilation, groups, cotangent)
    with jt.enable_grad():
        x, w = jt.array(source).start_grad(), jt.array(weight).start_grad()
        b = None if bias is None else jt.array(bias).start_grad()
        if soc_name().startswith("Ascend310P"):
            with pytest.raises(NotImplementedError, match="Conv3d gradients.*Ascend310P.*aclnnConvolutionBackward"):
                jt.nn.conv3d(x, w, b, stride, padding, dilation, groups)
            return
        output = jt.nn.conv3d(x, w, b, stride, padding, dilation, groups)
        variables = [x, w] if b is None else [x, w, b]
        expected_grads = [grad_source, grad_weight] if b is None else [grad_source, grad_weight, grad_bias]
        gradients = jt.grad((output * jt.array(cotangent)).sum(), variables)
        for actual, reference in zip(gradients, expected_grads):
            _check(actual, reference, dtype)


@pytest.mark.parametrize("variable", ["source", "weight"])
def test_conv3d_single_requested_gradient(variable):
    source, weight, _, stride, padding, dilation, groups = _arrays("patch", "float32", False)
    expected = _reference(source, weight, stride, padding, dilation, groups)
    cotangent = np.random.RandomState(29).uniform(-0.2, 0.2, expected.shape).astype("float32")
    _, gx, gw, _ = _reference(source, weight, stride, padding, dilation, groups, cotangent)
    with jt.enable_grad():
        x, w = jt.array(source), jt.array(weight)
        target = x if variable == "source" else w
        (w if variable == "source" else x).stop_grad()
        target.start_grad()
        if soc_name().startswith("Ascend310P"):
            with pytest.raises(NotImplementedError, match="Conv3d gradients.*Ascend310P.*aclnnConvolutionBackward"):
                jt.nn.conv3d(x, w, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return
        output = jt.nn.conv3d(x, w, stride=stride, padding=padding, dilation=dilation, groups=groups)
        gradient = jt.grad((output * jt.array(cotangent)).sum(), target)
        _check(gradient, gx if variable == "source" else gw, "float32")


@pytest.mark.parametrize("no_grad", [False, True])
def test_conv3d_forward_without_required_convolution_gradients(no_grad):
    source, weight, _, stride, padding, dilation, groups = _arrays("patch", "float32", False)
    expected = _reference(source, weight, stride, padding, dilation, groups)
    with jt.enable_grad():
        x, w = jt.array(source), jt.array(weight)
        if no_grad:
            # Explicit no_grad must override both tensors requiring gradients.
            x.start_grad()
            w.start_grad()
            with jt.no_grad():
                output = jt.nn.conv3d(x, w, stride=stride, padding=padding,
                                      dilation=dilation, groups=groups)
        else:
            # Merely enabling grad mode does not request these gradients.
            x.stop_grad()
            w.stop_grad()
            output = jt.nn.conv3d(x, w, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups)
        _check(output, expected, "float32")


def test_conv3d_separate_bias_gradient_remains_supported():
    source, weight, bias, stride, padding, dilation, groups = _arrays("patch", "float32", True)
    expected = _reference(source, weight, stride, padding, dilation, groups)
    cotangent = np.random.RandomState(37).uniform(-0.2, 0.2, expected.shape).astype("float32")
    with jt.enable_grad():
        x, w, b = jt.array(source).stop_grad(), jt.array(weight).stop_grad(), jt.array(bias).start_grad()
        output = jt.nn.conv3d(x, w, b, stride, padding, dilation, groups)
        gradient = jt.grad((output * jt.array(cotangent)).sum(), b)
        _check(gradient, cotangent.sum(axis=(0, 2, 3, 4)), "float32")


def test_conv3d_strided_input_and_weight():
    rng = np.random.RandomState(31)
    source = rng.uniform(-0.2, 0.2, (1, 2, 4, 5, 6)).astype("float16")
    weight = rng.uniform(-0.2, 0.2, (3, 2, 2, 3, 2)).astype("float16")
    permutation = (0, 1, 4, 3, 2)
    expected = _reference(source.transpose(permutation), weight.transpose(permutation),
                          (1, 1, 1), (0, 0, 0), (1, 1, 1), 1).astype("float16")
    with jt.no_grad():
        x, w = jt.array(source), jt.array(weight)
        jt.sync([x, w], True)
        x, w = x._storage_permute(permutation), w._storage_permute(permutation)
        assert not x._storage_is_contiguous()
        assert not w._storage_is_contiguous()
        _check(jt.nn.conv3d(x, w), expected, "float16")


@pytest.mark.parametrize("with_bias", [False, True])
def test_conv2d_forward_and_backward_preserved(with_bias):
    rng = np.random.RandomState(32)
    source = rng.uniform(-0.2, 0.2, (1, 2, 5, 6)).astype("float32")
    weight = rng.uniform(-0.2, 0.2, (3, 2, 3, 2)).astype("float32")
    bias = rng.uniform(-0.1, 0.1, 3).astype("float32") if with_bias else None
    geometry = ((1, 2, 1), (0, 1, 0), (1, 1, 1), 1)
    expected = _reference(source[:, :, None], weight[:, :, None], *geometry)[:, :, 0]
    cotangent = rng.uniform(-0.2, 0.2, expected.shape).astype("float32")
    _, gx, gw, gb = _reference(source[:, :, None], weight[:, :, None],
                               *geometry, cotangent[:, :, None])
    if bias is not None:
        expected += bias.reshape(1, -1, 1, 1)
    with jt.enable_grad():
        x, w = jt.array(source).start_grad(), jt.array(weight).start_grad()
        b = None if bias is None else jt.array(bias).start_grad()
        output = jt.nn.conv2d(x, w, b, stride=(2, 1), padding=(1, 0))
        _check(output, expected, "float32")
        variables = [x, w] if b is None else [x, w, b]
        references = [gx[:, :, 0], gw[:, :, 0]] + ([] if b is None else [gb])
        gradients = jt.grad((output * jt.array(cotangent)).sum(), variables)
        for actual, reference in zip(gradients, references):
            _check(actual, reference, "float32")
