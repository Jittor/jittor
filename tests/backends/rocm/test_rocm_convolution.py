"""Native HIP im2col/GEMM convolution values and all requested derivatives."""

import numpy as np
import pytest
import jittor as jt
from _helpers.rocm import native_rocm
from jittor._runtime.dispatch import select_kernel

pytestmark = pytest.mark.skipif(not jt.compiler.has_rocm, reason='ROCm build required')


def reference(x, w, bias, stride, padding, dilation, seed=None):
    n, c, h, width = x.shape
    oc, _, kh, kw = w.shape
    sh, sw = stride; ph, pw = padding; dh, dw = dilation
    oh = (h + 2*ph - dh*(kh-1) - 1)//sh + 1
    ow = (width + 2*pw - dw*(kw-1) - 1)//sw + 1
    output = np.zeros((n, oc, oh, ow), dtype='float64')
    dx, dwgt = np.zeros_like(x, dtype='float64'), np.zeros_like(w, dtype='float64')
    for i in range(oh):
        for j in range(ow):
            for r in range(kh):
                for s in range(kw):
                    hi, wi = i*sh-ph+r*dh, j*sw-pw+s*dw
                    if not (0 <= hi < h and 0 <= wi < width):
                        continue
                    output[:, :, i, j] += x[:, :, hi, wi] @ w[:, :, r, s].T
                    if seed is not None:
                        dx[:, :, hi, wi] += seed[:, :, i, j] @ w[:, :, r, s]
                        dwgt[:, :, r, s] += seed[:, :, i, j].T @ x[:, :, hi, wi]
    if bias is not None:
        output += bias.reshape(1, oc, 1, 1)
    return output, dx, dwgt


@pytest.mark.parametrize('stride,padding,dilation', [((1,1),(1,1),(1,1)),
    ((2,1),(1,0),(1,1)), ((1,2),(2,1),(2,1))])
@pytest.mark.parametrize('with_bias', [False, True])
def test_public_forward_and_nonuniform_input_weight_bias_gradients(stride, padding, dilation, with_bias):
    rng = np.random.RandomState(113)
    raw = rng.normal(size=(2, 3, 7, 8)).astype('float32')/4
    weights = rng.normal(size=(4, 3, 3, 2)).astype('float32')/4
    biases = rng.normal(size=4).astype('float32')/4 if with_bias else None
    expected, _, _ = reference(raw, weights, biases, stride, padding, dilation)
    seed = rng.normal(size=expected.shape).astype('float32')
    expected, dx, dw = reference(raw, weights, biases, stride, padding, dilation, seed)
    x, w = jt.array(raw), jt.array(weights)
    b = jt.array(biases) if with_bias else None
    assert select_kernel('conv2d', x, w, b, stride, padding, dilation, 1) is not None
    out = jt.nn.conv2d(x, w, b, stride, padding, dilation)
    targets = [x, w] + ([b] if with_bias else [])
    grads = jt.grad((out * jt.array(seed)).sum(), targets)
    wanted = [dx, dw] + ([seed.sum((0,2,3))] if with_bias else [])
    for value, truth in [(out, expected)] + list(zip(grads, wanted)):
        value.sync()
        assert value.location() == 'device'
        np.testing.assert_allclose(value.numpy(), truth, rtol=3e-5, atol=3e-5)
    np.testing.assert_array_equal(x.numpy(), raw)
    np.testing.assert_array_equal(w.numpy(), weights)


def test_noncontiguous_inputs_and_second_derivative():
    rng = np.random.RandomState(114)
    raw = rng.normal(size=(2, 3, 6, 5)).astype('float32')/8
    weights = rng.normal(size=(4, 3, 3, 3)).astype('float32')/8
    base = jt.array(raw.transpose(0,1,3,2).copy())
    x = base.transpose((0,1,3,2))
    w = jt.array(weights)
    out = jt.nn.conv2d(x, w, padding=1)
    first = jt.grad(out.sqr().sum(), w)
    second = jt.grad(first.sum(), w)
    # Hessian-vector product for ||Conv(x,w)||^2: 2 A^T A applied to ones.
    ones, _, _ = reference(raw, np.ones_like(weights), None, (1,1), (1,1), (1,1))
    _, _, expected = reference(raw, weights, None, (1,1), (1,1), (1,1), 2*ones)
    second.sync()
    assert second.location() == 'device'
    np.testing.assert_allclose(second.numpy(), expected, rtol=3e-5, atol=3e-5)
    np.testing.assert_array_equal(base.numpy(), raw.transpose(0,1,3,2))


def test_grouped_and_amp_keep_existing_device_path():
    from jittor.backends.rocm.kernels.convolution import supported
    x, w = jt.ones((1, 4, 5, 5)), jt.ones((4, 2, 3, 3))
    assert not supported(x, w, groups=2)
    y = jt.nn.conv2d(x, w, padding=1, groups=2)
    y.sync()
    assert y.location() == 'device'
    x, w = jt.ones((1, 4, 5, 5)), jt.ones((4, 4, 3, 3))
    with jt.flag_scope(amp_reg=18):
        assert not supported(x, w)


def test_noncontiguous_weight_and_bias_values_and_base_gradients():
    rng = np.random.RandomState(117)
    raw = rng.normal(size=(2,3,5,6)).astype('float32')/8
    weights = rng.normal(size=(4,3,3,2)).astype('float32')/8
    biases = rng.normal(size=8).astype('float32')/8
    x = jt.array(raw)
    weight_base = jt.array(weights.transpose(0,1,3,2).copy())
    w = weight_base.transpose((0,1,3,2))
    bias_base = jt.array(biases)
    b = bias_base[::2]
    out = jt.nn.conv2d(x,w,b,padding=1)
    truth, _, _ = reference(raw,weights,biases[::2],(1,1),(1,1),(1,1))
    seed = rng.normal(size=truth.shape).astype('float32')
    _, dx, dw = reference(raw,weights,biases[::2],(1,1),(1,1),(1,1),seed)
    gradients = jt.grad((out*jt.array(seed)).sum(),[x,weight_base,bias_base])
    db = np.zeros_like(biases);db[::2] = seed.sum((0,2,3))
    for actual,expected in [(out,truth),(gradients[0],dx),
            (gradients[1],dw.transpose(0,1,3,2)),(gradients[2],db)]:
        actual.sync()
        assert actual.location()=='device'
        np.testing.assert_allclose(actual.numpy(),expected,rtol=3e-5,atol=3e-5)
    np.testing.assert_array_equal(weight_base.numpy(),weights.transpose(0,1,3,2))
    np.testing.assert_array_equal(bias_base.numpy(),biases)


def test_workspace_and_blas_limits_decline_without_allocating():
    from types import SimpleNamespace
    from jittor.backends.rocm.kernels.convolution import supported
    def metadata(shape):
        return SimpleNamespace(ndim=4,dtype='float32',shape=shape,
                               _storage_is_contiguous=lambda: True)
    assert not supported(metadata((16,512,256,256)),metadata((512,512,3,3)),padding=1)
    assert not supported(metadata((1,2147483648,1,1)),metadata((1,2147483648,1,1)))
    assert supported(metadata((1,64,32,32)),metadata((64,64,3,3)),padding=1)
