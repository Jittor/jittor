"""Gradient-free channels-last stencils retain values, live inputs and layout."""
import numpy as np
import pytest

import jittor as jt
from jittor.backends.cuda.kernels.nn.depthwise_nhwc_cuda import (
    _source as stencil_source, supports_depthwise_nhwc)

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not jt.flags.use_cuda, reason='requires an explicitly selected real CUDA backend')]


def reference(x, weight, bias, stride, padding):
    x, weight = x.astype('float64'), weight.astype('float64')
    n, c, h, w = x.shape
    kh, kw = weight.shape[-2:]
    oh = (h + 2*padding[0] - kh)//stride[0] + 1
    ow = (w + 2*padding[1] - kw)//stride[1] + 1
    padded = np.pad(x, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])))
    result = np.zeros((n, c, oh, ow), dtype='float64')
    for ky in range(kh):
        for kx in range(kw):
            patch = padded[:, :, ky:ky+oh*stride[0]:stride[0], kx:kx+ow*stride[1]:stride[1]]
            result += patch * weight[:, 0, ky, kx][None, :, None, None]
    if bias is not None:
        result += bias.astype('float64')[None, :, None, None]
    return result


def channels_last(array, requires_grad=False):
    root = jt.array(np.ascontiguousarray(array.transpose(0, 2, 3, 1)))
    root = root.start_grad() if requires_grad else root.stop_grad()
    view = root._storage_permute((0, 3, 1, 2))
    # Starting a native intermediate as a new gradient leaf would sever the
    # root whose gradient the training regression below checks.
    view = view if requires_grad else view.stop_grad()
    assert bool(view.requires_grad) == requires_grad
    return root, view


@pytest.mark.parametrize('kernel,stride,padding', [
    ((3, 3), (1, 1), (1, 1)),
    ((5, 7), (1, 2), (2, 3)),
    ((7, 7), (2, 1), (3, 2)),
])
@pytest.mark.parametrize('with_bias', [False, True])
@jt.flag_scope(backend_fallback='error', amp_reg=0,
               float32_cudnn_precision='highest', cuda_allow_cudnn_tf32=0)
def test_public_stencil_matches_float64_and_keeps_channels_last(kernel, stride, padding, with_bias):
    rng = np.random.default_rng(193)
    x = rng.normal(size=(2, 64, 32, 17)).astype('float32')
    weight = rng.normal(0, .2, size=(64, 1, *kernel)).astype('float32')
    bias = rng.normal(size=64).astype('float32') if with_bias else None
    _, values = channels_last(x)
    filt = jt.array(weight).stop_grad()
    offset = jt.array(bias).stop_grad() if bias is not None else None
    assert supports_depthwise_nhwc(values, filt, offset, stride, padding, (1, 1), 64)
    before = jt.core.backend_fallback_count()
    actual = jt.nn.conv2d(values, filt, offset, stride, padding, groups=64)
    assert not actual._storage_is_contiguous()
    np.testing.assert_allclose(actual.numpy(), reference(x, weight, bias, stride, padding),
                               rtol=2e-5, atol=3e-6)
    assert actual.location() == 'device'
    assert jt.core.backend_fallback_count() == before


@jt.flag_scope(backend_fallback='error', amp_reg=0,
               float32_cudnn_precision='highest', cuda_allow_cudnn_tf32=0,
               missing_grad_error=1)
def test_trainable_operands_keep_the_existing_backward():
    rng = np.random.default_rng(59)
    x = rng.normal(size=(2, 32, 32, 33)).astype('float32')
    weight = rng.normal(0, .1, size=(32, 1, 3, 3)).astype('float32')
    bias = rng.normal(size=32).astype('float32')
    root, values = channels_last(x, True)
    filt, offset = jt.array(weight).start_grad(), jt.array(bias).start_grad()
    assert not supports_depthwise_nhwc(values, filt, offset, (1, 1), (0, 0), (1, 1), 32)
    output = jt.nn.conv2d(values, filt, offset, groups=32)
    dy = rng.normal(size=tuple(output.shape)).astype('float32')
    dx, dw, db = jt.grad((output*jt.array(dy).stop_grad()).sum(), [root, filt, offset])
    expected_x = np.zeros_like(x, dtype='float64')
    expected_w = np.zeros_like(weight, dtype='float64')
    oh, ow = dy.shape[-2:]
    for ky in range(3):
        for kx in range(3):
            expected_x[:, :, ky:ky+oh, kx:kx+ow] += dy*weight[:, 0, ky, kx][None, :, None, None]
            expected_w[:, 0, ky, kx] = (x[:, :, ky:ky+oh, kx:kx+ow].astype('float64')*dy).sum((0, 2, 3))
    for actual, expected in ((dx, expected_x.transpose(0, 2, 3, 1)), (dw, expected_w),
                             (db, dy.astype('float64').sum((0, 2, 3)))):
        np.testing.assert_allclose(actual.numpy(), expected, rtol=3e-5, atol=2e-5)
        assert actual.location() == 'device'


@jt.flag_scope(backend_fallback='error', amp_reg=0,
               float32_cudnn_precision='highest', cuda_allow_cudnn_tf32=0)
def test_cuda_graph_replay_reads_changed_inputs_filters_and_biases():
    rng = np.random.default_rng(41)
    runner = jt.capture_step(lambda x, w, b: jt.nn.conv2d(
        x._storage_permute((0, 3, 1, 2)).stop_grad(), w, b, padding=1, groups=32))
    retained = []
    try:
        for iteration in range(8):
            x = rng.normal(size=(2, 32, 32, 33)).astype('float32')
            weight = rng.normal(0, .2, size=(32, 1, 3, 3)).astype('float32')
            bias = rng.normal(size=32).astype('float32')
            root, _ = channels_last(x)
            actual = runner(root, jt.array(weight).stop_grad(), jt.array(bias).stop_grad())
            retained.append((actual, reference(x, weight, bias, (1, 1), (1, 1))))
        for actual, expected in retained:
            np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-5, atol=3e-6)
            assert actual.location() == 'device'
        assert runner.stats['graph'] >= 3, runner._graph_refused
    finally:
        runner.invalidate()


@jt.flag_scope(backend_fallback='error', amp_reg=0)
def test_autocast_and_unsupported_stencils_decline_the_specialized_route():
    _, values = channels_last(np.ones((2, 32, 32, 33), dtype='float32'))
    weight = jt.ones((32, 1, 3, 3)).stop_grad()
    assert supports_depthwise_nhwc(values, weight, None, (1, 1), (1, 1), (1, 1), 32)
    with jt.flag_scope(amp_reg=jt.amp_flags.prefer16):
        assert not supports_depthwise_nhwc(values, weight, None, (1, 1), (1, 1), (1, 1), 32)
    assert not supports_depthwise_nhwc(values, weight, None, (1, 1), (1, 1), (2, 2), 32)
    assert not supports_depthwise_nhwc(values, weight, None, (1, 1), (1, 1), (1, 1), 32,
                                       _depthwise_fast_path=False)


@jt.flag_scope(backend_fallback='error', amp_reg=0,
               float32_cudnn_precision='highest', cuda_allow_cudnn_tf32=0)
def test_frozen_cudnn_chain_observes_filter_writes_during_graph_replay():
    rng = np.random.default_rng(73)
    x = rng.normal(size=(2, 8, 15, 13)).astype('float32')
    root, _ = channels_last(x)
    weight = jt.zeros((12, 8, 3, 3)).stop_grad()
    runner = jt.capture_step(lambda v: jt.nn.conv2d(v._storage_permute((0, 3, 1, 2)).stop_grad(), weight))
    try:
        for iteration in range(8):
            current = rng.normal(0, .1, size=(12, 8, 3, 3)).astype('float32')
            weight._copy_into(jt.array(current).stop_grad().sync(), False)
            output = runner(root)
            expected = np.zeros((2, 12, 13, 11), dtype='float64')
            for ky in range(3):
                for kx in range(3):
                    expected += np.einsum('nchw,kc->nkhw',
                        x[:, :, ky:ky+13, kx:kx+11].astype('float64'),
                        current[:, :, ky, kx].astype('float64'))
            np.testing.assert_allclose(output.numpy(), expected, rtol=2e-5, atol=3e-6)
            assert output.location() == 'device'
            if iteration == 0:
                assert not output._storage_is_contiguous()
        assert runner.stats['graph'] >= 3, runner._graph_refused
    finally:
        runner.invalidate()


@jt.flag_scope(use_cuda=1, backend_fallback='error', amp_reg=0)
def test_stencil_handles_an_unaligned_device_pointer():
    rng = np.random.default_rng(109)
    x = rng.normal(size=(2, 32, 32, 33)).astype('float32')
    weight = rng.normal(0, .1, size=(32, 1, 3, 3)).astype('float32')
    # Offset the launch pointer by one float. This exercises scalar device IO
    # for externally owned CUDA buffers that do not provide 16-byte alignment.
    storage = np.concatenate((np.zeros(1, dtype='float32'), x.transpose(0, 2, 3, 1).ravel()))
    packed = np.ascontiguousarray(weight.transpose(2, 3, 1, 0))
    header, launch = stencil_source(2, 32, 32, 33, 3, 3, 1, 1, 1, 1, 1, 1)
    actual = jt.code((2, 32, 33, 32), 'float32',
                     [jt.array(storage).stop_grad(), jt.array(packed).stop_grad()],
                     cuda_header=header, cuda_src=launch.replace('in0_p', '(in0_p+1)')).stop_grad()
    expected = reference(x, weight, None, (1, 1), (1, 1)).transpose(0, 2, 3, 1)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-5, atol=3e-6)
    assert actual.location() == 'device'
