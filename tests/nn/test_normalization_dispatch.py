"""Normalization capabilities use registry selection and preserve device math."""

from _helpers import capability as _test_capability

import numpy as np
import pytest


def test_normalization_cuda_candidates_decline_cpu_inputs():
    import jittor as jt
    from jittor._runtime.dispatch import select_kernel
    from jittor.backends.cuda.kernels.nn import layer_norm_training_cuda, softmax_cuda

    with jt.flag_scope(use_cuda=0):
        x = jt.array(np.arange(16, dtype=np.float32).reshape(2, 8))
        weight, bias = jt.ones(8), jt.zeros(8)
        assert select_kernel("nn.layer_norm.training", x, (8,), weight, bias, 1e-5) is None
        assert layer_norm_training_cuda._layer_norm_cuda(x, (8,), weight, bias, 1e-5) is None
        assert not softmax_cuda.can_softmax_v1(x, -1)
        assert softmax_cuda.softmax_v1(x) is None


@pytest.mark.cuda
def test_registered_cuda_layer_norm_forward_backward_and_capability_limits():
    import jittor as jt
    from jittor._runtime.dispatch import select_kernel
    from jittor.backends.cuda.kernels.nn.layer_norm_training_cuda import _layer_norm_cuda

    if not _test_capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip("CUDA runtime required")
    data = np.arange(16, dtype=np.float32).reshape(2, 8) / 7
    weight_data = np.linspace(0.5, 1.5, 8, dtype=np.float32)
    grad_data = np.linspace(-1, 1, 16, dtype=np.float32).reshape(2, 8)
    eps = 1e-4
    with jt.flag_scope(use_cuda=1, auto_convert_64_to_32=0):
        x, weight = jt.array(data), jt.array(weight_data)
        bias = jt.zeros(8)
        for value in (x, weight, bias):
            value.start_grad()
        args = (x, (8,), weight, bias, eps)
        assert select_kernel("nn.layer_norm.training", *args) is _layer_norm_cuda.__wrapped__
        assert select_kernel("nn.layer_norm.training", x, (4,), weight, bias, eps) is None
        assert select_kernel("nn.layer_norm.training", x, (8,), weight.float64(), bias, eps) is None
        with jt.no_grad():
            assert select_kernel("nn.layer_norm.training", *args) is None
        result = _layer_norm_cuda(*args)
        assert result is not None
        result.sync()
        assert result.device_id >= 0
        assert result.location() == "device"
        gradients = jt.grad((result * jt.array(grad_data)).sum(), [x, weight, bias])
        actual = result.numpy()
        actual_gradients = [value.numpy() for value in gradients]

    centered = data - data.mean(-1, keepdims=True)
    inv_std = 1 / np.sqrt((centered * centered).mean(-1, keepdims=True) + eps)
    normalized = centered * inv_std
    dxhat = grad_data * weight_data
    dx = (dxhat - dxhat.mean(-1, keepdims=True)
          - normalized * (dxhat * normalized).mean(-1, keepdims=True)) * inv_std
    np.testing.assert_allclose(actual, normalized * weight_data, rtol=2e-5, atol=2e-5)
    expected_gradients = [dx, (grad_data * normalized).sum(0), grad_data.sum(0)]
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=2e-4, atol=2e-5)


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(2, 8), (8, 128), (1024, 64), (1031, 65), (8, 1025), (1, 1)])
@pytest.mark.parametrize("trainable", [(True, True, True), (True, False, False), (False, True, True)])
def test_cuda_layer_norm_repeated_vjps_and_frozen_inputs(shape, trainable):
    import jittor as jt
    from jittor.backends.cuda.kernels.nn.layer_norm_training_cuda import _layer_norm_cuda

    if not _test_capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip("CUDA runtime required")
    rng = np.random.default_rng(7392)
    hidden = shape[-1]
    data = rng.normal(size=shape).astype(np.float32)
    weight_data = rng.normal(size=hidden).astype(np.float32)
    bias_data = rng.normal(size=hidden).astype(np.float32)
    seeds = [rng.normal(size=shape).astype(np.float32) for _ in range(2)]
    centered = data.astype(np.float64) - data.astype(np.float64).mean(-1, keepdims=True)
    rstd = 1 / np.sqrt(np.square(centered).mean(-1, keepdims=True) + 1e-5)
    normalized = centered * rstd
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_cuda=1, backend_fallback='error', auto_graph_replay=0):
        args = [jt.array(value) for value in (data, weight_data, bias_data)]
        for value, enabled in zip(args, trainable):
            value.start_grad() if enabled else value.stop_grad()
        output = _layer_norm_cuda(args[0], (hidden,), args[1], args[2], 1e-5)
        losses = [(output * jt.array(seed)).sum() for seed in seeds]
        targets = [value for value, enabled in zip(args, trainable) if enabled]
        for loss, seed in zip(losses, seeds):
            # Independent VJPs of the same forward must use their own seed,
            # including when only affine parameters or only the input trains.
            gradients = jt.grad(loss, targets, retain_graph=True)
            scaled = seed * weight_data
            dx = (scaled - scaled.mean(-1, keepdims=True)
                  - normalized * (scaled * normalized).mean(-1, keepdims=True)) * rstd
            axes = tuple(range(len(shape) - 1))
            expected = [dx, (seed * normalized).sum(axes), seed.sum(axes)]
            trainable_expected = [value for value, enabled in zip(expected, trainable) if enabled]
            for gradient, target in zip(gradients, trainable_expected):
                gradient.sync()
                assert gradient.location() == 'device' and gradient.device_id >= 0
                np.testing.assert_allclose(gradient.numpy(), target, rtol=2e-4, atol=2e-5)
        output.sync()
        assert output.location() == 'device' and output.device_id >= 0
        np.testing.assert_allclose(output.numpy(), normalized * weight_data + bias_data,
                                   rtol=2e-5, atol=2e-5)
    assert jt.core.backend_fallback_count() == before


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(4, 8), (1024, 64)])
def test_cuda_layer_norm_backward_preserves_rebound_input_holders(shape):
    import jittor as jt
    from jittor.backends.cuda.kernels.nn.layer_norm_training_cuda import _layer_norm_cuda

    if not _test_capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip("CUDA runtime required")
    rng = np.random.default_rng(2301)
    data = rng.normal(size=shape).astype(np.float32)
    weight_data = rng.normal(size=shape[-1]).astype(np.float32)
    with jt.flag_scope(use_cuda=1, backend_fallback='error', auto_graph_replay=0):
        x, weight, bias = jt.array(data), jt.array(weight_data), jt.zeros(shape[-1])
        for value in (x, weight, bias):
            value.start_grad()
        xholder, wholder = (jt.ops.reshape(value, value.shape) for value in (x, weight))
        output = _layer_norm_cuda(xholder, (shape[-1],), wholder, bias, 1e-5)
        output.sync()
        xholder.assign(jt.full(shape, 7.))
        wholder.assign(jt.full(weight.shape, 7.))
        actual = jt.grad(output.sqr().sum(), [x, weight, bias])
        reference_x, reference_weight = jt.array(data), jt.array(weight_data)
        reference_bias = jt.zeros(shape[-1])
        for value in (reference_x, reference_weight, reference_bias):
            value.start_grad()
        reference = _layer_norm_cuda(reference_x, (shape[-1],), reference_weight, reference_bias, 1e-5)
        expected = jt.grad(reference.sqr().sum(), [reference_x, reference_weight, reference_bias])
        for value, target in zip(actual, expected):
            value.sync()
            assert value.location() == 'device' and value.device_id >= 0
            np.testing.assert_array_equal(value.numpy(), target.numpy())


@pytest.mark.cuda
def test_attention_lengths_cache_tracks_real_cuda_device():
    import jittor as jt
    from jittor.nn import attention

    if not _test_capability.check_accelerator('cuda', backend=jt).enabled or jt.core.backend_device_count("cuda") < 2:
        pytest.skip("two CUDA devices required")
    lengths = (3, 2)
    with jt.flag_scope(use_cuda=1, device_id=0):
        first = attention.cumulative_sequence_lengths(lengths)
        first.sync()
        assert first.device_id == 0
        assert first.location() == "device"
    with jt.flag_scope(use_cuda=1, device_id=1):
        second = attention.cumulative_sequence_lengths(lengths)
        packed = jt.zeros((5, 1, 2))
        packed.sync()
        second.sync()
        assert second.device_id == 1
        assert second.location() == "device"
        assert second is not first
    with jt.flag_scope(use_cuda=1, device_id=0):
        _, _, _, actual_lengths, _ = attention._prepare_varlen(packed, lengths, 2, None)
        assert actual_lengths is second
        np.testing.assert_array_equal(actual_lengths.numpy(), [0, 3, 5])
        assert attention.cumulative_sequence_lengths(lengths) is first
