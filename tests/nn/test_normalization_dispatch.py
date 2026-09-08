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
