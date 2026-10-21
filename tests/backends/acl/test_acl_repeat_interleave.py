"""Uniform repetition uses native placement and sums every repeated cotangent."""
import numpy as np
import pytest
import jittor as jt

from _helpers.capability import require_accelerator
from jittor._runtime.fallback import forbid_backend_fallbacks


CASES = [
    ((2, 2, 4, 4), 2, 1),
    ((2, 2, 4, 4), 2, -3),
    ((2, 3), 3, 0),
    ((2, 3), 4, -1),
    ((2, 3), 2, None),
    ((2, 3), 1, 1),
]


def _assert_placement(value, device):
    value.sync()
    expected = ("acl", 0) if device == "npu" else ("cpu", -1)
    assert tuple(jt.core.dispatch_context([value])) == expected
    assert value.location() == ("device" if device == "npu" else "cpu")
    assert str(value.dtype) == "float32"


@pytest.mark.parametrize("device", ["cpu", pytest.param("npu", marks=pytest.mark.npu)])
@pytest.mark.parametrize("shape,repeats,dim", CASES)
def test_uniform_repeat_full_forward_and_source_gradient(device, shape, repeats, dim):
    if device == "npu":
        require_accelerator("acl")
    source = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - np.prod(shape) // 2) / 8
    expected = np.repeat(source, repeats, axis=dim)
    weights = (np.arange(expected.size).reshape(expected.shape) % 7 + 1).astype(np.float32) / 8
    indices = np.repeat(np.arange(source.size).reshape(shape), repeats, axis=dim)
    expected_gradient = np.zeros(source.size, dtype=np.float32)
    np.add.at(expected_gradient, indices.reshape(-1), weights.reshape(-1))
    with jt.flag_scope(use_cuda=int(device == "npu")), forbid_backend_fallbacks():
        value = jt.array(source, dtype="float32").start_grad()
        result = jt.repeat_interleave(value, repeats, dim=dim)
        _assert_placement(result, device)
        np.testing.assert_array_equal(result.numpy(), expected)
        gradient = jt.grad((result * jt.array(weights, dtype="float32")).sum(), value)
        _assert_placement(gradient, device)
        np.testing.assert_array_equal(gradient.numpy(), expected_gradient.reshape(shape))


@pytest.mark.parametrize("device", ["cpu", pytest.param("npu", marks=pytest.mark.npu)])
def test_gqa_sdpa_full_forward_and_all_input_gradients(device):
    if device == "npu":
        require_accelerator("acl")
    q = (np.arange(128, dtype=np.float32).reshape(2, 4, 4, 4) % 13 - 6) / 8
    k = (np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4) % 11 - 5) / 8
    v = (np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4) % 7 - 3) / 8
    kr, vr = np.repeat(k, 2, axis=1), np.repeat(v, 2, axis=1)
    scores = q @ np.swapaxes(kr, -1, -2) / 2
    scores -= scores.max(axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = probabilities @ vr
    weights = (np.arange(expected.size).reshape(expected.shape) % 5 + 1).astype(np.float32) / 8
    dp = weights @ np.swapaxes(vr, -1, -2)
    ds = probabilities * (dp - (dp * probabilities).sum(axis=-1, keepdims=True))
    gq = ds @ kr / 2
    gk = (np.swapaxes(ds, -1, -2) @ q / 2).reshape(2, 2, 2, 4, 4).sum(axis=2)
    gv = (np.swapaxes(probabilities, -1, -2) @ weights).reshape(2, 2, 2, 4, 4).sum(axis=2)
    with jt.flag_scope(use_cuda=int(device == "npu")), forbid_backend_fallbacks():
        inputs = [jt.array(array, dtype="float32").start_grad() for array in (q, k, v)]
        output = jt.nn.scaled_dot_product_attention(*inputs, enable_gqa=True)
        _assert_placement(output, device)
        np.testing.assert_allclose(output.numpy(), expected, atol=1e-5, rtol=1e-4)
        gradients = jt.grad((output * jt.array(weights, dtype="float32")).sum(), inputs)
        for gradient, reference in zip(gradients, (gq, gk, gv)):
            _assert_placement(gradient, device)
            np.testing.assert_allclose(gradient.numpy(), reference, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("device", ["cpu", pytest.param("npu", marks=pytest.mark.npu)])
def test_repeat_once_has_independent_storage_and_keeps_source_gradient(device):
    if device == "npu":
        require_accelerator("acl")
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    with jt.flag_scope(use_cuda=int(device == "npu")), forbid_backend_fallbacks():
        value = jt.array(source, dtype="float32").start_grad()
        output = jt.repeat_interleave(value, 1, dim=1)
        _assert_placement(value, device)
        _assert_placement(output, device)
        assert value._mem_ptr_now and output._mem_ptr_now
        assert value._mem_ptr_now != output._mem_ptr_now
        value._write_inplace(source + 100)
        jt.sync_all(True)
        np.testing.assert_array_equal(output.numpy(), source)
        gradient = jt.grad((output * output).sum(), value)
        _assert_placement(gradient, device)
        np.testing.assert_array_equal(gradient.numpy(), 2 * source)
