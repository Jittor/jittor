"""Host construction contracts; CUDA arithmetic is tested on CUDA separately."""

from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pytest

import jittor as jt
from jittor.backends.cuda.kernels.nn import fused_attention_f32_cuda as attention


def _check_unexecuted_graph(sequence_major):
    shape = (2, 4, 3, 8) if sequence_major else (2, 3, 4, 8)
    key_shape = (2, 7, 3, 8) if sequence_major else (2, 3, 7, 8)
    shapes = (shape, key_shape, key_shape)
    inputs = [jt.array(np.arange(np.prod(s), dtype="float32").reshape(s) / 100)
              for s in shapes]
    for value in inputs:
        value.start_grad()
    call = attention._Call(8 ** -0.5, False, (0, None, (0, 0, 0, 0)),
                           (sequence_major,) * 3, 0.)
    output = attention._native_forward(*inputs, call)[0]
    assert tuple(output.shape) == shape
    assert output.location() == "none"
    # A disconnected native constructor must fail here rather than returning
    # zeros. No CUDA program is executed by this host-only ABI contract.
    gradients = jt.grad(output.sum(), inputs, retain_graph=True)
    assert len(gradients) == 3
    for gradient, expected_shape in zip(gradients, shapes):
        assert tuple(gradient.shape) == expected_shape
        assert gradient.location() == "none"
    other = jt.grad(output.sum() * 2, inputs, retain_graph=True)
    assert all(a is not b for a, b in zip(gradients, other))


@pytest.mark.parametrize("sequence_major", [False, True])
def test_native_gradient_builder_connects_inputs_and_outputs(sequence_major):
    # Drop every graph holder in the helper before restoring submission flags.
    with jt.flag_scope(use_cuda=0, lazy_execution=1, auto_flush_ops=0,
                       async_flush_ops=0, missing_grad_error=1):
        _check_unexecuted_graph(sequence_major)


def test_physical_training_view_preserves_the_intermediate_gradient_target():
    with jt.flag_scope(use_cuda=0, missing_grad_error=1):
        source = jt.array(np.arange(48, dtype="float32").reshape(1, 3, 2, 8))
        source.start_grad()
        logical = source._storage_permute((0, 2, 1, 3))
        physical, sequence_major = attention._physical(logical, preserve_gradient_edge=True)
        assert sequence_major and physical._storage_is_contiguous()
        assert physical._producer_is_view()
        gradient = jt.grad(physical.sum(), logical)
        np.testing.assert_array_equal(gradient.numpy(), np.ones(logical.shape, dtype="float32"))


@pytest.mark.parametrize("group", [1, 2])
def test_grouped_training_preserves_its_gradient_owner(group):
    query = jt.array(np.ones((1, 2, 3, 8), dtype="float32"))
    key = jt.array(np.ones((1, 2 // group, 3, 8), dtype="float32"))
    value = jt.array(np.ones((1, 2 // group, 3, 8), dtype="float32"))
    for tensor in (query, key, value):
        tensor.start_grad()
    with jt.flag_scope(keep_graph=0), \
            patch.object(attention, "_forward", return_value=(query, None)) as forward, \
            patch.object(attention._FusedAttentionF32, "apply", return_value=query) as legacy:
        assert attention._fused_attention_f32(
            query, key, value, enable_gqa=True) is query
    assert forward.call_count == int(group == 1)
    assert legacy.call_count == int(group > 1)


def test_kept_graph_preserves_its_gradient_owner():
    tensor = jt.array(np.ones((1, 2, 3, 8), dtype="float32"))
    tensor.start_grad()
    with jt.flag_scope(keep_graph=1), \
            patch.object(attention, "_native_forward", side_effect=AssertionError(
                "native gradient owner used in an existing graph mode")), \
            patch.object(attention._FusedAttentionF32, "apply", return_value=tensor) as legacy:
        assert attention._fused_attention_f32(tensor, tensor, tensor) is tensor
    legacy.assert_called_once()


@pytest.mark.parametrize("mode", ["active", "tracing"])
def test_capture_preserves_its_gradient_owner(mode):
    from jittor._runtime import step_capture
    tensor = jt.array(np.ones((1, 2, 3, 8), dtype="float32"))
    tensor.start_grad()
    with jt.flag_scope(keep_graph=0), \
            patch.object(step_capture, mode, return_value=True), \
            patch.object(attention, "_native_vjp_factory", side_effect=AssertionError(
                "native gradient owner used during capture")), \
            patch.object(attention._FusedAttentionF32, "apply", return_value=tensor) as legacy:
        assert attention._fused_attention_f32(tensor, tensor, tensor) is tensor
    legacy.assert_called_once()


def test_disabled_native_build_preserves_its_gradient_owner():
    tensor = jt.array(np.ones((1, 2, 3, 8), dtype="float32"))
    tensor.start_grad()
    with jt.flag_scope(keep_graph=0), \
            patch.object(attention, "_native_vjp_factory",
                         return_value=SimpleNamespace(native_vjp_enabled=False)), \
            patch.object(attention._FusedAttentionF32, "apply", return_value=tensor) as legacy:
        assert attention._fused_attention_f32(tensor, tensor, tensor) is tensor
    legacy.assert_called_once()
