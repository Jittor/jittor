"""Concatenation must preserve input values and promotion inside autocast."""
import os
from contextlib import ExitStack

import numpy as np
import pytest
import torch

_DEVICE = os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(",")[0]
_SHIM = hasattr(torch, "_torch_compat_install_context")
if _SHIM:
    import jittor as jt
    from jittor._runtime.fallback import forbid_backend_fallbacks


def _values(value):
    return value.detach().float().cpu().numpy().copy()


@pytest.mark.parametrize("amp_name", ("float16", "bfloat16"))
@pytest.mark.parametrize("training", (False, True))
@pytest.mark.parametrize("pair", ("full", "mixed", "reduced"))
def test_cat_autocast_preserves_values_dtype_and_backward(amp_name, training, pair):
    if _DEVICE == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a real CUDA device")
    amp_dtype = getattr(torch, amp_name)
    source = np.array([[1.0001220703125, -18.69140625, .9998779296875,
                        1.226959228515625, -.3000030517578125, 16.5673828125,
                        -1.1044921875, .031253814697265625]], dtype=np.float32)
    with ExitStack() as scope:
        before = 0
        if _SHIM:
            scope.enter_context(jt.runtime.scope(use_cuda=int(_DEVICE == "cuda"),
                                                backend_fallback="error"))
            scope.enter_context(forbid_backend_fallbacks())
            before = jt.core.backend_fallback_count()
        for operation in ("cat", "rotate"):
            left_dtype = amp_dtype if pair == "reduced" else torch.float32
            right_dtype = torch.float32 if pair == "full" else amp_dtype
            x = torch.tensor(source, dtype=left_dtype, device=_DEVICE,
                             requires_grad=training)
            y = torch.tensor(source[:, ::-1].copy(), dtype=right_dtype,
                             device=_DEVICE, requires_grad=training)
            x_values, y_values = _values(x), _values(y)
            with ExitStack() as computation:
                if not training:
                    computation.enter_context(torch.no_grad())
                computation.enter_context(torch.autocast(_DEVICE, dtype=amp_dtype))
                if operation == "cat":
                    result = torch.cat((x, y), dim=-1)
                    expected = np.concatenate((x_values, y_values), axis=-1)
                    expected_dtype = amp_dtype if pair == "reduced" else torch.float32
                else:
                    # Qwen's rotate_half: both inputs stay float32 after
                    # mixed-dtype RMSNorm, including values not on a BF16 grid.
                    result = torch.cat((-x[..., 4:], x[..., :4]), dim=-1)
                    expected = np.concatenate((-x_values[..., 4:], x_values[..., :4]), axis=-1)
                    expected_dtype = left_dtype
            assert result.dtype == expected_dtype
            assert result.device.type == _DEVICE
            if _SHIM:
                assert jt.core.dispatch_context([result])[0] == _DEVICE
            np.testing.assert_array_equal(_values(result), expected)
            if training:
                coefficients = np.arange(1, result.numel() + 1, dtype=np.float32).reshape(result.shape)
                coeff = torch.tensor(coefficients, dtype=result.dtype, device=_DEVICE)
                leaves = (x, y) if operation == "cat" else (x,)
                gradients = torch.autograd.grad((result * coeff).sum(), leaves)
                for leaf, gradient in zip(leaves, gradients):
                    assert gradient.dtype == leaf.dtype
                    assert gradient.device.type == _DEVICE
                    if _SHIM:
                        assert jt.core.dispatch_context([gradient])[0] == _DEVICE
                if operation == "cat":
                    np.testing.assert_array_equal(_values(gradients[0]), coefficients[..., :8])
                    np.testing.assert_array_equal(_values(gradients[1]), coefficients[..., 8:])
                else:
                    expected_grad = np.concatenate((coefficients[..., 4:], -coefficients[..., :4]), axis=-1)
                    np.testing.assert_array_equal(_values(gradients[0]), expected_grad)
        if _SHIM:
            assert jt.core.backend_fallback_count() == before


@pytest.mark.parametrize("amp_name", ("float16", "bfloat16"))
def test_mixed_cat_quadratic_keeps_second_order_gradients(amp_name):
    """Casting a first derivative to its leaf dtype must keep its graph."""
    if _DEVICE == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a real CUDA device")
    amp_dtype = getattr(torch, amp_name)
    source = np.array([[.25, -.5, 1., -2.]], dtype=np.float32)
    coefficients = np.arange(1, 9, dtype=np.float32).reshape(1, 8)
    with ExitStack() as scope:
        before = 0
        if _SHIM:
            scope.enter_context(jt.runtime.scope(use_cuda=int(_DEVICE == "cuda"),
                                                backend_fallback="error"))
            scope.enter_context(forbid_backend_fallbacks())
            before = jt.core.backend_fallback_count()
        x = torch.tensor(source, dtype=torch.float32, device=_DEVICE, requires_grad=True)
        y = torch.tensor(source / 2, dtype=amp_dtype, device=_DEVICE, requires_grad=True)
        weights = torch.tensor(coefficients, dtype=torch.float32, device=_DEVICE)
        with torch.autocast(_DEVICE, dtype=amp_dtype):
            joined = torch.cat((x, y), dim=-1)
            loss = (joined.square() * weights).sum()
        assert joined.dtype == torch.float32
        first = torch.autograd.grad(loss, (x, y), create_graph=True)
        second = torch.autograd.grad(first[0].sum() + first[1].sum(), (x, y))
        for leaf, gradient, curvature, weight in zip(
                (x, y), first, second, np.split(coefficients, 2, axis=-1)):
            assert gradient.requires_grad
            for derivative in (gradient, curvature):
                assert derivative.dtype == leaf.dtype
                assert derivative.device.type == _DEVICE
                if _SHIM:
                    assert jt.core.dispatch_context([derivative])[0] == _DEVICE
            # All chosen values and derivatives are exactly representable in
            # FP16/BF16, so the graph contract needs no numeric tolerance.
            np.testing.assert_array_equal(_values(gradient), 2 * _values(leaf) * weight)
            np.testing.assert_array_equal(_values(curvature), 2 * weight)
        if _SHIM:
            assert jt.core.backend_fallback_count() == before
