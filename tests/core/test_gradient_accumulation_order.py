"""A selectable consumer order keeps native defaults and higher AD explicit."""

import os

import jittor as jt
import numpy as np
import pytest

from jittor._runtime import forbid_backend_fallbacks


@pytest.mark.parametrize("order,coefficient", [(0, 1.0), (1, 0.0)])
def test_consumer_order_cancellation_and_connected_hessian(order, coefficient):
    with jt.flag_scope(
        use_cuda=int(os.environ.get("JITTOR_TEST_DEVICE", "cpu") == "cuda"),
        amp_reg=0,
        gradient_accumulation_order=order,
    ), forbid_backend_fallbacks():
        x = jt.array(np.array([0.2, -0.3], dtype="float32"))
        z = x.sin()
        first = z * 1e8
        second = z * (-1e8)
        last = z * 1.0
        loss = ((first + second) + last).sum()
        gradient = jt.grad(loss, x)
        hessian = jt.grad(gradient.sum(), x)
        np.testing.assert_allclose(
            gradient.numpy(), coefficient * np.cos(np.array([0.2, -0.3])), rtol=2e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            hessian.numpy(), -coefficient * np.sin(np.array([0.2, -0.3])), rtol=2e-6, atol=1e-7
        )
        assert jt.runtime.context.snapshot()["gradient_accumulation_order"] == order


def test_consumer_order_scope_restores_default_and_rejects_unknown_values():
    original = jt.flags.gradient_accumulation_order
    try:
        assert original == 0
        with jt.flag_scope(gradient_accumulation_order=1):
            assert jt.flags.gradient_accumulation_order == 1
        assert jt.flags.gradient_accumulation_order == original
        with pytest.raises(RuntimeError, match="gradient_accumulation_order must be 0 or 1"):
            jt.flags.gradient_accumulation_order = 2
    finally:
        jt.flags.gradient_accumulation_order = original


@pytest.mark.parametrize("order,coefficient", [(0, 1.0), (1, 0.0)])
def test_nonleaf_hook_receives_complete_ordered_sum_and_preserves_target_boundary(
    order, coefficient
):
    with jt.flag_scope(
        use_cuda=int(os.environ.get("JITTOR_TEST_DEVICE", "cpu") == "cuda"),
        amp_reg=0,
        gradient_accumulation_order=order,
    ), forbid_backend_fallbacks():
        x = jt.array(np.array([0.2, -0.3], dtype="float32"))
        z = x.sin()
        seen = []

        def twice(value):
            seen.append(value)
            return value * 2.0

        handle = z.register_hook(twice)
        first = z * 1e8
        second = z * (-1e8)
        last = z * 1.0
        gx, gz = jt.grad(((first + second) + last).sum(), [x, z])
        handle.remove()
        assert len(seen) == 1
        np.testing.assert_array_equal(seen[0].numpy(), np.full((2,), coefficient, dtype="float32"))
        np.testing.assert_array_equal(gz.numpy(), np.full((2,), coefficient, dtype="float32"))
        np.testing.assert_allclose(
            gx.numpy(), 2 * coefficient * np.cos(np.array([0.2, -0.3])), rtol=2e-6, atol=1e-7
        )
