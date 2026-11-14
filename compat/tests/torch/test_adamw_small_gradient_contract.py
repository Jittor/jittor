"""Independent FP64 AdamW reference where gradients approach epsilon.

The CUDA variance budget covers native single/foreach/fused FP32 rounding,
calibrated on Torch 70d99e998b4955e0049d13a98d77ae1b14db1f45 before
candidate execution. CPU and CUDA have separate budgets. These small-state
checks complement ordinary trajectory tests, whose absolute tolerances can
exceed the moments themselves. Calibration evidence is recorded in PR18126.
"""
import os

import jittor as jt
import numpy as np
import pytest
import torch


@pytest.fixture(params=os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(","))
def device(request):
    return request.param


@pytest.mark.parametrize("initial", [0., .001, 1.])
def test_adamw_epsilon_scale_updates_and_moments(device, initial):
    if device not in ("cpu", "cuda") or os.environ.get("JT_BACKEND", "cpu") not in ("cpu", "cuda"):
        pytest.skip("This FP32 reference budget is calibrated for CPU and CUDA")
    gradients = np.array([0., 1e-12, -1e-12, 1e-10, -1e-10, 1e-8, -1e-8,
                          1e-6, -1e-6, 1e-3, -1e-3], dtype=np.float32)
    smallest = float(np.nextafter(np.float32(0), np.float32(1)))
    expected = np.full(gradients.shape, initial, dtype=np.float32).astype(np.float64)
    moment, variance = np.zeros_like(expected), np.zeros_like(expected)
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(backend_fallback="error", auto_graph_replay=0):
        param = torch.nn.Parameter(torch.full(gradients.shape, initial, device=device))
        optimizer = torch.optim.AdamW([param], lr=1e-4, betas=(.9, .999),
                                      eps=1e-8, weight_decay=.01)
        for step in range(1, 11):
            scale = 1. if step <= 3 else -.5 if step <= 7 else 2.
            gradient = (gradients * np.float32(scale)).astype(np.float32)
            param.grad = torch.tensor(gradient, device=device)
            optimizer.step()
            grad64 = gradient.astype(np.float64)
            expected *= 1 - 1e-4 * .01
            moment = .9 * moment + .1 * grad64
            variance = .999 * variance + .001 * grad64 * grad64
            expected -= 1e-4 * (moment / (1 - .9 ** step)) / (
                np.sqrt(variance / (1 - .999 ** step)) + 1e-8)
            state = optimizer.state[param]
            assert int(state["step"].item()) == step
            assert param.grad.device.type == device
            for tensor in (param, param.grad, state["exp_avg"], state["exp_avg_sq"]):
                tensor.sync()
                assert tensor.device.type == device
                assert tensor.location() == ("device" if device == "cuda" else "cpu")
            actual = param.detach().cpu().numpy()
            assert actual.dtype == np.float32
            np.testing.assert_allclose(actual, expected,
                atol=1e-10 if initial == 0 else 1e-6,
                rtol=1e-5 if initial == 0 else 1e-4)
            np.testing.assert_allclose(state["exp_avg"].cpu().numpy(), moment,
                                       atol=8 * smallest, rtol=1e-5)
            np.testing.assert_allclose(state["exp_avg_sq"].cpu().numpy(), variance,
                atol=8 * smallest, rtol=1e-5 if device == "cpu" else 2e-5)
    assert jt.core.backend_fallback_count() == before
