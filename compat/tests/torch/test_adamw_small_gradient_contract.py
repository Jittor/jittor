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


@pytest.mark.parametrize("captured", [False, True])
def test_adam_host_scalars_preserve_small_variances(device, captured):
    if device not in ("cpu", "cuda"):
        pytest.skip("This FP32 scalar reference is calibrated for CPU and CUDA")
    if captured and device != "cuda":
        pytest.skip("CUDA graph replay requires real CUDA")

    gradients = np.array([0., 1e-12, -1e-12, 1e-10, -1e-10, 1e-8,
                          -1e-8, 1e-6, -1e-6, 1e-3, -1e-3], dtype="float32")
    expected = np.full(gradients.shape, .25, dtype="float64")
    moment, variance = np.zeros_like(expected), np.zeros_like(expected)
    smallest = float(np.nextafter(np.float32(0), np.float32(1)))
    before = jt.core.backend_fallback_count()
    runner = None
    with jt.flag_scope(backend_fallback="error", auto_graph_replay=0):
        parameter = torch.nn.Parameter(torch.full(gradients.shape, .25, device=device))
        optimizer = torch.optim.Adam([parameter], lr=.0012, eps=1e-8)

        def step(gradient):
            optimizer.zero_grad(set_to_none=True)
            loss = (parameter * gradient).sum()
            loss.backward()
            snapshot = parameter.grad.detach().clone()
            optimizer.step()
            return loss.detach(), snapshot

        runner = torch.compile(step, mode="reduce-overhead", dynamic=False) if captured else step
        try:
            for index in range(16):
                rate = .0012 + .0003 * (index % 4)
                beta1, beta2 = .95 - .01 * (index % 5), .999
                optimizer.param_groups[0].update(lr=rate, betas=(beta1, beta2))
                gradient = gradients * np.float32((1., -.5, 2.)[index % 3])
                _, observed_gradient = runner(torch.tensor(gradient, device=device))
                grad64 = gradient.astype("float64")
                moment = beta1 * moment + (1 - beta1) * grad64
                variance = beta2 * variance + (1 - beta2) * grad64 * grad64
                expected -= rate * (moment / (1 - beta1 ** (index + 1))) / (
                    np.sqrt(variance / (1 - beta2 ** (index + 1))) + 1e-8)

                # Read the existing state buffers without publishing a GPU
                # step counter before tracing: that counter's authoritative
                # host reads would make this a different capture scenario.
                group = optimizer.param_groups[0]
                values = (parameter, group["m"][0], group["values"][0], observed_gradient)
                snapshots = []
                for value in values:
                    value.sync()
                    assert value.device.type == device
                    assert value.location() == ("device" if device == "cuda" else "cpu")
                    snapshot = value.detach().cpu().numpy().copy()
                    assert snapshot.dtype == np.float32
                    snapshots.append(snapshot)
                np.testing.assert_array_equal(snapshots[3], gradient)
                np.testing.assert_allclose(snapshots[0], expected, atol=2e-7, rtol=2e-6)
                np.testing.assert_allclose(snapshots[1], moment, atol=8 * smallest, rtol=2e-6)
                # Native default/foreach CUDA FP32 is below this relative
                # budget. An absolute 1e-5 tolerance would hide this error.
                np.testing.assert_allclose(snapshots[2], variance, atol=8 * smallest, rtol=3e-6)
            assert int(optimizer.state[parameter]["step"].item()) == 16
            if captured:
                assert runner.stats["graph"] >= 8, (runner.stats, runner._graph_refused)
        finally:
            if captured:
                runner.invalidate()
    assert jt.core.backend_fallback_count() == before
