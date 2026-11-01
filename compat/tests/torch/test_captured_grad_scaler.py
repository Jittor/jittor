"""CUDA regression for GradScaler overflow inside a captured AdamW step."""
import os
from contextlib import nullcontext

import numpy as np
import pytest
import torch

try:
    import jittor as jt
    from accelerate.state import AcceleratorState, GradientState
except ImportError:
    jt = None
    AcceleratorState = GradientState = None


_DEVICE = os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(",")[0]
_SHIM = hasattr(torch, "_torch_compat_install_context")


def _values(value):
    return value.detach().clone().cpu().numpy().copy()


@pytest.fixture(autouse=True)
def _isolated_accelerator_state():
    if AcceleratorState is not None:
        AcceleratorState._reset_state(reset_partial_state=True)
        GradientState._shared_state.clear()
    fallback_before = int(jt.core.backend_fallback_count()) if jt is not None else None
    scope = (jt.runtime.scope(use_cuda=int(_DEVICE == "cuda"),
                              backend_fallback="error")
             if jt is not None else nullcontext())
    with scope:
        if jt is None:
            yield
        else:
            from jittor._runtime.fallback import forbid_backend_fallbacks
            with forbid_backend_fallbacks():
                yield
    if jt is not None:
        assert int(jt.core.backend_fallback_count()) == fallback_before
    if AcceleratorState is not None:
        AcceleratorState._reset_state(reset_partial_state=True)
        GradientState._shared_state.clear()


@pytest.mark.skipif(_DEVICE != "cuda" or jt is None or not _SHIM,
                    reason="requires the Torch compatibility CUDA path")
def test_captured_adamw_grad_scaler_skips_overflow_and_resumes():
    if not torch.cuda.is_available():
        pytest.skip("requires a real CUDA device")

    def make_training_state():
        parameter = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
        optimizer = torch.optim.AdamW(
            [parameter], lr=0.1, weight_decay=0.0, fused=True)
        scaler = torch.amp.GradScaler(
            "cuda", init_scale=8.0, growth_interval=100)

        def train_step(feed):
            optimizer.zero_grad(set_to_none=True)
            loss = (parameter * feed).sum()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            return loss

        return parameter, optimizer, scaler, train_step

    captured_parameter, captured_optimizer, captured_scaler, captured_fn = (
        make_training_state())
    eager_parameter, eager_optimizer, eager_scaler, eager_fn = make_training_state()
    jt.sync_all(True)
    for parameter in (captured_parameter, eager_parameter):
        assert parameter.device.type == "cuda"
        assert parameter.placement_backend == 1
        assert parameter.device_id == 0
        assert parameter.location() == "device"

    def finite():
        return torch.tensor([2.0], dtype=torch.float32, device="cuda")

    def overflow():
        return torch.tensor([float("inf")], dtype=torch.float32, device="cuda")

    captured = jt.capture_step(captured_fn)
    captured(finite())       # first signature initializes the optimizer eagerly
    captured(finite())       # the next call records the update
    assert captured.refused is None, captured.refused
    assert captured.stats["captured"] == 1
    captured(overflow())     # replay with found_inf set on device
    captured(finite())       # a finite replay must recover and update

    for feed in (finite(), finite(), overflow(), finite()):
        eager_fn(feed)

    assert captured.stats["replayed"] == 2
    assert captured_scaler.get_scale() == eager_scaler.get_scale() == 4.0
    assert captured_scaler.state_dict() == eager_scaler.state_dict()

    def assert_state_matches_eager():
        for name in ("parameter", "exp_avg", "exp_avg_sq"):
            actual = (captured_parameter if name == "parameter" else
                      captured_optimizer.state[captured_parameter][name])
            expected = (eager_parameter if name == "parameter" else
                        eager_optimizer.state[eager_parameter][name])
            np.testing.assert_allclose(
                _values(actual), _values(expected), rtol=1e-6, atol=1e-7,
                err_msg="captured recovery diverged for %s" % name)
        assert float(captured_optimizer.state[captured_parameter]["step"]) == float(
            eager_optimizer.state[eager_parameter]["step"])
        captured_state = captured_optimizer.state_dict()["state"]
        eager_state = eager_optimizer.state_dict()["state"]
        captured_step = next(iter(captured_state.values()))["step"]
        eager_step = next(iter(eager_state.values()))["step"]
        assert float(_values(captured_step).reshape(-1)[0]) == float(
            _values(eager_step).reshape(-1)[0])

    # Public state reads settle the device skip count back into the Python
    # optimizer book. The existing capture must then rebuild from that step.
    assert_state_matches_eager()
    captured(finite())
    eager_fn(finite())
    assert captured.stats["captured"] >= 2
    assert_state_matches_eager()
    assert captured_scaler.get_scale() == eager_scaler.get_scale() == 4.0
