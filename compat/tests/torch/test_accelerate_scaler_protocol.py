"""Independent AMP state-machine and calculation contracts used by Accelerate."""
import os
from contextlib import nullcontext

import numpy as np
import pytest
import torch
try:
    from accelerate.state import AcceleratorState, GradientState
except ModuleNotFoundError as exc:
    if exc.name != "accelerate" or os.environ.get("JITTOR_REQUIRE_ACCELERATE") == "1":
        raise
    AcceleratorState = GradientState = None

pytestmark = pytest.mark.skipif(AcceleratorState is None,
                                reason="optional dependency accelerate is not installed")


_DEVICE = os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(",")[0]
_SHIM = hasattr(torch, "_torch_compat_install_context")
if _SHIM:
    import jittor as jt


def _values(value):
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.detach().clone().cpu().numpy().copy()


def _parameter(value):
    return torch.nn.Parameter(torch.tensor([value], device=_DEVICE))


@pytest.fixture(autouse=True)
def _isolated_state():
    if _DEVICE == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a real CUDA device")
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._shared_state.clear()
    scope = jt.flag_scope(use_cuda=int(_DEVICE == "cuda")) if _SHIM else nullcontext()
    with scope:
        yield
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._shared_state.clear()


def test_bf16_autocast_preserves_fp32_master_weights_and_dynamic_range():
    x = torch.full((2, 2), 100000., device=_DEVICE)
    weight = torch.nn.Parameter(torch.ones((2, 2), device=_DEVICE))
    optimizer = torch.optim.SGD([weight], lr=.1)
    with torch.autocast(_DEVICE, dtype=torch.bfloat16):
        result = torch.matmul(x, weight)
        assert result.dtype == torch.bfloat16
        assert result.device.type == _DEVICE
        assert weight.dtype == torch.float32
        np.testing.assert_array_equal(_values(result), np.full((2, 2), 199680.))
        with torch.autocast(_DEVICE, enabled=False):
            assert torch.matmul(x, weight).dtype == torch.float32
        assert torch.matmul(x, weight).dtype == torch.bfloat16
        double_input = x.double()
        double_result = torch.matmul(double_input, weight.double())
        assert double_result.dtype == torch.float64
        np.testing.assert_array_equal(_values(double_result), np.full((2, 2), 200000.))
        if _SHIM:
            assert double_input.float_auto().dtype == torch.float64
    result.float().sum().backward()
    assert weight.grad.dtype == torch.float32
    assert torch.isfinite(weight.grad).all().item()
    optimizer.step()
    assert weight.dtype == torch.float32


@pytest.mark.parametrize("kind", ("linear", "conv2d"))
def test_bf16_autocast_casts_bias_for_computation_without_mutating_parameters(kind):
    if kind == "linear":
        model = torch.nn.Linear(2, 2).to(_DEVICE)
        data = torch.full((2, 2), 100000., device=_DEVICE)
    else:
        model = torch.nn.Conv2d(1, 1, 1).to(_DEVICE)
        data = torch.full((1, 1, 2, 2), 100000., device=_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    with torch.no_grad():
        model.weight.fill_(1.)
        model.bias.zero_()
    with torch.autocast(_DEVICE, dtype=torch.bfloat16):
        result = model(data)
    assert result.dtype == torch.bfloat16
    assert torch.isfinite(result).all().item()
    result.float().sum().backward()
    for parameter in model.parameters():
        assert parameter.dtype == torch.float32
        assert parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all().item()
    optimizer.step()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    for parameter in model.parameters():
        assert optimizer.state[parameter]["exp_avg"].dtype == torch.float32
        assert optimizer.state[parameter]["exp_avg_sq"].dtype == torch.float32


@pytest.mark.parametrize("reverse", (False, True))
def test_scaler_tracks_unscale_and_overflow_per_optimizer(reverse):
    a, b = _parameter(1.), _parameter(2.)
    oa, ob = torch.optim.SGD([a], lr=.1), torch.optim.SGD([b], lr=.1)
    scaler = torch.amp.GradScaler(_DEVICE, init_scale=8)
    scaler.scale((a * float("inf") + b * 2).sum()).backward()
    optimizers = (ob, oa) if reverse else (oa, ob)
    for optimizer in optimizers:
        scaler.unscale_(optimizer)
    for optimizer in optimizers:
        scaler.step(optimizer)
    scaler.update()
    np.testing.assert_array_equal(_values(a), [1.])
    np.testing.assert_allclose(_values(b), [1.8], rtol=0, atol=2e-7)
    np.testing.assert_array_equal(_values(b.grad), [2.])
    assert scaler.get_scale() == 4.


def test_scaler_rejects_invalid_unscale_and_step_order():
    p = _parameter(1.)
    optimizer = torch.optim.SGD([p], lr=.1)
    scaler = torch.amp.GradScaler(_DEVICE, init_scale=8)
    with pytest.raises(AssertionError, match="_scale is None"):
        scaler.update()
    scaler.scale(p.square().sum()).backward()
    scaler.unscale_(optimizer)
    with pytest.raises(RuntimeError, match="already been called"):
        scaler.unscale_(optimizer)
    scaler.step(optimizer)
    with pytest.raises(RuntimeError, match="already been called"):
        scaler.step(optimizer)
    with pytest.raises(RuntimeError, match="after step"):
        scaler.unscale_(optimizer)
    scaler.update()
    with pytest.raises(AssertionError, match="No inf checks"):
        scaler.update()


def test_scaler_rejects_missing_gradients_and_enabled_closures():
    p = _parameter(1.)
    optimizer = torch.optim.SGD([p], lr=.1)
    scaler = torch.amp.GradScaler(_DEVICE)
    scaler.scale(p.sum())
    with pytest.raises(RuntimeError, match="Closure use"):
        scaler.step(optimizer, closure=lambda: p.sum())
    with pytest.raises(AssertionError, match="No inf checks"):
        scaler.step(optimizer)


def test_scaler_rejects_fp16_gradients():
    p = torch.nn.Parameter(torch.ones(2, device=_DEVICE, dtype=torch.float16))
    optimizer = torch.optim.SGD([p], lr=.1)
    scaler = torch.amp.GradScaler(_DEVICE, init_scale=8)
    scaler.scale(p.float().sum()).backward()
    with pytest.raises(ValueError, match="unscale FP16"):
        scaler.unscale_(optimizer)


def test_scaler_grows_and_backs_off_below_one():
    p = _parameter(1.)
    optimizer = torch.optim.SGD([p], lr=.1)
    scaler = torch.amp.GradScaler(_DEVICE, init_scale=.5,
                                 growth_factor=4, backoff_factor=.25, growth_interval=2)
    for finite, expected_scale, expected_tracker in (
            (False, .125, 0), (True, .125, 1), (True, .5, 0), (False, .125, 0)):
        optimizer.zero_grad(set_to_none=True)
        before = _values(p)
        scaler.scale((p * (1. if finite else float("inf"))).sum()).backward()
        scaler.step(optimizer)
        scaler.update()
        assert scaler.get_scale() == expected_scale
        assert scaler.state_dict()["_growth_tracker"] == expected_tracker
        if not finite:
            np.testing.assert_array_equal(_values(p), before)


def test_scaler_state_restores_policy_and_continuation(tmp_path):
    p = _parameter(1.)
    optimizer = torch.optim.SGD([p], lr=.1)
    scaler = torch.amp.GradScaler(_DEVICE, init_scale=8,
                                 growth_factor=4, backoff_factor=.25, growth_interval=2)
    scaler.scale(p.square().sum()).backward()
    scaler.step(optimizer)
    scaler.update()
    state = scaler.state_dict()
    assert set(state) == {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}
    path = tmp_path / "scaler.bin"
    torch.save(state, path)
    restored = torch.amp.GradScaler(_DEVICE)
    restored.load_state_dict(torch.load(path, weights_only=False))
    assert restored.state_dict() == state
    q = _parameter(float(_values(p)[0]))
    other_optimizer = torch.optim.SGD([q], lr=.1)
    optimizer.zero_grad(set_to_none=True)
    for target, opt, scale in ((p, optimizer, scaler), (q, other_optimizer, restored)):
        scale.scale(target.square().sum()).backward()
        scale.step(opt)
        scale.update()
    np.testing.assert_array_equal(_values(q), _values(p))
    assert restored.state_dict() == scaler.state_dict()


def test_scaler_new_positional_signature_disabled_and_nested_outputs():
    scaler = torch.amp.GradScaler(_DEVICE, 8, 4, .25, 2, True)
    assert scaler.state_dict()["growth_factor"] == 4
    assert scaler.state_dict()["backoff_factor"] == .25
    assert scaler.state_dict()["growth_interval"] == 2
    p = _parameter(1.)
    outputs = [p, (p * 2,)]
    scaled = scaler.scale(outputs)
    if _SHIM:
        assert scaler._scale.ndim == 0
        assert jt.core.dispatch_context([scaler._scale]) == jt.core.dispatch_context([p])
    assert isinstance(scaled, list) and isinstance(scaled[1], tuple)
    np.testing.assert_array_equal(_values(scaled[0]), [8.])
    np.testing.assert_array_equal(_values(scaled[1][0]), [16.])
    disabled = torch.amp.GradScaler(_DEVICE, enabled=False)
    assert disabled.scale(outputs) is outputs
    assert disabled.state_dict() == {}
    disabled.load_state_dict({})
    assert disabled.get_scale() == 1.
    with pytest.raises(RuntimeError, match="source state dict is empty"):
        scaler.load_state_dict({})


def test_scaler_growth_does_not_overflow_to_infinity():
    p = _parameter(1.)
    optimizer = torch.optim.SGD([p], lr=.1)
    initial = float(np.float32(2. ** 127))
    scaler = torch.amp.GradScaler(_DEVICE, init_scale=initial, growth_interval=1)
    scaler.scale((p * 0.).sum()).backward()
    scaler.step(optimizer)
    scaler.update()
    assert scaler.get_scale() == initial
    assert scaler.state_dict()["_growth_tracker"] == 0


def test_scaler_legacy_positional_signature_when_cuda_is_available():
    if _DEVICE != "cuda":
        pytest.skip("legacy CUDA GradScaler requires CUDA")
    scaler = torch.cuda.amp.GradScaler(8, 4, .25, 2, True)
    assert scaler.is_enabled()
    assert scaler.state_dict() == {
        "scale": 8., "growth_factor": 4., "backoff_factor": .25,
        "growth_interval": 2, "_growth_tracker": 0,
    }


def test_scaler_manual_scale_updates_keep_copies_current_and_allow_zero():
    p = _parameter(1.)
    optimizer = torch.optim.SGD([p], lr=.1)
    scaler = torch.amp.GradScaler(_DEVICE, init_scale=8)
    scaler.scale(p)
    cpu = torch.ones(2, device="cpu")
    np.testing.assert_array_equal(_values(scaler.scale(cpu)), [8., 8.])
    scaler.update(2.)
    np.testing.assert_array_equal(_values(scaler.scale(cpu)), [2., 2.])
    scaler.update(torch.tensor(4., device=_DEVICE))
    np.testing.assert_array_equal(_values(scaler.scale(cpu)), [4., 4.])
    scaler.update(0.)
    scaler.scale(p.sum()).backward()
    scaler.step(optimizer)
    scaler.update()
    assert scaler.get_scale() == 0.
    # Torch checks the scaled gradient for non-finites before multiplying by
    # the inverse scale. A zero scale yields 0 * inf during unscale and NaN.
    assert np.isnan(_values(p)).all()


def test_scaler_float64_nonfinite_check_matches_backend_detection_domain():
    for magnitude in (1e20, 1e100):
        p = torch.nn.Parameter(torch.ones(1, dtype=torch.float64, device=_DEVICE))
        optimizer = torch.optim.SGD([p], lr=.1 / magnitude)
        scaler = torch.amp.GradScaler(_DEVICE, init_scale=8)
        scaler.scale((p * magnitude).sum()).backward()
        assert np.isfinite(_values(p.grad)).all()
        scaler.step(optimizer)
        scaler.update()
        skipped = _DEVICE == "cuda" and magnitude == 1e100
        assert scaler.get_scale() == (4. if skipped else 8.)
        assert p.grad.dtype == torch.float64
        assert p.grad.device.type == _DEVICE
        np.testing.assert_allclose(_values(p.grad), [magnitude], rtol=1e-14)
        np.testing.assert_allclose(_values(p), [1. if skipped else .9], rtol=1e-14)


@pytest.mark.parametrize("dimension_dtype,scalar_dtype,expected_dtype", (
    (torch.float16, torch.float32, torch.float16),
    (torch.int16, torch.int64, torch.int16),
    (torch.int16, torch.float32, torch.float32),
))
def test_zero_dimensional_scalar_promotion_and_reflected_arithmetic(
        dimension_dtype, scalar_dtype, expected_dtype):
    vector = torch.tensor([1, 2], dtype=dimension_dtype, device=_DEVICE)
    scalar = torch.tensor(4, dtype=scalar_dtype, device=_DEVICE)
    assert scalar.ndim == 0
    for result in (vector + scalar, scalar + vector):
        assert result.dtype == expected_dtype
        np.testing.assert_array_equal(_values(result), [5, 6])
    for result, expected in ((vector - scalar, [-3, -2]), (scalar - vector, [3, 2])):
        assert result.dtype == expected_dtype
        np.testing.assert_array_equal(_values(result), expected)
    division_dtype = torch.float32 if expected_dtype in (torch.int16, torch.int64) else expected_dtype
    for result, expected in ((vector / scalar, [.25, .5]), (scalar / vector, [4., 2.])):
        assert result.dtype == division_dtype
        np.testing.assert_array_equal(_values(result), expected)


def test_half_weak_scalar_and_typed_scale_keep_distinct_backend_protocols():
    p = torch.nn.Parameter(torch.tensor([.25, -.125], dtype=torch.float16, device=_DEVICE))
    weak = p * 65536.
    np.testing.assert_array_equal(_values(weak), [16384., -8192.])
    np.testing.assert_array_equal(_values(65536. * p), [16384., -8192.])
    for result in (p.mul(65536.), torch.mul(p, 65536.), torch.multiply(p, 65536.)):
        assert result.dtype == torch.float16
        np.testing.assert_array_equal(_values(result), [16384., -8192.])
    if _DEVICE == "cpu":
        with pytest.raises(RuntimeError, match="without overflow"):
            torch.add(torch.zeros_like(p), p, alpha=65536.)
    else:
        added = torch.add(torch.zeros_like(p), p, alpha=65536.)
        assert added.dtype == torch.float16
        np.testing.assert_array_equal(_values(added), [16384., -8192.])
    scaler = torch.amp.GradScaler(_DEVICE)
    scaled = scaler.scale(p)
    assert scaled.dtype == torch.float16
    scalar = torch.tensor(65536., dtype=torch.float32, device=_DEVICE)
    typed = p * scalar
    assert typed.dtype == torch.float16
    if _DEVICE == "cuda":
        assert np.isposinf(_values(scaled)[0]) and np.isneginf(_values(scaled)[1])
    else:
        np.testing.assert_array_equal(_values(scaled), [16384., -8192.])
    np.testing.assert_array_equal(_values(scaled), _values(typed))
    for result in (p.mul(scalar), torch.mul(p, scalar), torch.multiply(p, scalar)):
        assert result.dtype == torch.float16
        np.testing.assert_array_equal(_values(result), _values(typed))


def test_public_sqrt_autocast_preserves_dtype_values_and_backward():
    p = torch.nn.Parameter(torch.tensor([.25, 1.], device=_DEVICE))
    with torch.autocast(_DEVICE, dtype=torch.bfloat16):
        output = torch.sqrt(p)
        method_output = p.sqrt()
    assert output.dtype == torch.float32
    assert method_output.dtype == torch.float32
    np.testing.assert_array_equal(_values(output), [.5, 1.])
    np.testing.assert_array_equal(_values(method_output), [.5, 1.])
    output.sum().backward()
    np.testing.assert_array_equal(_values(p.grad), [1., .5])
