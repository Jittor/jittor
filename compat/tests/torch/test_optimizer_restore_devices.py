"""Optimizer checkpoint tensors follow their own parameter on restoration."""
import numpy as np
import pytest
import torch
import jittor as jt

from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.fixture(params=("cpu", "cuda:0"))
def device(request):
    if request.param == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


@pytest.fixture
def execution(device):
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=int(device == "cuda:0"), backend_fallback="error",
                          auto_graph_replay=0), forbid_backend_fallbacks():
        yield
    assert jt.core.backend_fallback_count() == before


def _values(tensor):
    return tensor.detach().float().cpu().numpy().copy()


def _placement(tensor, device):
    assert str(tensor.device) == device
    assert tuple(jt.core.dispatch_context([tensor])) == (
        ("cuda", 0) if device == "cuda:0" else ("cpu", -1))


def _optimizer(kind, params):
    groups = [{"params": [params[0]], "lr": .01},
              {"params": [params[1]], "lr": .03}]
    if kind == "adamw":
        return torch.optim.AdamW(groups, lr=.02, weight_decay=0.)
    return torch.optim.SGD(groups, lr=.02, momentum=.9)


def _step(opt, params):
    opt.zero_grad()
    # Distinct CPU/CUDA groups cannot be combined into a single scalar.
    for p in params:
        (p.float() * p.float()).sum().backward()
    opt.step()


def _make_params(device, dtype):
    return [torch.nn.Parameter(torch.tensor([1., -2.], device="cpu")),
            torch.nn.Parameter(torch.tensor([2., -3.], device=device, dtype=dtype))]


def _fields(kind):
    return ("exp_avg", "exp_avg_sq") if kind == "adamw" else ("momentum_buffer",)


@pytest.mark.parametrize("kind,dtype", [("adamw", torch.float32),
                                         ("adamw", torch.bfloat16),
                                         ("sgd", torch.float32)])
def test_map_cpu_restore_to_per_parameter_device_and_next_step(tmp_path, device, execution, kind, dtype):
    reference_params = _make_params(device, dtype)
    reference = _optimizer(kind, reference_params)
    _step(reference, reference_params)
    checkpoint = tmp_path / "optimizer.pt"
    torch.save(reference.state_dict(), checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for state in loaded["state"].values():
        for field in _fields(kind):
            _placement(state[field], "cpu")

    restored_params = [torch.nn.Parameter(torch.tensor(_values(p), dtype=p.dtype,
                                                       device=str(p.device)))
                       for p in reference_params]
    restored = _optimizer(kind, restored_params)
    # A frontend default is not the placement of each optimizer parameter.
    # Keep both groups in the same load while the default is deliberately
    # different from at least one of them.
    ambient = "cuda:0" if device == "cuda:0" else "meta"
    with torch.device(ambient):
        restored.load_state_dict(loaded)
    for original, target in zip(reference_params, restored_params):
        source_state, target_state = reference.state[original], restored.state[target]
        _placement(target, str(original.device))
        for field in _fields(kind):
            expected, actual = source_state[field], target_state[field]
            _placement(actual, str(target.device))
            assert tuple(actual.shape) == tuple(target.shape)
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(_values(actual), _values(expected))
        if kind == "adamw":
            assert float(target_state["step"]) == float(source_state["step"]) == 1
            if target.dtype == torch.bfloat16:
                assert target_state["exp_avg"].dtype == target.dtype
                assert target_state["exp_avg_sq"].dtype == target.dtype
    assert [group["lr"] for group in restored.param_groups] == [.01, .03]

    _step(reference, reference_params)
    _step(restored, restored_params)
    for original, target in zip(reference_params, restored_params):
        np.testing.assert_allclose(_values(target), _values(original), rtol=1e-5, atol=1e-6)
        for field in _fields(kind):
            actual = restored.state[target][field]
            _placement(actual, str(target.device))
            np.testing.assert_allclose(_values(actual), _values(reference.state[original][field]),
                                       rtol=1e-5, atol=1e-6)
        if kind == "adamw":
            assert float(restored.state[target]["step"]) == 2


@pytest.mark.parametrize("kind", ["adamw", "sgd"])
@pytest.mark.parametrize("fault", ["step", "shape", "type"])
def test_invalid_later_slot_does_not_change_live_optimizer(device, execution, kind, fault):
    params = _make_params(device, torch.float32)
    optimizer = _optimizer(kind, params)
    _step(optimizer, params)
    before = [{field: (_values(optimizer.state[p][field]),
                       optimizer.state[p][field].data_ptr())
               for field in _fields(kind)} for p in params]
    before_lrs = [group["lr"] for group in optimizer.param_groups]
    before_step = optimizer.n_step
    saved = optimizer.state_dict()
    bad = {"param_groups": [dict(g) for g in saved["param_groups"]],
           "state": {key: dict(value) for key, value in saved["state"].items()}}
    last_id = bad["param_groups"][1]["params"][0]
    bad["param_groups"][0]["lr"] = .9
    if fault == "step":
        bad["state"][last_id]["step"] = -1
    elif fault == "shape":
        bad["state"][last_id][_fields(kind)[0]] = torch.tensor([8.], device="cpu")
    else:
        bad["state"][last_id][_fields(kind)[0]] = "not a tensor"
    with pytest.raises((TypeError, ValueError)):
        optimizer.load_state_dict(bad)
    assert optimizer.n_step == before_step
    assert [group["lr"] for group in optimizer.param_groups] == before_lrs
    for p, expected in zip(params, before):
        for field, (values, pointer) in expected.items():
            actual = optimizer.state[p][field]
            assert actual.data_ptr() == pointer
            _placement(actual, str(p.device))
            np.testing.assert_array_equal(_values(actual), values)


@pytest.mark.parametrize("kind", ["adamw", "sgd"])
def test_empty_saved_state_clears_existing_slots_without_creating_entries(device, execution, kind):
    source_params = _make_params(device, torch.float32)
    empty = _optimizer(kind, source_params).state_dict()
    assert not empty["state"]
    target_params = _make_params(device, torch.float32)
    target = _optimizer(kind, target_params)
    _step(target, target_params)
    assert all(target.state.get(p) for p in target_params)
    target.load_state_dict(empty)
    assert all(target.state.get(p) is None for p in target_params)
    assert not target.state_dict()["state"]
    assert target.n_step == 0


@pytest.mark.parametrize("kind", ["adamw", "sgd"])
def test_fresh_empty_restore_reuses_initial_zero_slots_without_materialization(
        device, execution, kind, monkeypatch):
    params = _make_params(device, torch.float32)
    optimizer = _optimizer(kind, params)
    empty = optimizer.state_dict()
    assert not empty["state"]
    slot_names = ("m", "values") if kind == "adamw" else ("values",)
    before = [[list(group[name]) for name in slot_names]
              for group in optimizer.param_groups]
    for old_slots in before:
        for buffers in old_slots:
            for buffer in buffers:
                values = _values(buffer)
                np.testing.assert_array_equal(values, np.zeros_like(values))

    def unexpected_zero_allocation(*args, **kwargs):
        raise AssertionError("fresh step-zero restore must reuse zero slots")
    monkeypatch.setattr(jt, "zeros_like", unexpected_zero_allocation)

    optimizer.load_state_dict(empty)

    for group, old_slots in zip(optimizer.param_groups, before):
        for name, old_values in zip(slot_names, old_slots):
            assert all(new is old for new, old in zip(group[name], old_values))
    assert optimizer.n_step == 0
    assert not optimizer.state_dict()["state"]


def test_empty_restore_clears_loaded_step_zero_adamw_moments(device, execution):
    params = _make_params(device, torch.float32)
    optimizer = _optimizer("adamw", params)
    empty = optimizer.state_dict()
    param_id = empty["param_groups"][0]["params"][0]
    step_zero = {
        "param_groups": empty["param_groups"],
        "state": {
            param_id: {
                "step": 0,
                "exp_avg": torch.tensor([.25, -.5], dtype=torch.float32),
                "exp_avg_sq": torch.tensor([.5, .75], dtype=torch.float32),
            }
        },
    }

    optimizer.load_state_dict(step_zero)
    np.testing.assert_array_equal(
        _values(optimizer.param_groups[0]["m"][0]), np.array([.25, -.5], np.float32))
    np.testing.assert_array_equal(
        _values(optimizer.param_groups[0]["values"][0]), np.array([.5, .75], np.float32))

    optimizer.load_state_dict(empty)
    np.testing.assert_array_equal(
        _values(optimizer.param_groups[0]["m"][0]), np.zeros(2, np.float32))
    np.testing.assert_array_equal(
        _values(optimizer.param_groups[0]["values"][0]), np.zeros(2, np.float32))
