"""Public AdamW groups retain supported options before state serialization."""
import copy
import numpy as np
import pytest
import jittor as jt
from jittor.compat.torch.tensor_state import compatibility_owner

_KEYS = ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize",
         "foreach", "capturable", "differentiable", "fused", "decoupled_weight_decay")


def _options(**overrides):
    result = dict(lr=.03, betas=(.7, .93), eps=1e-4, weight_decay=.02,
                  amsgrad=False, maximize=False, foreach=None, capturable=False,
                  differentiable=False, fused=None, decoupled_weight_decay=True)
    result.update(overrides)
    return result


def _parameter(torch, values):
    return torch.tensor(values, dtype=torch.float32, device="cpu", requires_grad=True)


def _optimizer(torch, params):
    return torch.optim.AdamW(params, lr=.03, betas=(.7, .93), eps=1e-4, weight_decay=.02)


def _public_options(group):
    return {key: group[key] for key in _KEYS}


def _array(tensor):
    return tensor.detach().float().cpu().numpy().copy()


def _assert_full_state_and_parameters(left, right, lhs, rhs):
    for p, q in zip(lhs, rhs):
        np.testing.assert_array_equal(_array(p), _array(q))
        a, b = left.state[p], right.state[q]
        assert set(a) == set(b) == {"step", "exp_avg", "exp_avg_sq"}
        assert float(a["step"]) == float(b["step"])
        for key in ("exp_avg", "exp_avg_sq"):
            assert a[key].dtype == b[key].dtype
            np.testing.assert_array_equal(_array(a[key]), _array(b[key]))


def test_adamw_publishes_complete_groups_and_stable_defaults_immediately():
    torch = compatibility_owner(jt)
    params = [_parameter(torch, [1., -2.]), _parameter(torch, [.25])]
    opt = _optimizer(torch, [{"params": [params[0]], "lr": .01},
                             {"params": [params[1]], "lr": .015}])
    assert _public_options(opt.param_groups[0]) == _options(lr=.01)
    assert _public_options(opt.param_groups[1]) == _options(lr=.015)
    assert set(opt.defaults) == set(_KEYS)
    assert opt.defaults == _options() and opt.defaults is opt.defaults


def test_adamw_public_group_rebind_keeps_two_full_parameter_and_moment_updates():
    torch = compatibility_owner(jt)
    left_params = [_parameter(torch, [[.1, -.2], [.3, -.4]]),
                   _parameter(torch, [.05, -.075])]
    right_params = [_parameter(torch, _array(p)) for p in left_params]
    left = _optimizer(torch, [{"params": [left_params[0]], "lr": .01},
                              {"params": [left_params[1]], "lr": .015}])
    # Rebind exactly the public dictionaries before asking for state_dict().
    right = torch.optim.AdamW([
        dict(group, params=[parameter])
        for group, parameter in zip(left.param_groups, right_params)
    ])
    assert [_public_options(g) for g in right.param_groups] == [
        _options(lr=.01), _options(lr=.015)]
    for step in range(2):
        for opt, parameters in ((left, left_params), (right, right_params)):
            opt.zero_grad(set_to_none=True)
            for index, p in enumerate(parameters):
                p.grad = torch.full_like(p, -.025 * (step + 1) * (index + 1))
            opt.step()
        _assert_full_state_and_parameters(left, right, left_params, right_params)


def test_adamw_explicit_groups_and_late_groups_override_live_defaults():
    torch = compatibility_owner(jt)
    first, second, late = [_parameter(torch, [v]) for v in (1., 2., 3.)]
    opt = _optimizer(torch, [
        {"params": [first], "lr": .01, "betas": (.4, .8), "eps": .002,
         "weight_decay": .04, "foreach": False, "fused": False},
        {"params": [second], "weight_decay": 0},
    ])
    assert _public_options(opt.param_groups[0]) == _options(
        lr=.01, betas=(.4, .8), eps=.002, weight_decay=.04, foreach=False, fused=False)
    assert _public_options(opt.param_groups[1]) == _options(weight_decay=0)
    opt.defaults["eps"] = .006
    opt.defaults["betas"] = (.6, .97)
    opt.add_param_group({"params": late, "lr": .02})
    assert opt.param_groups[2]["params"][0] is late
    assert _public_options(opt.param_groups[2]) == _options(lr=.02, eps=.006, betas=(.6, .97))
    assert opt.param_groups[1]["eps"] == 1e-4
    assert opt.param_groups[1]["betas"] == (.7, .93)


def test_adamw_group_metadata_and_full_state_roundtrip_preserve_next_update():
    torch = compatibility_owner(jt)
    parameters = [_parameter(torch, [1., -2.]), _parameter(torch, [.5])]
    source = _optimizer(torch, [{"params": [parameters[0]], "lr": .01},
                                {"params": [parameters[1]], "eps": .002}])
    for index, p in enumerate(parameters):
        p.grad = torch.full_like(p, .125 * (index + 1))
    source.step()
    saved = copy.deepcopy(source.state_dict())
    restored_parameters = [_parameter(torch, _array(p)) for p in parameters]
    restored = torch.optim.AdamW([
        {"params": [p]} for p in restored_parameters
    ], lr=.002, betas=(.5, .8), eps=.004, weight_decay=0)
    restored.load_state_dict(saved)
    assert [_public_options(g) for g in restored.param_groups] == [
        _public_options(g) for g in source.param_groups]
    assert restored.defaults == _options(lr=.002, betas=(.5, .8), eps=.004, weight_decay=0)
    _assert_full_state_and_parameters(source, restored, parameters, restored_parameters)
    for opt, params in ((source, parameters), (restored, restored_parameters)):
        opt.zero_grad(set_to_none=True)
        for index, p in enumerate(params):
            p.grad = torch.full_like(p, -.25 * (index + 1))
        opt.step()
    _assert_full_state_and_parameters(source, restored, parameters, restored_parameters)


@pytest.mark.parametrize("option", ("amsgrad", "maximize", "differentiable", "capturable"))
def test_adamw_unsupported_variants_reject_ctor_group_add_and_step_before_update(option):
    torch = compatibility_owner(jt)
    parameter = _parameter(torch, [1., -2.])
    with pytest.raises(NotImplementedError, match=option):
        torch.optim.AdamW([parameter], **{option: True})
    with pytest.raises(NotImplementedError, match=option):
        torch.optim.AdamW([{"params": [parameter], option: True}])
    opt = _optimizer(torch, [parameter])
    late = _parameter(torch, [.5])
    with pytest.raises(NotImplementedError, match=option):
        opt.add_param_group({"params": [late], option: True})
    assert len(opt.param_groups) == 1
    parameter.grad = torch.ones_like(parameter)
    before = _array(parameter)
    opt.param_groups[0][option] = True
    with pytest.raises(NotImplementedError, match=option):
        opt.step()
    np.testing.assert_array_equal(_array(parameter), before)
    assert opt.state_dict()["state"] == {}


def test_adamw_public_defaults_do_not_rewrite_native_optimizer_policy():
    value = jt.array([1., 2.]).start_grad()
    native = jt.optim.AdamW([value], .01)
    assert native.defaults["param_groups"] is native.param_groups
    assert not hasattr(native, "_torch_adamw_defaults")

@pytest.mark.parametrize("entry", ("constructor", "group", "late_group", "step"))
def test_adamw_foreach_supported_selection_preserves_first_update(entry):
    torch = compatibility_owner(jt)
    parameter = _parameter(torch, [1., -2.])
    if entry == "constructor":
        optimizer = torch.optim.AdamW([parameter], lr=.01, foreach=True)
    elif entry == "group":
        optimizer = torch.optim.AdamW([{"params": [parameter], "foreach": True}], lr=.01)
    elif entry == "late_group":
        first = _parameter(torch, [3.])
        optimizer = torch.optim.AdamW([first], lr=.01)
        optimizer.add_param_group({"params": [parameter], "foreach": True})
    else:
        optimizer = torch.optim.AdamW([parameter], lr=.01)
        optimizer.param_groups[0]["foreach"] = True
    parameter.grad = _parameter(torch, [2., -4.]).detach()
    optimizer.step()
    np.testing.assert_allclose(_array(parameter), [.9899, -1.9898], atol=2e-7, rtol=0)
    assert float(optimizer.state[parameter]["step"]) == 1
