"""Dense Adagrad trajectories against an independent binary PyTorch oracle.

Select CPU or NPU with JITTOR_TEST_DEVICES (one backend per process). This tests
fixed gradients, not distributed optimizer or ZeRO/FSDP support.
"""
import inspect
import json
import os
import subprocess

import numpy as np
import pytest
import torch

from _helpers.child_process import child_env, default_timeout


def _trajectory(torch_module, device):
    import copy
    t = torch_module
    assert t.float32 == t.tensor(0., dtype=t.float32, device=device).dtype
    params = [t.nn.Parameter(t.tensor(values, dtype=t.float32, device=device))
              for values in ([1., -2., .5], [-.25, .75], [2., -1.])]
    groups = [{"params": params[:2], "lr": .13, "lr_decay": .07,
               "weight_decay": .11},
              {"params": params[2:], "lr": .09, "lr_decay": .03,
               "weight_decay": .04, "maximize": True}]
    opt = t.optim.Adagrad(groups, initial_accumulator_value=.2, eps=1e-6,
                          foreach=False)

    def snapshot():
        result = []
        for p in params:
            state = opt.state[p]
            assert state["sum"].device == p.device
            assert state["sum"].dtype == t.float32
            assert state["step"].dtype == t.float32
            assert state["step"].device.type == "cpu"
            assert p.device.type == device.split(":")[0]
            result.append({"param": p.detach().cpu().tolist(),
                           "sum": state["sum"].detach().cpu().tolist(),
                           "step": float(state["step"].item()),
                           "grad": (None if p.grad is None else p.grad.detach().cpu().tolist())})
        return result

    records = [snapshot()]  # Adagrad initializes state even before a gradient.
    gradients = [
        ([.2, -.3, 0.], None, [.4, -.5]),
        (None, [0., .1], [0., 0.]),
        ([.7, .1, -.4], [-.3, .8], None),
    ]
    for grads in gradients:
        opt.zero_grad(set_to_none=True)
        for p, grad in zip(params, grads):
            p.grad = None if grad is None else t.tensor(grad, dtype=t.float32, device=device)
        assert opt.step() is None
        records.append(snapshot())
    opt.zero_grad(set_to_none=False)
    records.append(snapshot())
    opt.step()  # zero gradients still advance counters and apply weight decay.
    records.append(snapshot())

    # Reload copied state into a new optimizer, then continue from the exact
    # same parameter values. Groups/options must be restored from the state.
    checkpoint = copy.deepcopy(opt.state_dict())
    params = [t.nn.Parameter(p.detach().clone()) for p in params]
    opt = t.optim.Adagrad([{"params": params[:2]}, {"params": params[2:]}], lr=.9)
    opt.load_state_dict(checkpoint)
    records.append(snapshot())
    for p, grad in zip(params, ([.1, .5, -.2], [.8, -.4], [.3, -.9])):
        p.grad = t.tensor(grad, dtype=t.float32, device=device)
    returned = opt.step(lambda: "closure-result")
    assert returned == "closure-result"
    records.append(snapshot())

    return records


def _device():
    selected = os.environ.get("JITTOR_TEST_DEVICES", "cpu").strip().lower()
    if selected not in ("cpu", "npu", "npu:0"):
        pytest.fail("Adagrad protocol gate requires one of CPU or NPU in a separate process")
    return "npu:0" if selected.startswith("npu") else "cpu"


def test_adagrad_matches_real_torch_multistep_state_and_resume(tmp_path):
    oracle = os.environ.get("REAL_TORCH_PYTHON", "").strip()
    if not oracle:
        reason = "REAL_TORCH_PYTHON is not configured; Adagrad PyTorch comparison unverified"
        if os.environ.get("JITTOR_REQUIRE_REAL_TORCH", "").lower() in ("1", "true", "yes", "on"):
            pytest.fail(reason)
        pytest.skip(reason)
    device = _device()
    source = (
        "import json\nimport torch\n"
        "assert not hasattr(torch, '_torch_compat_install_context')\n"
        "assert hasattr(torch._C, '_c10d_init')\n"
        + ("import torch_npu\nassert torch.npu.is_available()\n" if device.startswith("npu") else "")
        + inspect.getsource(_trajectory)
        + "\nprint(json.dumps({'version': torch.__version__, 'device': %r, " % device
        + "'dtype': 'float32', 'records': _trajectory(torch, %r)}))\n" % device
    )
    result = subprocess.run(
        [oracle, "-c", source],
        env=child_env(without_torch_mode=True, repo_paths=False),
        cwd=str(tmp_path), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=default_timeout())
    assert result.returncode == 0, result.stdout + result.stderr
    expected = json.loads(result.stdout.strip().splitlines()[-1])
    assert expected["device"] == device and expected["dtype"] == "float32"
    import jittor as jt
    if device.startswith("npu"):
        assert jt.compiler.has_acl, "NPU verification requires the real ACL backend"
        assert jt.flags.use_acl and jt.flags.use_cuda, "ACL dispatch must be active"
    before = jt.core.backend_fallback_count()
    actual = _trajectory(torch, device)
    jt.sync_all()
    assert jt.core.backend_fallback_count() == before, "Adagrad used backend fallback"
    assert len(actual) == len(expected["records"])
    errors = {field: [] for field in ("param", "sum", "grad")}
    references = {field: [] for field in errors}
    for index, (actual_step, reference_step) in enumerate(zip(actual, expected["records"])):
        assert len(actual_step) == len(reference_step)
        for actual_param, reference_param in zip(actual_step, reference_step):
            assert actual_param["step"] == reference_param["step"], (index, expected["version"])
            for field in ("param", "sum", "grad"):
                if reference_param[field] is None:
                    assert actual_param[field] is None
                else:
                    reference_array = np.asarray(reference_param[field], dtype=np.float64)
                    actual_array = np.asarray(actual_param[field], dtype=np.float64)
                    references[field].append(reference_array.ravel())
                    errors[field].append(np.abs(actual_array - reference_array).ravel())
                    np.testing.assert_allclose(actual_param[field], reference_param[field],
                                               rtol=2e-6, atol=2e-7,
                                               err_msg="step %d field %s" % (index, field))


    metrics = {"device": device, "oracle_version": expected["version"],
               "rtol": 2e-6, "atol": 2e-7, "fields": {}}
    for field in errors:
        max_abs = float(np.max(np.concatenate(errors[field])))
        scale = max(float(np.max(np.abs(np.concatenate(references[field])))), 1e-12)
        metrics["fields"][field] = {"max_abs": max_abs,
                                    "scaled_relative": max_abs / scale,
                                    "reference_global_scale": scale}
    (tmp_path / "adagrad_oracle_metrics.json").write_text(
        json.dumps(metrics, sort_keys=True), encoding="utf-8")
    print("ADAGRAD_ORACLE_METRICS=" + json.dumps(metrics, sort_keys=True), flush=True)


@pytest.mark.parametrize("flag", ["foreach", "fused", "differentiable"])
def test_adagrad_rejects_unsupported_execution_flags(flag):
    p = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float32, device=_device()))
    with pytest.raises(NotImplementedError):
        torch.optim.Adagrad([p], **{flag: True})


@pytest.mark.parametrize("option", ["lr", "lr_decay", "weight_decay", "initial_accumulator_value", "eps"])
def test_adagrad_rejects_negative_options(option):
    p = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float32, device=_device()))
    with pytest.raises(ValueError):
        torch.optim.Adagrad([p], **{option: -1.})


def test_adagrad_missing_gradient_leaves_initial_state_unchanged():
    p = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float32, device=_device()))
    opt = torch.optim.Adagrad([p], weight_decay=.3, initial_accumulator_value=.4)
    opt.step()
    np.testing.assert_array_equal(p.detach().cpu().numpy(), [1., -2.])
    np.testing.assert_allclose(opt.state[p]["sum"].detach().cpu().numpy(), [.4, .4])
    assert opt.state[p]["step"].item() == 0
    assert p.grad is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_adagrad_rejects_unverified_parameter_dtypes(dtype):
    p = torch.nn.Parameter(torch.tensor([1.], dtype=dtype, device="cpu"))
    with pytest.raises(NotImplementedError, match="float32"):
        torch.optim.Adagrad([p])


def test_adagrad_public_module_alias_and_default_hyperparameters():
    from torch.optim.adagrad import Adagrad
    assert Adagrad is torch.optim.Adagrad
    p = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float32, device=_device()))
    opt = Adagrad([p])
    expected = dict(lr=.01, lr_decay=0, weight_decay=0,
                    initial_accumulator_value=0, eps=1e-10, foreach=None,
                    maximize=False, differentiable=False, fused=None)
    group = opt.state_dict()["param_groups"][0]
    assert opt.defaults == expected
    assert {key: group[key] for key in expected} == expected
    assert set(opt.state[p]) == {"sum", "step"}


def test_adagrad_step_assignment_and_reload_keep_cpu_placement_before_sync():
    p = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float32, device=_device()))
    opt = torch.optim.Adagrad([p])
    counter = opt.state[p]["step"]
    opt.state[p]["step"] = 7
    assert opt.state[p]["step"] is counter
    assert counter.device.type == "cpu"
    # Do not call item()/sync before device assertions: materializing the old
    # unplaced lazy copy hid the placement defect this regression covers.
    state = opt.state_dict()
    q = torch.nn.Parameter(p.detach().clone())
    restored = torch.optim.Adagrad([q])
    restored.load_state_dict(state)
    assert restored.state[q]["step"].device.type == "cpu"
    assert restored.state[q]["step"].item() == 7
