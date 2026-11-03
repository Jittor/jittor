"""Native test isolation follows activation state, not import membership."""

import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from _helpers.native_runtime import assert_native_runtime, assert_no_new_fallbacks


@pytest.fixture
def native(monkeypatch):
    original = type("Var", (), {})
    module = ModuleType("jittor")
    module.Var = original
    module.core = SimpleNamespace(Var=original, _get_autograd_policy=lambda: 0)
    module.runtime = SimpleNamespace(service_state=lambda name, factory=None: None)
    owner = ModuleType("jittor._core.var")
    owner.Var = original
    monkeypatch.setitem(sys.modules, "jittor._core.var", owner)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    return module


def test_pure_native_and_definition_imports_are_allowed(native, monkeypatch):
    monkeypatch.setitem(sys.modules, "jittor.compat.torch", ModuleType("definitions"))
    assert assert_native_runtime(native)["native_var"]


def test_pure_independent_torch_is_allowed(native, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    assert assert_native_runtime(native)["torch_imported"]


@pytest.mark.parametrize("phase,installed", [("active", True), ("activating", False),
                                           ("active", False), ("failed", True)])
def test_pure_activation_is_rejected(native, phase, installed):
    state = {"phase": phase, "installed": installed}
    native.runtime.service_state = lambda name, factory=None: state
    with pytest.raises(AssertionError, match="Torch"):
        assert_native_runtime(native)


@pytest.mark.parametrize("kind", ["alias", "independent", "placeholder"])
def test_pure_published_frontend_is_rejected(native, monkeypatch, kind):
    torch = native if kind == "alias" else ModuleType("torch")
    if kind == "independent":
        torch._torch_owner = native
    if kind == "placeholder":
        torch._jittor_torch_shim_placeholder = True
    monkeypatch.setitem(sys.modules, "torch", torch)
    with pytest.raises(AssertionError, match="Torch"):
        assert_native_runtime(native)


@pytest.mark.parametrize("target", ["native", "published", "detached"])
@pytest.mark.parametrize("marker", ["complete", "steps", "context"])
def test_pure_partial_install_is_rejected(native, monkeypatch, target, marker):
    owner = native if target == "native" else ModuleType("torch")
    if target == "published":
        monkeypatch.setitem(sys.modules, "torch", owner)
    if target == "detached":
        native.runtime.service_state = lambda name, factory=None: {
            "phase": "failed", "installed": False, "installation_target": owner}
    if marker == "complete":
        owner._torch_compat_install_complete = True
    elif marker == "steps":
        owner._torch_compat_install_steps = {"tensor": "complete"}
    else:
        owner._torch_compat_install_context = SimpleNamespace(
            complete=False, markers={"tensor": "failed"})
    with pytest.raises(AssertionError, match="Torch install"):
        assert_native_runtime(native)


def test_pure_prepared_empty_context_is_allowed(native):
    native._torch_compat_install_context = SimpleNamespace(complete=False, markers={})
    native._torch_compat_install_steps = {}
    assert_native_runtime(native)


def test_pure_non_native_policy_is_rejected(native):
    native.core._get_autograd_policy = lambda: 3
    with pytest.raises(AssertionError, match="graph policy"):
        assert_native_runtime(native)


@pytest.mark.parametrize("replace_core", [False, True])
def test_pure_replaced_var_is_rejected(native, replace_core):
    native.Var = type("Tensor", (native.Var,), {})
    if replace_core:
        native.core.Var = native.Var
    with pytest.raises(AssertionError, match="Var identity"):
        assert_native_runtime(native)


@pytest.mark.parametrize("before", [0, 7])
def test_pure_protocol_fallback_delta_allows_prior_attempts(before):
    assert assert_no_new_fallbacks(before, before) == {
        "before": before, "after": before, "delta": 0}


@pytest.mark.parametrize("before,after", [(0, 1), (7, 8)])
def test_pure_protocol_rejects_new_fallbacks(before, after):
    with pytest.raises(AssertionError, match="new backend fallback"):
        assert_no_new_fallbacks(before, after)


@pytest.mark.parametrize("before,after", [(2, 1), (-1, -1), (True, True), (0, None)])
def test_pure_protocol_rejects_invalid_counters(before, after):
    with pytest.raises(AssertionError):
        assert_no_new_fallbacks(before, after)


def _child(source, *, env=None):
    from _helpers.child_process import run_python_child

    result = run_python_child(["-c", source], env=env, without_torch_mode=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_child_definition_only_imports_preserve_native_runtime():
    _child("""
import jittor as jt
from _helpers.native_runtime import assert_native_runtime
assert_native_runtime(jt)
import jittor.compat.external_backend
import jittor.compat.integrations
import jittor.compat.module_patcher
import jittor.compat.fsdp2.shard
from jittor.compat.torch.context import InstallContext
InstallContext.for_module(jt)
assert_native_runtime(jt)
""")


def test_child_real_activation_is_rejected():
    _child("""
import jittor as jt
from _helpers.native_runtime import assert_native_runtime
assert_native_runtime(jt)
from jittor.compat.shim import activate
activate(auto_scan_extensions=False, build_extensions=False,
         extension_dirs=[], local_home=False, configure_cuda=False)
import torch
assert torch is not jt and vars(torch).get('_torch_owner') is jt
assert jt.Var is jt.core.Var
try:
    assert_native_runtime(jt)
except AssertionError:
    pass
else:
    raise AssertionError('active independent Torch frontend was accepted')
""")


def test_child_genuine_torch_import_is_allowed(monkeypatch):
    from _helpers import child_process

    executable = os.environ.get("REAL_TORCH_PYTHON", "").strip()
    if not executable:
        pytest.skip("REAL_TORCH_PYTHON is required for independent genuine Torch control")
    monkeypatch.setattr(child_process, "PYTHON", executable)
    _child("""
import torch
assert vars(torch).get('_torch_owner') is None
assert torch._C.__file__.endswith(('.so', '.pyd'))
import jittor as jt
from _helpers.native_runtime import assert_native_runtime
assert assert_native_runtime(jt)['torch_imported']
""", env={"PYTHONPATH": ""})
