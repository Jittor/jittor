"""Exercise preflight's policy transition without importing the native core."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _configure():
    path = ROOT / "python/jittor/compat/shim/preflight.py"
    tree = ast.parse(path.read_text())
    names = {
        "is_truthy", "_remove_strict_math_nvcc_flags",
        "_acl_environment", "_prepare_kernel_math", "configure_torch_math_flags",
    }
    def wanted(node):
        if isinstance(node, ast.FunctionDef):
            return node.name in names
        # The pre-2.22 unprefixed variable names the module still reads. Taken
        # from the module rather than restated here, so this test cannot pass
        # against a module that has quietly started reading a third one.
        return (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.startswith("_DEPRECATED_"))

    selected = ast.Module(body=[node for node in tree.body if wanted(node)],
                          type_ignores=[])
    namespace = {"os": os, "_TRUTHY": frozenset(("1", "true", "yes", "on"))}
    exec(compile(selected, str(path), "exec"), namespace)
    return namespace["configure_torch_math_flags"]


@pytest.mark.parametrize("acl,keep,policy", [
    (False, False, "strict"), (False, True, "default"), (True, False, "backend"),
])
def test_math_selection_does_not_mutate_startup_flags(monkeypatch, acl, keep, policy):
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("tikcc_path", raising=False)
    monkeypatch.delenv("JT_BUILD_TIKCC_PATH", raising=False)
    monkeypatch.setenv("JITTOR_TORCH_KEEP_FAST_MATH", "1" if keep else "0")
    monkeypatch.setenv("JT_BUILD_NVCC_FLAGS", " -lineinfo --use_fast_math ")

    class Flags:
        cuda_kernel_math = "default"

        @property
        def nvcc_flags(self):
            return " -lineinfo --use_fast_math "

    flags = Flags()
    root = SimpleNamespace(flags=flags, compiler=SimpleNamespace(has_acl=acl))
    _configure()(root)
    assert flags.cuda_kernel_math == policy
    assert flags.nvcc_flags == " -lineinfo --use_fast_math "


def test_math_policy_failure_is_not_swallowed(monkeypatch):
    monkeypatch.setenv("JITTOR_TORCH_KEEP_FAST_MATH", "0")

    class Flags:
        @property
        def cuda_kernel_math(self):
            return "default"

        @cuda_kernel_math.setter
        def cuda_kernel_math(self, value):
            raise RuntimeError("pending graph failed")

    root = SimpleNamespace(flags=Flags(), compiler=SimpleNamespace(has_acl=False))
    with pytest.raises(RuntimeError, match="pending graph failed"):
        _configure()(root)
