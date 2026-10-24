import ast
from functools import wraps
import hashlib
import os
import pathlib
import re
from pathlib import Path
import sys
import threading

import pytest


ROOT = Path(__file__).resolve().parents[3]
GUIDE = ROOT / "docs/development/torch-install-transactions.md"
HELPER = ROOT / "tests/_helpers/child_process.py"
DIST_INSTALLER = ROOT / "compat/torch/installers/distributed.py"
RUNTIME = ROOT / "compat/shim/runtime.py"
INTEGRATIONS = ROOT / "compat/integrations.py"
TORCH_INSTALL = ROOT / "compat/torch/__init__.py"


def test_install_boundary_distinguishes_namespace_and_process_isolation():
    text = GUIDE.read_text(encoding="utf-8")
    for token in (
        "torch*` namespace",
        "jt.flags",
        "os.environ",
        "builtins.__import__",
        "sys.meta_path",
        "module_patcher",
        "child_env()",
        "PYTHONPATH",
        "reversible mutation ledger",
        "hard-failure contract",
        "must",
        "not claim full install rollback",
        "flags.__dict__",
        "JT_NCCL_WORLD_SIZE",
        "JT_NCCL_RANK",
        "JT_NCCL_LOCAL_RANK",
        "JT_NCCL_ROOTINFO_FILE",
        "use_nccl",
        "use_mpi",
        "explicit allowlist",
    ):
        assert token in text


def test_distributed_env_writes_are_explicit_and_child_helper_is_pure():
    helper = HELPER.read_text(encoding="utf-8")
    installer = DIST_INSTALLER.read_text(encoding="utf-8")
    assert "def child_env(" in helper
    assert "env = dict(os.environ) if inherit else {}" in helper
    # The rendezvous variables and use_cuda go through the one shared ledger
    # owner in jittor/compat/transaction.py. This installer used to carry its own
    # copy of the "is a transaction recording?" lookup, and asserting on the
    # spelling of that copy is what made the contract read as satisfied while
    # five such copies disagreed about whether a closed ledger still counts.
    assert "from ...transaction import set_env, set_flag" in installer
    assert 'set_flag(jt.flags, "use_cuda", 1)' in installer
    assert not re.search(r"os\.environ\[[^]]+\] *=[^=]", installer)
    assert not re.search(r"jt\.flags\.[A-Za-z_]+ *=[^=]", installer)
    for name in (
        "JT_NCCL_WORLD_SIZE", "JT_NCCL_RANK", "JT_NCCL_LOCAL_RANK",
        "JT_NCCL_ROOTINFO_FILE", "use_nccl", "use_mpi",
    ):
        assert 'set_env("%s"' % name in installer
    assert "JITTOR_TORCH_SHIM" in helper


def test_activation_passes_outer_transaction_and_clears_inner_install_state():
    runtime = RUNTIME.read_text(encoding="utf-8")
    integrations = INTEGRATIONS.read_text(encoding="utf-8")
    torch_install = TORCH_INSTALL.read_text(encoding="utf-8")
    assert "def apply_external_runtime_patches(logger=None, transaction=None):" in integrations
    assert "apply_external_runtime_patches(\n        logger=" in runtime
    assert "transaction=_transaction" in runtime
    assert "transaction=transaction" in integrations
    assert "context.state.pop(\"_install_transaction\", None)" in torch_install


class _BeforePreflightWrites(RuntimeError):
    """Stop the actual orchestration before filesystem/native runtime changes."""


def _host_activation_boundary():
    """Execute source-owned activation/preflight functions, never import Jittor.

    Only installation/transaction services are stubs. The caller argument,
    env-vs-cwd selection and project-path normalization execute actual source.
    """
    events = {}
    state = {"installed": False, "phase": "inactive"}

    class Transaction:
        _lock = threading.RLock()

        def __init__(self, name):
            self.name = name

        def acquire(self):
            pass

        def release(self):
            pass

        def rollback(self):
            events["rolled_back"] = True

    def before_writes(path, purpose):
        events["resolved_runtime"] = path
        raise _BeforePreflightWrites(purpose)

    def source_functions(path, names, namespace):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        selected = [node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in selected} == set(names)
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        module = ast.fix_missing_locations(ast.Module(body=[future] + selected, type_ignores=[]))
        exec(compile(module, str(path), "exec"), namespace)

    preflight = {"os": os, "pathlib": pathlib, "hashlib": hashlib,
                 "_TRUTHY": {"1", "true", "yes", "on"},
                 "require_independent_frontend": lambda **kwargs: None,
                 "_ensure_dir": before_writes}
    source_functions(ROOT / "compat/shim/preflight.py",
                     ("is_truthy", "project_dir", "project_runtime_root", "prepare_import_environment"),
                     preflight)
    normalize_project = preflight["project_dir"]

    def project_path(path=None):
        result = normalize_project(path)
        events["resolved_project"] = result
        return result

    preflight["project_dir"] = project_path

    def prepare(**kwargs):
        events["preflight_arguments"] = kwargs
        return preflight["prepare_import_environment"](**kwargs)

    runtime = {"os": os, "pathlib": pathlib, "sys": sys, "wraps": wraps,
               "ActivationTransaction": Transaction,
               "_require_independent_frontend": lambda *args: None,
               "torch_namespace_claimable": lambda root: True,
               "_runtime_state": lambda root: state,
               "is_truthy": preflight["is_truthy"],
               "prepare_import_environment": prepare}
    source_functions(RUNTIME, ("_serialized_activation", "_activate_once", "activate"), runtime)
    return runtime["activate"], events


@pytest.mark.parametrize("project_argument,environment,expected", (
    (None, "configured-project", "configured-project"),
    (None, None, "unrelated-cwd"),
    ("explicit-project", "configured-project", "explicit-project"),
    ("explicit-project/entry.py", "configured-project", "explicit-project"),
    ("", "configured-project", "unrelated-cwd"),
))
def test_activation_defers_project_precedence_to_preflight(tmp_path, monkeypatch, project_argument, environment, expected):
    """activate(default None) must not turn cwd into an explicit override."""
    cwd = tmp_path / "unrelated-cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("JITTOR_TORCH_RUNTIME_ROOT", str(tmp_path / "runtime-cache"))
    if environment is None:
        monkeypatch.delenv("JITTOR_TORCH_PROJECT_ROOT", raising=False)
    else:
        monkeypatch.setenv("JITTOR_TORCH_PROJECT_ROOT", str(tmp_path / environment))
    argument = None if project_argument is None else ("" if project_argument == "" else tmp_path / project_argument)
    activate, events = _host_activation_boundary()
    with pytest.raises(_BeforePreflightWrites):
        activate(project_root=argument, _root_module=object(), verbose=False, strict=False,
                 configure_cuda=False)
    assert events["resolved_project"] == (tmp_path / expected).resolve()
    assert events["preflight_arguments"]["project_root"] is argument
    assert events["resolved_runtime"] == (tmp_path / "runtime-cache").resolve()
    assert events["rolled_back"] is True
