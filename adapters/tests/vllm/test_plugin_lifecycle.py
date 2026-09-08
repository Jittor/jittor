"""Exercise real discovery/finders/ledgers with controlled external modules."""
import builtins
import importlib
import os
import subprocess
import sys
import types

import pytest

from jittor.compat import module_patcher as patcher
from jittor.compat import transaction
from jittor_adapters.vllm import bootstrap, register
from jittor_adapters.vllm import backend, custom_ops, flash_attn, layers


class EntryPoint:
    group = "jittor.module_patches"
    value = "jittor_adapters.vllm:register"

    def __init__(self, name, loader):
        self.name, self.loader = name, loader

    def load(self):
        return self.loader()


@pytest.fixture(autouse=True)
def isolated_hooks():
    original_meta = list(sys.meta_path)
    registry = {key: list(value) for key, value in patcher._REGISTRY.items()}
    loaded = set(patcher._ENTRY_POINTS_LOADED)
    original_finder = patcher._FINDER
    original_installed = bootstrap._installed
    existing = {k: v for k, v in sys.modules.items() if k == "vllm" or k.startswith("vllm.")}
    for key in existing:
        del sys.modules[key]
    bootstrap._installed = False
    patcher._REGISTRY.clear()
    patcher._ENTRY_POINTS_LOADED.clear()
    try:
        yield
    finally:
        patcher.release_module_patch_hooks()
        transaction.release_runtime_hooks("vllm.activation")
        transaction.release_runtime_hooks("vllm.flash_attention")
        sys.meta_path[:] = original_meta
        patcher._REGISTRY.clear()
        patcher._REGISTRY.update(registry)
        patcher._ENTRY_POINTS_LOADED.clear()
        patcher._ENTRY_POINTS_LOADED.update(loaded)
        patcher._FINDER = original_finder
        bootstrap._installed = original_installed
        for key in list(sys.modules):
            if key == "vllm" or key.startswith("vllm."):
                del sys.modules[key]
        sys.modules.update(existing)


def test_importing_distribution_does_not_activate_backend():
    result = subprocess.run([sys.executable, "-c", "import sys; import jittor_adapters.vllm as jittor_vllm; assert not {'jittor', 'torch', 'vllm'} & sys.modules.keys()"],
                            env=os.environ.copy(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_named_discovery_skips_other_plugins_and_missing_is_optional(monkeypatch):
    def poison():
        raise AssertionError("unselected adapter was imported")
    monkeypatch.setattr(patcher, "_entry_points", lambda group: [EntryPoint("unrelated", poison)])
    before = list(sys.meta_path)
    report = patcher.install_module_patches(expected_entry_points=("jittor_vllm",), entry_point_names=("jittor_vllm",))
    assert report.ok
    assert any(item.name == "jittor_vllm" and item.status == "unavailable" for item in report.results)
    assert "vllm" not in sys.modules
    assert not any(isinstance(f, bootstrap._ArmOnFirstImport) for f in sys.meta_path if f not in before)


def test_before_import_extensions_after_import_patch_and_rollback(tmp_path, monkeypatch):
    events = []
    torch = types.ModuleType("torch")
    torch.events = events
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(custom_ops, "register", lambda target: events.append("before"))
    monkeypatch.setattr(flash_attn, "install", lambda: None)
    def after(module):
        assert module.body_finished
        events.append("after")
        transaction.set_attr(module, "patched", True)
    monkeypatch.setattr(backend, "PATCHES", {"vllm": after})
    monkeypatch.setattr(layers, "PATCHES", {})
    monkeypatch.setattr(flash_attn, "PATCHES", {})
    folder = tmp_path / "vllm"
    folder.mkdir()
    (folder / "__init__.py").write_text("import torch\nimport vllm._C\ntorch.events.append('body')\nbody_finished = True\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(patcher, "_entry_points", lambda group: [EntryPoint("jittor_vllm", lambda: register)])
    original_import = builtins.__import__
    tx = transaction.InstallTransaction("vllm entrypoint")
    tx.acquire()
    try:
        report = patcher.install_module_patches(transaction=tx, expected_entry_points=("jittor_vllm",))
        assert report.ok
        assert "vllm" not in sys.modules
        module = importlib.import_module("vllm")
        assert events == ["before", "body", "after"]
        assert module.patched
        assert builtins.__import__ is original_import
        tx.rollback()
        assert not hasattr(module, "patched")
        assert "vllm._C" not in sys.modules
        assert bootstrap._installed is False
    finally:
        tx.release()


def test_failed_preimport_install_restores_extensions(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    def fail(target):
        assert "vllm._C" in sys.modules
        raise ValueError("operator setup failed")
    monkeypatch.setattr(custom_ops, "register", fail)
    with pytest.raises(ValueError, match="operator setup failed"):
        bootstrap.install()
    assert not bootstrap._installed
    assert not set(bootstrap._EXTENSION_MODULES) & sys.modules.keys()


def test_entrypoint_callback_contract_returns_none(monkeypatch):
    calls = []
    monkeypatch.setattr(bootstrap, "arm", lambda **kwargs: calls.append(kwargs))
    callback = object()
    tx = transaction.InstallTransaction("registrar contract")
    tx.acquire()
    try:
        assert register(callback) is None
    finally:
        tx.release()
    assert calls[0]["register_callback"] is callback
