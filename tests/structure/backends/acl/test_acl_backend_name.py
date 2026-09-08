"""The old ACL spelling resolves to one canonical kernel registry."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[4]


def test_legacy_acl_registry_operations_share_one_entry(monkeypatch):
    path = ROOT / "python/jittor/_runtime/dispatch.py"
    spec = importlib.util.spec_from_file_location("acl_dispatch_name_probe", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "jittor", SimpleNamespace(
        core=SimpleNamespace(Var=type("Var", (), {}), dispatch_context=lambda inputs: ("acl", 0))))
    def implementation():
        return 7
    module.register_kernel("test", "acl_legacy", implementation)
    module.register_kernel("test", "acl", implementation)
    assert module.try_dispatch("test") == 7
    assert list(module._kernels) == [("test", "acl")]
    assert len(module._kernels[("test", "acl")]) == 1
    with module.override_kernel("test", "acl_legacy", lambda: 9):
        assert module.try_dispatch("test") == 9
    assert module.registered_kernel("test", "acl") is implementation
    module.unregister_kernel("test", "acl", implementation)
    assert module.registered_kernel("test", "acl_legacy") is None


def test_acl_descriptor_has_only_canonical_name():
    source = (ROOT / "backends/acl/src/backend.cc").read_text()
    assert 'ops.name = "acl";' in source
    assert "acl_legacy" not in source
