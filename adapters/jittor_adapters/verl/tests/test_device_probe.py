"""Pinned probe/lifecycle contracts, host fakes; not NPU/CUDA execution proof."""
import hashlib
import importlib
import inspect
from pathlib import Path
import sys
import types

import pytest

ADAPTERS = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ADAPTERS))
sys.path.insert(0, str(ADAPTERS / "tests"))
from jittor_adapters.verl import device_probe as adapter
from jittor_adapters._common import UnsupportedAdapterVersion
from _adapter_boundary import boundary_errors

# Exact upstream function; module-file pin alone is replaced by the fixture pin.
SOURCE = '''def is_torch_npu_available(check_device=True) -> bool:
    """Check if Ascend NPU is available for PyTorch operations.

    Attempts to detect NPU availability by checking for the torch.npu module
    and its is_available() function.

    Args:
        check_device : only check torch_npu package or strictly check if NPU device is available

    Returns:
        bool: True if NPU is available, False otherwise.
    """
    try:
        if not hasattr(torch, "npu"):
            return False

        if check_device:
            return torch.npu.is_available()
        else:
            return True
    except ImportError:
        return False
'''


@pytest.fixture
def probe(tmp_path, monkeypatch):
    filename = tmp_path / "device.py"
    filename.write_text(SOURCE)
    module = types.ModuleType(adapter.MODULE)
    module.__file__ = str(filename)
    module.torch = types.SimpleNamespace(npu=types.SimpleNamespace(is_available=lambda: False))
    exec(compile(SOURCE, str(filename), "exec"), vars(module))
    original = getattr(module, adapter.FUNCTION)
    root = types.ModuleType("verl")
    root.__version__, root.__path__ = "0.9.1", []
    alias = types.ModuleType("verl.workers.device_probe_alias")
    setattr(alias, adapter.FUNCTION, original)
    external = types.ModuleType("unrelated.device_probe_alias")
    setattr(external, adapter.FUNCTION, original)
    compat = types.ModuleType("_verl_device_probe_host_compat")
    compat.__path__ = [str(ADAPTERS.parent / "compat")]
    monkeypatch.setitem(sys.modules, compat.__name__, compat)
    patcher = importlib.import_module(compat.__name__ + ".module_patcher")
    monkeypatch.setitem(sys.modules, "jittor.compat.module_patcher", patcher)
    for value in (root, module, alias, external):
        monkeypatch.setitem(sys.modules, value.__name__, value)
    monkeypatch.setattr(adapter, "SOURCE_SHA256", hashlib.sha256(SOURCE.encode()).hexdigest())
    adapter.register(patcher.register_module_patch)
    yield module, original, alias, external, patcher
    patcher.uninstall_module_patches()
    patcher._REGISTRY.clear()


def test_original_false_positive_and_owned_restore(probe):
    module, original, alias, external, patcher = probe
    assert original(False) is True
    assert original(True) is False
    assert hashlib.sha256(inspect.getsource(original).encode()).hexdigest() == adapter.FUNCTION_SHA256
    patcher.install_module_patches(load_entry_points=False)
    replacement = getattr(module, adapter.FUNCTION)
    assert replacement(False) is False
    assert replacement(True) is False
    assert replacement() is False
    assert inspect.signature(replacement) == inspect.signature(original)
    assert getattr(alias, adapter.FUNCTION) is replacement
    assert getattr(external, adapter.FUNCTION) is original
    assert adapter.patch_device_probe(module) is False
    patcher.uninstall_module_patches()
    assert getattr(module, adapter.FUNCTION) is original
    assert getattr(alias, adapter.FUNCTION) is original


@pytest.mark.parametrize("available", (False, True))
@pytest.mark.parametrize("check", (False, True))
def test_public_availability_is_dynamic(probe, available, check):
    module, _, _, _, patcher = probe
    patcher.install_module_patches(load_entry_points=False)
    calls = []
    module.torch.npu.is_available = lambda: calls.append(available) or available
    assert getattr(module, adapter.FUNCTION)(check) is available
    assert calls == [available]


def test_no_namespace_and_no_new_import(probe):
    module, _, _, _, patcher = probe
    before = set(sys.modules)
    patcher.install_module_patches(load_entry_points=False)
    del module.torch.npu
    assert getattr(module, adapter.FUNCTION)(False) is False
    assert "torch_npu" not in set(sys.modules) - before


def test_backend_errors_are_not_hidden(probe):
    module, _, _, _, patcher = probe
    patcher.install_module_patches(load_entry_points=False)
    def failed():
        raise RuntimeError("owned backend failure")
    module.torch.npu.is_available = failed
    with pytest.raises(RuntimeError, match="owned backend failure"):
        getattr(module, adapter.FUNCTION)(False)


def test_external_replacement_survives_cleanup(probe):
    module, _, _, _, patcher = probe
    patcher.install_module_patches(load_entry_points=False)
    external = lambda check_device=True: "external"
    setattr(module, adapter.FUNCTION, external)
    with pytest.raises(patcher.TransactionConflict, match="lost attribute"):
        patcher.uninstall_module_patches()
    assert getattr(module, adapter.FUNCTION) is external


def test_changed_source_rejected_before_write(probe, monkeypatch):
    module, original, alias, _, _ = probe
    monkeypatch.setattr(adapter, "FUNCTION_SHA256", "0" * 64)
    with pytest.raises(UnsupportedAdapterVersion, match="function source"):
        adapter.patch_device_probe(module)
    assert getattr(module, adapter.FUNCTION) is original
    assert getattr(alias, adapter.FUNCTION) is original


def test_unknown_version_rejected(probe, monkeypatch):
    module, original, _, _, _ = probe
    monkeypatch.setattr(sys.modules["verl"], "__version__", "0.9.2")
    with pytest.raises(UnsupportedAdapterVersion, match="supports"):
        adapter.patch_device_probe(module)
    assert getattr(module, adapter.FUNCTION) is original


def test_relocatable_public_boundary():
    assert boundary_errors(Path(adapter.__file__).read_text()) == []


def test_root_version_published_late(probe, monkeypatch):
    module, original, _, _, _ = probe
    root = sys.modules["verl"]
    monkeypatch.delattr(root, "__version__")
    assert adapter.patch_deferred_device_probe(module) is False
    assert getattr(module, adapter.FUNCTION) is original
    root.__version__ = "0.9.1"
    assert adapter.patch_root_device_probe(root) is True
    assert getattr(module, adapter.FUNCTION)(False) is False


def test_missing_device_module_does_not_force_import(probe, monkeypatch):
    monkeypatch.delitem(sys.modules, adapter.MODULE)
    before = set(sys.modules)
    assert adapter.patch_root_device_probe(sys.modules["verl"]) is False
    assert set(sys.modules) == before
