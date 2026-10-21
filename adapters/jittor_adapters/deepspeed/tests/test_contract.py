"""Controlled fake packages exercise ownership; they are no training evidence."""
import hashlib
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
from relocatable_contract import audit_adapter, check_source
from jittor_adapters._common import UnsupportedAdapterVersion, require_version
from jittor_adapters.deepspeed import bootstrap
from jittor.compat.transaction import (
    InstallTransaction, TransactionConflict, release_runtime_hooks,
)


@pytest.fixture
def fake_package(tmp_path, monkeypatch):
    existing = {name: value for name, value in tuple(sys.modules.items())
                if name == "deepspeed" or name.startswith("deepspeed.")}
    for name in existing:
        monkeypatch.delitem(sys.modules, name)
    root = tmp_path / "deepspeed"
    sources = {
        "__init__.py": "from .runtime.engine import DeepSpeedEngine\nfrom .comm import torch\nfrom .elasticity import utils\n__version__='0.19.7'\n",
        "git_version_info_installed.py": "version='0.19.7+fixture'\n",
        "runtime/__init__.py": "",
        "runtime/utils.py": "def count_used_parameters_in_backward(parameters):\n raise AssertionError('private graph task unavailable')\ndef check_internal_apis_for_count_used_parameters():\n return False\n",
        "runtime/engine.py": "from deepspeed.compile.util import is_deepcompile_supported, compile_private\nclass DeepSpeedEngine:\n def __init__(self, config_class=None):\n  self.config=config_class\n def eager(self):\n  return self.config.sentinel\n",
        "compile/__init__.py": "from .util import pad_tensors\n",
        "comm/__init__.py": "",
        "comm/torch.py": "def build_shm_op():\n raise AssertionError('binary Torch extension unavailable')\n",
        "elasticity/__init__.py": "",
        "elasticity/utils.py": "def is_torch_elastic_compatible():\n return True\n",
        "utils/__init__.py": "",
        "utils/torch.py": "def register_grad_hook(parameter, hook):\n raise AssertionError('private AccumulateGrad unavailable')\n",
    }
    for relative, source in sources.items():
        file = root / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(source)
    hashes = {}
    for name in bootstrap.SOURCE_SHA256:
        path = "compile/__init__.py" if name == "deepspeed.compile" else name.removeprefix("deepspeed.").replace(".", "/") + ".py"
        hashes[name] = hashlib.sha256(sources[path].encode()).hexdigest()
    monkeypatch.setattr(bootstrap, "SOURCE_SHA256", hashes)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield root
    # Tests that deliberately lose ownership repair their own foreign slots.
    release_runtime_hooks(bootstrap._OWNER)
    for name in tuple(sys.modules):
        if name == "deepspeed" or name.startswith("deepspeed."):
            sys.modules.pop(name)


def test_eager_engine_body_is_preserved_and_optional_calls_refuse(fake_package):
    bootstrap.arm()
    library = importlib.import_module("deepspeed")
    config = SimpleNamespace(compile_config=SimpleNamespace(deepcompile=False), elasticity_enabled=False, sentinel=object())
    assert library.DeepSpeedEngine(config).eager() is config.sentinel
    assert library.torch.build_shm_op() is None
    assert library.utils.is_torch_elastic_compatible() is False
    config.compile_config.deepcompile = True
    with pytest.raises(NotImplementedError, match="DeepCompile"):
        library.DeepSpeedEngine(config)
    config.compile_config.deepcompile = False
    config.elasticity_enabled = True
    with pytest.raises(NotImplementedError, match="elastic"):
        library.DeepSpeedEngine(config)
    compile_module = importlib.import_module("deepspeed.compile")
    with pytest.raises(NotImplementedError, match="DeepCompile"):
        compile_module.pad_tensors([])


def test_unknown_version_and_changed_source_fail_before_activation(fake_package):
    path = fake_package / "git_version_info_installed.py"
    path.write_text("version='0.19.8'\n")
    before = list(sys.meta_path)
    with pytest.raises(UnsupportedAdapterVersion):
        bootstrap.arm()
    assert sys.meta_path == before
    path.write_text("version='0.19.7'\n")
    with (fake_package / "runtime/engine.py").open("a") as stream:
        stream.write("changed_contract = True\n")
    with pytest.raises(UnsupportedAdapterVersion, match="source contract"):
        bootstrap.arm()
    assert sys.meta_path == before


def test_parent_rollback_removes_source_transformed_package(fake_package):
    transaction = InstallTransaction("deepspeed-test")
    before = list(sys.meta_path)
    bootstrap.arm(transaction=transaction)
    assert bootstrap.arm(transaction=transaction)
    library = importlib.import_module("deepspeed")
    assert library.__version__ == "0.19.7"
    # Runtime import hooks are separate owned activations from the arming
    # transaction. Release their package state before undoing the finder.
    release_runtime_hooks(bootstrap._OWNER)
    transaction.rollback()
    assert sys.meta_path == before
    assert not any(name == "deepspeed" or name.startswith("deepspeed.") for name in sys.modules)


def test_foreign_finder_replacement_is_preserved_and_reported(fake_package):
    bootstrap.arm()
    finder = next(item for item in sys.meta_path if isinstance(item, bootstrap._Finder))
    index = sys.meta_path.index(finder)
    foreign = object()
    sys.meta_path[index] = foreign
    try:
        with pytest.raises(TransactionConflict, match="finder"):
            release_runtime_hooks(bootstrap._OWNER)
        assert sys.meta_path[index] is foreign
    finally:
        sys.meta_path[index] = finder
        sys.meta_path.remove(finder)


def test_live_package_version_wins_over_preimport_literal(monkeypatch):
    library = ModuleType("deepspeed")
    library.__version__ = "0.19.8"
    monkeypatch.setitem(sys.modules, "deepspeed", library)
    with pytest.raises(UnsupportedAdapterVersion):
        require_version("deepspeed", bootstrap.SUPPORTED_VERSIONS, source_version="0.19.7")


def test_source_changes_after_arming_fail_and_remove_partial_imports(fake_package):
    bootstrap.arm()
    with (fake_package / "runtime/engine.py").open("a") as stream:
        stream.write("contract_changed = True\n")
    with pytest.raises(UnsupportedAdapterVersion, match="source contract"):
        importlib.import_module("deepspeed")
    assert not any(name == "deepspeed" or name.startswith("deepspeed.") for name in sys.modules)


def test_foreign_root_module_is_preserved_and_reported(fake_package):
    bootstrap.arm()
    importlib.import_module("deepspeed")
    foreign = ModuleType("deepspeed")
    sys.modules["deepspeed"] = foreign
    with pytest.raises(TransactionConflict, match="modules replaced"):
        release_runtime_hooks(bootstrap._OWNER)
    assert sys.modules["deepspeed"] is foreign


def test_relocatable_contract_and_its_negative_rules(tmp_path):
    audit_adapter(Path(bootstrap.__file__).resolve().parent)
    for invalid in ("import jittor._core.module", "import jittor as engine\nengine.flag=1",
                    "import torch\ngetattr(torch, '_private')", "from jittor.compat.torch import api",
                    "from jittor import _private", "from jittor.compat.transaction import _runtime_hooks"):
        with pytest.raises(AssertionError):
            check_source(invalid)
    package = tmp_path / "package"
    (package / "sub").mkdir(parents=True)
    (package / "sub" / "bad.py").write_text("import jittor._core.module")
    with pytest.raises(AssertionError):
        audit_adapter(package)
