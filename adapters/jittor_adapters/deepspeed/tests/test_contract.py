"""Controlled fake packages exercise ownership; they are no training evidence."""
from contextlib import nullcontext
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
    InstallTransaction, TransactionConflict, release_runtime_hooks, runtime_hook,
)


@pytest.fixture
def fake_package(tmp_path, monkeypatch):
    existing = {name: value for name, value in tuple(sys.modules.items())
                if name == "deepspeed" or name.startswith("deepspeed.")}
    for name in existing:
        monkeypatch.delitem(sys.modules, name)
    root = tmp_path / "deepspeed"
    sources = {
        "__init__.py": "from .runtime.engine import DeepSpeedEngine\nfrom .runtime import bf16_optimizer\nfrom .comm import torch\nfrom .elasticity import utils\n__version__='0.19.7'\n",
        "git_version_info_installed.py": "version='0.19.7+fixture'\n",
        "runtime/__init__.py": "",
        "runtime/utils.py": "import torch\ndef count_used_parameters_in_backward(parameters):\n raise AssertionError('private graph task unavailable')\ndef check_internal_apis_for_count_used_parameters():\n return False\ndef get_global_norm_of_tensors(input_tensors, **kwargs):\n return (input_tensors, kwargs)\ndef mask_nan_or_inf_with_val_inplace(input, **kwargs):\n return None\ndef has_inf_or_nan(input):\n return False\n",
        "runtime/bf16_optimizer.py": "from .utils import get_global_norm_of_tensors\n",
        "runtime/zero/__init__.py": "",
        "runtime/zero/stage_1_and_2.py": "import torch\nclass DeepSpeedZeroOptimizer:\n def independent_gradient_partition_epilogue(self): return None\n def allreduce_bucket(self, *args, **kwargs): return None\n def allreduce_and_copy(self, *args, **kwargs): return None\n def get_grad_norm_direct(self, *args, **kwargs): return None\n def has_overflow_partitioned_grads_serial(self): return None\n def has_overflow(self, partition_gradients=True): return False\n def _update_model_bit16_weights(self, group_index): return None\n",
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


def test_norm_alias_import_and_fallback_are_owned(fake_package):
    bootstrap.arm()
    library = importlib.import_module("deepspeed")
    utils = importlib.import_module("deepspeed.runtime.utils")
    alias = library.bf16_optimizer.get_global_norm_of_tensors
    assert alias is utils.get_global_norm_of_tensors
    assert alias is not alias.__wrapped__
    assert alias([], norm_type=1) == ([], {
        "norm_type": 1, "mpu": None, "use_graph": False, "moe_ep_group": None})
    original = alias.__wrapped__
    release_runtime_hooks(bootstrap._OWNER)
    assert utils.get_global_norm_of_tensors is original
    assert "deepspeed.runtime.bf16_optimizer" not in sys.modules


def test_norm_source_change_refuses_activation(fake_package):
    path = fake_package / "runtime" / "utils.py"
    with path.open("a") as stream:
        stream.write("norm_source_changed = True\n")
    before = list(sys.meta_path)
    with pytest.raises(UnsupportedAdapterVersion, match="source contract"):
        bootstrap.arm()
    assert sys.meta_path == before


def _fake_norm_modules(monkeypatch):
    utils = ModuleType("deepspeed.runtime.utils")
    alias = ModuleType("deepspeed.runtime.early_alias")
    utils.torch = SimpleNamespace()
    def original(*args, **kwargs):
        return args, kwargs
    utils.get_global_norm_of_tensors = original
    alias.get_global_norm_of_tensors = original
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    return utils, alias, original


def test_norm_prebound_alias_rollback(monkeypatch):
    utils, alias, original = _fake_norm_modules(monkeypatch)
    transaction = InstallTransaction("deepspeed-norm-test")
    with runtime_hook(bootstrap._OWNER, parent_transaction=transaction):
        bootstrap._install_global_norm(utils)
    assert alias.get_global_norm_of_tensors is utils.get_global_norm_of_tensors
    assert alias.get_global_norm_of_tensors is not original
    transaction.rollback()
    assert utils.get_global_norm_of_tensors is original
    assert alias.get_global_norm_of_tensors is original
    release_runtime_hooks(bootstrap._OWNER)


def test_norm_foreign_alias_is_preserved_on_release(monkeypatch):
    utils, alias, original = _fake_norm_modules(monkeypatch)
    with runtime_hook(bootstrap._OWNER):
        bootstrap._install_global_norm(utils)
    foreign = object()
    alias.get_global_norm_of_tensors = foreign
    with pytest.raises(TransactionConflict):
        release_runtime_hooks(bootstrap._OWNER)
    assert alias.get_global_norm_of_tensors is foreign
    assert utils.get_global_norm_of_tensors is original


@pytest.mark.parametrize("options", [
    {"norm_type": 1}, {"norm_type": float("inf")}, {"use_graph": True},
    {"mpu": object()}, {"moe_ep_group": object()},
])
def test_norm_fallback_forwards_original_arguments(options):
    from jittor_adapters.deepspeed._norm import make_global_norm
    calls = []
    sentinel = object()
    tensors = [object()]
    def original(values, **kwargs):
        calls.append((values, kwargs))
        return sentinel
    wrapper = make_global_norm(
        SimpleNamespace(torch=SimpleNamespace()), original, cuda_allowed=True)
    assert wrapper(tensors, **options) is sentinel
    assert calls[0][0] is tensors
    for name, value in options.items():
        assert calls[0][1][name] is value


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
def test_norm_input_and_old_result_survive_input_write(dtype_name):
    import torch
    from jittor_adapters.deepspeed._norm import make_global_norm
    dtype = getattr(torch, dtype_name)
    values = [torch.tensor([3., 4.], dtype=dtype, requires_grad=True),
              torch.tensor([12.], dtype=dtype, requires_grad=True)]
    def original(*args, **kwargs):
        raise AssertionError("eligible norm unexpectedly fell back")
    def mask(value, device=None):
        value.masked_fill_(value.isnan().logical_or(value.isinf()), -1.)
    module = SimpleNamespace(torch=torch, mask_nan_or_inf_with_val_inplace=mask)
    scope = nullcontext()
    if hasattr(torch, "_torch_compat_install_context"):
        import jittor as jt
        scope = jt.flag_scope(auto_flush_ops=0)
    with scope:
        norm = make_global_norm(module, original, cuda_allowed=True)(values)
        assert norm.dtype == torch.float32
        assert not norm.requires_grad
        # Do not first materialize norm: writes must preserve its pending readers.
        with torch.no_grad():
            values[0].detach().zero_()
            values[1].detach().copy_(torch.tensor([1.], dtype=dtype))
        assert float(norm.item()) == pytest.approx(13., rel=3e-5)
        assert all(value.grad is None for value in values)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("values, expected", [
    ([[float("nan")], [3.]], -1.),
    ([[float("inf")], [3.]], -1.),
    ([[float("-inf")], [3.]], -1.),
    ([[0., 0.], [0.]], 0.),
    ([[], []], 0.),
    ([[], [3., 4.]], 5.),
])
def test_norm_nonfinite_zero_and_empty_inputs(dtype_name, values, expected):
    import torch
    from jittor_adapters.deepspeed._norm import make_global_norm
    tensors = [torch.tensor(value, dtype=getattr(torch, dtype_name))
               for value in values]
    def original(*args, **kwargs):
        raise AssertionError("eligible norm unexpectedly fell back")
    def mask(value, device=None):
        value.masked_fill_(value.isnan().logical_or(value.isinf()), -1.)
    module = SimpleNamespace(torch=torch, mask_nan_or_inf_with_val_inplace=mask)
    result = make_global_norm(module, original, cuda_allowed=True)(tensors)
    assert result.dtype == torch.float32
    assert not result.requires_grad
    assert result.item() == pytest.approx(expected)


@pytest.mark.parametrize("backend, enabled, expected", [
    ("cuda", True, True),
    ("rocm", True, False),
    ("acl", True, False),
    ("corex", True, False),
    (None, True, False),
    ("cuda", False, False),
])
def test_cuda_norm_requires_verified_build_backend(backend, enabled, expected):
    cuda = SimpleNamespace(enabled=enabled, evidence={"build_backend": backend})
    capability = SimpleNamespace(accelerator=lambda name: cuda)
    assert bootstrap._cuda_norm_allowed(capability) is expected


@pytest.mark.parametrize("device_type, cuda_allowed", [
    ("npu", True), ("mps", True), ("xpu", True), ("privateuseone", True),
    ("cuda", False),
])
def test_norm_unverified_device_preserves_upstream(device_type, cuda_allowed):
    from jittor_adapters.deepspeed._norm import make_global_norm
    tensor = SimpleNamespace(device=SimpleNamespace(type=device_type))
    torch_api = SimpleNamespace(is_tensor=lambda value: value is tensor)
    calls = []
    sentinel = object()
    def original(values, **kwargs):
        calls.append((values, kwargs))
        return sentinel
    wrapper = make_global_norm(
        SimpleNamespace(torch=torch_api), original, cuda_allowed=cuda_allowed)
    tensors = [tensor]
    assert wrapper(tensors) is sentinel
    assert calls == [(tensors, {
        "norm_type": 2, "mpu": None, "use_graph": False, "moe_ep_group": None})]
