"""Pinned value-head loader control flow; host fakes, no Torch/Jittor/JIT."""
import ast
import hashlib
import importlib
import inspect
from pathlib import Path
import sys
import types

import pytest


ADAPTERS = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ADAPTERS))
SOURCE = '''def load_valuehead_model(local_path, torch_dtype, model_config, trust_remote_code):
    from transformers import AutoModelForCausalLM, AutoModelForTokenClassification

    try:
        model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=local_path,
            torch_dtype=torch_dtype,
            config=model_config,
            attn_implementation="flash_attention_2",
            trust_remote_code=trust_remote_code,
        )
        return model
    except BaseException as e:
        if not is_trl_available():
            raise RuntimeError(
                f"model({local_path}) is not a value head model, please install trl to make it valid"
            ) from e

    assert is_trl_available()

    try:
        from trl.experimental.ppo import AutoModelForCausalLMWithValueHead  # type: ignore
    except ImportError:
        from trl import AutoModelForCausalLMWithValueHead  # type: ignore

    if type(model_config) in AutoModelForVision2Seq._model_mapping.keys():
        module_class = AutoModelForVision2Seq
    else:
        module_class = AutoModelForCausalLM
    ori_model = module_class.from_pretrained(
        pretrained_model_name_or_path=local_path,
        torch_dtype=torch_dtype,
        config=model_config,
        attn_implementation="flash_attention_2",
        trust_remote_code=trust_remote_code,
    )
    # vlm models
    if hasattr(model_config, "text_config"):
        ori_model.config.hidden_size = model_config.text_config.hidden_size
    model = AutoModelForCausalLMWithValueHead.from_pretrained(ori_model)
    patch_valuehead_model(model)
    return model
'''


@pytest.fixture
def loader(tmp_path, monkeypatch):
    filename = tmp_path / "model.py"
    filename.write_text(SOURCE)
    module = types.ModuleType("verl.utils.model")
    module.__file__ = str(filename)
    module.is_trl_available = lambda: False
    module.AutoModelForVision2Seq = types.SimpleNamespace(_model_mapping={})
    module.patch_valuehead_model = lambda model: None
    exec(compile(SOURCE, str(filename), "exec"), vars(module))
    root = types.ModuleType("verl")
    root.__version__ = "0.9.1"
    monkeypatch.setitem(sys.modules, "verl", root)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    calls = []
    result = types.SimpleNamespace(config=types.SimpleNamespace())

    def load(**kwargs):
        calls.append(kwargs)
        if kwargs["attn_implementation"] == "flash_attention_2":
            raise ImportError("flash_attn is not installed")
        return result

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForTokenClassification = types.SimpleNamespace(from_pretrained=load)
    transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=load)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setenv("VERL_DEVICE", "cuda")
    monkeypatch.setenv("JITTOR_VERL_HF_ASYNC_ROLLOUT", "1")
    return module, calls, result


def test_unpatched_loader_overrides_explicit_eager_and_misreports_trl(loader):
    module, calls, _ = loader
    with pytest.raises(RuntimeError, match="please install trl") as error:
        module.load_valuehead_model("offline-model", "fp32", types.SimpleNamespace(_attn_implementation="eager"), False)
    assert isinstance(error.value.__cause__, ImportError)
    assert calls[0]["attn_implementation"] == "flash_attention_2"


def test_shared_policy_respects_explicit_eager(loader):
    adapter = importlib.import_module("jittor_adapters.verl.valuehead_attention")
    module, calls, result = loader
    original, replacement = adapter.prepare(module)
    assert original is module.load_valuehead_model
    assert replacement("offline-model", "fp32", types.SimpleNamespace(_attn_implementation="eager"), False) is result
    assert calls[0]["attn_implementation"] == "eager"
    assert replacement.__globals__ is vars(module)
    assert replacement.__code__.co_filename == original.__code__.co_filename
    assert hashlib.sha256(SOURCE.encode()).hexdigest() == adapter.FUNCTION_SHA256


@pytest.mark.parametrize("setting", (None, "sdpa", "flash_attention_2", {"": "eager"}))
def test_missing_or_non_eager_config_keeps_upstream_request(loader, setting):
    adapter = importlib.import_module("jittor_adapters.verl.valuehead_attention")
    module, calls, _ = loader
    config = types.SimpleNamespace() if setting is None else types.SimpleNamespace(_attn_implementation=setting)
    _, replacement = adapter.prepare(module)
    with pytest.raises(RuntimeError, match="please install trl") as error:
        replacement("offline-model", "fp32", config, False)
    assert isinstance(error.value.__cause__, ImportError)
    assert calls[0]["attn_implementation"] == "flash_attention_2"


@pytest.mark.parametrize("experimental", (False, True))
@pytest.mark.parametrize("vision", (False, True))
def test_original_trl_fallback_and_both_eager_kwargs_remain(loader, monkeypatch, experimental, vision):
    adapter = importlib.import_module("jittor_adapters.verl.valuehead_attention")
    module, calls, result = loader
    module.is_trl_available = lambda: True
    first_error = ValueError("not a token-classification model")

    def fail(**kwargs):
        calls.append(kwargs)
        raise first_error

    sys.modules["transformers"].AutoModelForTokenClassification.from_pretrained = fail
    config = types.SimpleNamespace(_attn_implementation="eager", text_config=types.SimpleNamespace(hidden_size=7))
    vision_calls = []

    def vision_load(**kwargs):
        calls.append(kwargs)
        vision_calls.append(kwargs)
        return result

    module.AutoModelForVision2Seq = types.SimpleNamespace(
        _model_mapping={type(config): object()} if vision else {}, from_pretrained=vision_load)
    wrapped = types.SimpleNamespace(pretrained_model=result)
    wrapper_calls = []
    patch_calls = []
    wrapper = types.SimpleNamespace(from_pretrained=lambda model: wrapper_calls.append(model) or wrapped)
    trl = types.ModuleType("trl")
    trl.AutoModelForCausalLMWithValueHead = wrapper
    monkeypatch.setitem(sys.modules, "trl", trl)
    if experimental:
        package = types.ModuleType("trl.experimental")
        package.__path__ = []
        ppo = types.ModuleType("trl.experimental.ppo")
        ppo.AutoModelForCausalLMWithValueHead = wrapper
        monkeypatch.setitem(sys.modules, package.__name__, package)
        monkeypatch.setitem(sys.modules, ppo.__name__, ppo)
    else:
        monkeypatch.delitem(sys.modules, "trl.experimental.ppo", raising=False)
        monkeypatch.delitem(sys.modules, "trl.experimental", raising=False)
    module.patch_valuehead_model = lambda model: patch_calls.append(model)
    _, replacement = adapter.prepare(module)
    assert replacement("offline-model", "fp32", config, True) is wrapped
    assert [call["attn_implementation"] for call in calls] == ["eager", "eager"]
    assert bool(vision_calls) is vision
    assert result.config.hidden_size == 7
    assert wrapper_calls == [result] and patch_calls == [wrapped]


def test_original_exception_cause_is_not_hidden_by_new_policy(loader):
    adapter = importlib.import_module("jittor_adapters.verl.valuehead_attention")
    module, _, _ = loader
    error = KeyboardInterrupt("original upstream BaseException path")

    def fail(**kwargs):
        raise error

    sys.modules["transformers"].AutoModelForTokenClassification.from_pretrained = fail
    _, replacement = adapter.prepare(module)
    with pytest.raises(RuntimeError, match="please install trl") as caught:
        replacement("offline-model", "fp32", types.SimpleNamespace(_attn_implementation="eager"), False)
    assert caught.value.__cause__ is error


def test_source_pin_and_exact_keyword_count_fail_closed(loader, monkeypatch):
    from jittor_adapters._common import UnsupportedAdapterVersion
    adapter = importlib.import_module("jittor_adapters.verl.valuehead_attention")
    module, _, _ = loader
    filename = Path(module.__file__)
    filename.write_text(SOURCE.replace('attn_implementation="flash_attention_2"', 'attn_implementation="eager"', 1))
    with pytest.raises(UnsupportedAdapterVersion, match="source SHA256"):
        adapter.prepare(module)
    filename.write_text(SOURCE)
    monkeypatch.setattr(adapter, "EXPECTED_CHANGES", 1)
    with pytest.raises(UnsupportedAdapterVersion, match="keyword count"):
        adapter.prepare(module)


def test_source_missing_and_foreign_owner_fail_closed(loader):
    from jittor_adapters._common import UnsupportedAdapterVersion
    adapter = importlib.import_module("jittor_adapters.verl.valuehead_attention")
    module, _, _ = loader
    Path(module.__file__).unlink()
    with pytest.raises(UnsupportedAdapterVersion, match="source is required"):
        adapter.prepare(module)
    module.load_valuehead_model = lambda *args: None
    with pytest.raises(UnsupportedAdapterVersion, match="ownership"):
        adapter.prepare(module)


def test_unsupported_release_and_module_rejected(loader):
    from jittor_adapters._common import UnsupportedAdapterVersion
    adapter = importlib.import_module("jittor_adapters.verl.valuehead_attention")
    module, _, _ = loader
    sys.modules["verl"].__version__ = "0.9.2"
    with pytest.raises(UnsupportedAdapterVersion, match="supports"):
        adapter.prepare(module)
    sys.modules["verl"].__version__ = "0.9.1"
    module.__name__ = "verl.utils.foreign"
    with pytest.raises(UnsupportedAdapterVersion, match="unexpected"):
        adapter.prepare(module)


@pytest.mark.parametrize("device,flag", ((None, None), (None, "1"), ("cuda", None),
                                         ("cuda", "0"), ("npu", "1"), ("cpu", "1")))
def test_default_and_other_devices_never_register_or_patch(loader, monkeypatch, device, flag):
    from jittor_adapters._common import UnsupportedAdapterVersion
    adapter = importlib.import_module("jittor_adapters.verl.valuehead_attention")
    module, _, _ = loader
    for name, value in (("VERL_DEVICE", device), ("JITTOR_VERL_HF_ASYNC_ROLLOUT", flag)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    registrations = []
    adapter.register(lambda *entry: registrations.append(entry))
    assert not registrations
    assert adapter.patch_valuehead_attention(module) is False
    with pytest.raises(UnsupportedAdapterVersion, match="explicit CUDA"):
        adapter.prepare(module)


@pytest.fixture
def owned_patcher(monkeypatch):
    # Load the real lifecycle using a host-only package name: no Jittor/JIT.
    package = types.ModuleType("_valuehead_host_compat")
    package.__path__ = [str(ADAPTERS.parent / "compat")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    patcher = importlib.import_module(package.__name__ + ".module_patcher")
    transaction = importlib.import_module(package.__name__ + ".transaction")
    monkeypatch.setitem(sys.modules, "jittor", types.ModuleType("jittor"))
    monkeypatch.setitem(sys.modules, "jittor.compat", package)
    monkeypatch.setitem(sys.modules, "jittor.compat.module_patcher", patcher)
    yield patcher, transaction
    patcher.uninstall_module_patches()
    patcher._REGISTRY.clear()


def test_candidate_owned_lifecycle_restores_aliases_and_is_idempotent(loader, owned_patcher, monkeypatch):
    adapter = importlib.import_module("jittor_adapters.verl.valuehead_attention")
    module, _, _ = loader
    patcher, _ = owned_patcher
    original = module.load_valuehead_model
    alias = types.ModuleType("verl.workers.valuehead_alias")
    alias.load_valuehead_model = original
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    adapter.register(patcher.register_module_patch)
    report = patcher.install_module_patches(load_entry_points=False)
    assert report.ok
    replacement = module.load_valuehead_model
    assert replacement is not original and alias.load_valuehead_model is replacement
    assert inspect.signature(replacement) == inspect.signature(original)
    assert replacement.__code__.co_firstlineno == original.__code__.co_firstlineno
    assert adapter.patch_valuehead_attention(module) is False
    patcher.uninstall_module_patches()
    assert module.load_valuehead_model is original
    assert alias.load_valuehead_model is original


def test_candidate_restore_preserves_external_owner(loader, owned_patcher):
    adapter = importlib.import_module("jittor_adapters.verl.valuehead_attention")
    module, _, _ = loader
    patcher, transaction = owned_patcher
    adapter.register(patcher.register_module_patch)
    assert patcher.install_module_patches(load_entry_points=False).ok
    external = lambda *args: "external owner"
    module.load_valuehead_model = external
    with pytest.raises(transaction.TransactionConflict):
        patcher.uninstall_module_patches()
    assert module.load_valuehead_model is external


def test_adapter_uses_only_relocatable_imports_and_does_not_mutate_framework():
    sys.path.insert(0, str(ADAPTERS / "tests"))
    from _adapter_boundary import boundary_errors
    adapter = importlib.import_module("jittor_adapters.verl.valuehead_attention")
    assert not boundary_errors(Path(adapter.__file__).read_text())


def test_ast_changes_only_the_two_exact_attention_keyword_values():
    adapter = importlib.import_module("jittor_adapters.verl.valuehead_attention")
    original = ast.parse(SOURCE)
    changed = ast.parse(SOURCE)
    rewrite = adapter._RespectEagerAttention()
    rewrite.visit(changed)
    assert rewrite.changed == 2
    restored = 0
    for node in ast.walk(changed):
        if isinstance(node, ast.keyword) and node.arg == "attn_implementation":
            assert isinstance(node.value, ast.IfExp)
            assert isinstance(node.value.orelse, ast.Constant)
            assert node.value.orelse.value == "flash_attention_2"
            node.value = ast.Constant(value="flash_attention_2")
            restored += 1
    assert restored == 2
    assert ast.dump(changed) == ast.dump(original)


def test_verl_root_registration_includes_valuehead_only_in_explicit_scope(loader, monkeypatch):
    from jittor_adapters import verl
    from jittor_adapters.verl import valuehead_attention as adapter

    monkeypatch.setenv("JITTOR_VERL_NATIVE_NPU_KERNELS", "0")
    registrations = []
    verl.register(lambda *entry: registrations.append(entry))
    assert (adapter.MODULE, adapter.patch_valuehead_attention) in registrations
    monkeypatch.setenv("VERL_DEVICE", "npu")
    registrations.clear()
    verl.register(lambda *entry: registrations.append(entry))
    assert not any(entry[0] == adapter.MODULE for entry in registrations)
