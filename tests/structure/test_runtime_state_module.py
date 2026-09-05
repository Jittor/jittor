"""Startup configuration and live runtime policy without native bootstrap."""

import builtins
from collections.abc import Mapping
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

import pytest


RUNTIME_SOURCE = Path(__file__).resolve().parents[2] / "python/jittor/_runtime"


def _load_state():
    package_name = "_isolated_runtime_policy_test"
    package = ModuleType(package_name)
    package.__path__ = [str(RUNTIME_SOURCE)]
    names = [package_name, package_name + ".flag_policy", package_name + ".state"]
    previous = {name: sys.modules.get(name) for name in names}
    try:
        sys.modules[package_name] = package
        for name, filename in zip(names[1:], ("flag_policy.py", "state.py")):
            spec = importlib.util.spec_from_file_location(name, RUNTIME_SOURCE / filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return module
    finally:
        for name in names:
            if previous[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous[name]


def _fake_flags(module):
    names = module.STARTUP_FLAGS | module.RUNTIME_FLAGS | module.READONLY_FLAGS
    values = {name: index for index, name in enumerate(sorted(names))}
    values.update(float32_matmul_precision="highest", cuda_kernel_math="fast",
                  cuda_archs=[80, 89], compile_options={"nested": [1, {"value": 2}]})
    return SimpleNamespace(**values)


def test_state_module_load_does_not_import_native_runtime(monkeypatch):
    original_import = builtins.__import__
    loaded_before = {name for name in sys.modules
                     if name == "jittor" or name.startswith("jittor.")}

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"jittor", "jittor_utils", "numpy"}:
            raise AssertionError("runtime state imported bootstrap dependency: " + name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = _load_state()
    flags = _fake_flags(module)
    context = module.RuntimeContext(flags)
    state = module.RuntimeState(context)
    assert state.context is context
    assert context._flags is flags
    assert state.sync_run == flags.sync_run
    assert {name for name in sys.modules
            if name == "jittor" or name.startswith("jittor.")} == loaded_before


def test_injected_flags_are_live_and_runtime_writes_reach_native_storage():
    module = _load_state()
    flags = _fake_flags(module)
    context = module.RuntimeContext(flags)
    state = module.RuntimeState(context)
    for name in module.RUNTIME_FLAGS | module.READONLY_FLAGS:
        assert getattr(context, name) == getattr(flags, name)
        assert getattr(state, name) == getattr(flags, name)
        setattr(flags, name, 900)
        assert getattr(context, name) == getattr(state, name) == 900
        with pytest.raises(AttributeError):
            setattr(context, name, 901)
        if name in module.READONLY_FLAGS:
            with pytest.raises(AttributeError, match="read-only"):
                setattr(state, name, 901)
            assert getattr(flags, name) == 900
        else:
            setattr(state, name, 901)
            assert getattr(context, name) == getattr(flags, name) == 901
    with pytest.raises(AttributeError):
        state.context = context


def test_runtime_aliases_read_and_write_the_canonical_flag():
    module = _load_state()
    flags = _fake_flags(module)
    context = module.RuntimeContext(flags)
    state = module.RuntimeState(context)
    for alias, canonical in module.FLAG_ALIASES.items():
        assert getattr(state, alias) == getattr(flags, canonical)
        setattr(state, alias, 903)
        assert getattr(context, alias) == getattr(flags, canonical) == 903
        assert alias not in vars(flags)
        with pytest.raises(AttributeError):
            setattr(context, alias, 904)


def test_runtime_snapshot_is_deeply_detached():
    module = _load_state()
    flags = _fake_flags(module)
    context = module.RuntimeContext(flags)
    state = module.RuntimeState(context)
    snapshot = state.snapshot()
    assert set(snapshot) == module.RUNTIME_FLAGS | module.READONLY_FLAGS
    assert snapshot == context.snapshot()
    flags.sync_run = 999
    flags.compile_options["nested"][1]["value"] = 3
    assert snapshot["sync_run"] != 999
    assert snapshot["compile_options"]["nested"][1]["value"] == 2
    snapshot["compile_options"]["nested"].append(4)
    assert context.snapshot()["compile_options"] == {"nested": [1, {"value": 3}]}
    assert context.snapshot()["sync_run"] == 999


def test_startup_config_is_deeply_immutable_and_detached():
    module = _load_state()
    flags = _fake_flags(module)
    flags.cuda_archs = [80, {"nested": [89]}]
    config = module.StartupConfig(flags)
    assert config.cuda_archs[0] == 80
    assert config.cuda_archs[1]["nested"] == (89,)
    with pytest.raises(AttributeError, match="immutable"):
        config.cc_flags = "changed"
    with pytest.raises(AttributeError, match="immutable"):
        config._values = {}
    with pytest.raises(TypeError):
        config.cuda_archs[0] = 90
    with pytest.raises(TypeError):
        config.cuda_archs[1]["nested"] = ()
    flags.cuda_archs[1]["nested"].append(90)
    snapshot = config.snapshot()
    assert snapshot["cuda_archs"] == [80, {"nested": [89]}]
    snapshot["cuda_archs"][1]["nested"].append(91)
    assert config.snapshot()["cuda_archs"] == [80, {"nested": [89]}]
    assert set(config.snapshot()) == module.STARTUP_FLAGS


def test_startup_flags_are_not_runtime_fields_or_writable_through_runtime():
    module = _load_state()
    flags = _fake_flags(module)
    state = module.RuntimeState(module.RuntimeContext(flags))
    for name in module.STARTUP_FLAGS:
        original = getattr(flags, name)
        with pytest.raises(AttributeError):
            getattr(state, name)
        with pytest.raises(AttributeError, match="startup"):
            setattr(state, name, "changed")
        assert getattr(flags, name) == original


def test_runtime_scope_validates_every_key_before_invoking_the_factory():
    module = _load_state()
    flags = _fake_flags(module)
    calls = []

    def scope_factory(**changes):
        calls.append(changes)
        for name, value in changes.items():
            setattr(flags, name, value)

    state = module.RuntimeState(module.RuntimeContext(flags), scope_factory)
    original = flags.sync_run
    invalid = module.STARTUP_FLAGS | module.READONLY_FLAGS | {"unknown_flag"}
    for name in invalid:
        with pytest.raises(AttributeError):
            state.scope(sync_run=900, **{name: 901})
        assert calls == []
        assert flags.sync_run == original


def test_runtime_scope_delegates_and_restores_through_exceptions():
    module = _load_state()
    flags = _fake_flags(module)
    calls = []

    @contextmanager
    def scope_factory(**changes):
        calls.append(changes)
        canonical = {module.FLAG_ALIASES.get(name, name): value
                     for name, value in changes.items()}
        previous = {name: getattr(flags, name) for name in canonical}
        try:
            for name, value in canonical.items():
                setattr(flags, name, value)
            yield
        finally:
            for name, value in previous.items():
                setattr(flags, name, value)

    state = module.RuntimeState(module.RuntimeContext(flags), scope_factory)
    original = state.snapshot()
    with pytest.raises(ValueError, match="scope body"):
        with state.scope(sync_run=900, amp_level=4):
            assert state.sync_run == 900
            assert state.auto_mixed_precision_level == 4
            with state.scope(sync_run=901):
                assert state.sync_run == 901
            assert state.sync_run == 900
            raise ValueError("scope body")
    assert state.snapshot() == original
    assert calls == [{"sync_run": 900, "amp_level": 4}, {"sync_run": 901}]
    with pytest.raises(RuntimeError, match="unavailable"):
        module.RuntimeState(state.context).scope(sync_run=1)


def test_field_directories_cover_the_native_partition_and_aliases():
    module = _load_state()
    flags = _fake_flags(module)
    config = module.StartupConfig(flags)
    context = module.RuntimeContext(flags)
    state = module.RuntimeState(context)
    assert not module.STARTUP_FLAGS & (module.RUNTIME_FLAGS | module.READONLY_FLAGS)
    assert not module.RUNTIME_FLAGS & module.READONLY_FLAGS
    assert module.STARTUP_FLAGS <= set(dir(config))
    fields = module.RUNTIME_FLAGS | module.READONLY_FLAGS | module.FLAG_ALIASES.keys()
    assert fields <= set(dir(context))
    assert fields <= set(dir(state))
    assert not module.STARTUP_FLAGS & set(dir(state))
    source_root = RUNTIME_SOURCE.parent / "src"
    declarations = set()
    pattern = re.compile(r"^DEFINE_(?:RUNTIME_)?FLAG(?:_WITH_SETTER)?\([^,]+,\s*(\w+),", re.M)
    for source in source_root.rglob("*.cc"):
        declarations.update(pattern.findall(source.read_text()))
    assert declarations == module.STARTUP_FLAGS | module.RUNTIME_FLAGS | module.READONLY_FLAGS


def test_device_id_falls_back_to_cpu_without_native_device_field():
    module = _load_state()
    flags = _fake_flags(module)
    del flags.device_id
    context = module.RuntimeContext(flags)
    state = module.RuntimeState(context)
    assert context.device_id == state.device_id == -1
    assert context.snapshot()["device_id"] == -1
    flags.device_id = 3
    assert context.device_id == state.device_id == 3


def test_native_public_runtime_classes_reexport_the_state_owner():
    import jittor as jt
    from jittor._runtime import core_api, state

    assert core_api.RuntimeContext is state.RuntimeContext
    assert core_api.RuntimeState is state.RuntimeState
    assert jt.runtime is core_api.runtime
    assert type(jt.runtime) is state.RuntimeState
    assert type(jt.runtime.context) is state.RuntimeContext
    assert jt.runtime.context._flags is jt.flags
    assert type(jt.config) is state.StartupConfig


def test_native_snapshot_contains_python_values_without_allocating_tensors():
    import jittor as jt

    def check_value(value):
        if isinstance(value, Mapping):
            for item in value.values():
                check_value(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                check_value(item)
        else:
            assert value is None or type(value) in (int, bool, float, str), type(value)

    held_before = jt.liveness_info()["hold_vars"]
    check_value(jt.runtime.snapshot())
    check_value(jt.config.snapshot())
    assert jt.liveness_info()["hold_vars"] == held_before
