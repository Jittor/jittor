"""The runtime state owner is independently loadable and publicly re-exported."""

import builtins
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


STATE_SOURCE = (
    Path(__file__).resolve().parents[2] / "python/jittor/_runtime/state.py"
)
FLAG_NAMES = (
    "sync_run", "device_id", "use_cuda", "cpu_mem_limit", "device_mem_limit",
    "node_order", "lazy_execution", "auto_flush_ops", "auto_convert_64_to_32",
    "reuse_array", "no_grad", "amp_reg", "float32_matmul_precision",
    "use_tensorcore", "cuda_allow_tf32", "auto_mixed_precision_level",
    "try_use_32bit_index", "no_fuse", "gopt_disable", "enable_tuner",
    "exec_called", "use_threading", "use_parallel_op_compiler",
    "profile_memory_enable", "profiler_warmup", "profiler_enable",
    "profiler_rerun", "profiler_record_peek", "profiler_record_shape",
    "profiler_hide_relay", "check_graph", "missing_grad_error", "disable_lock",
    "rewrite_op", "trace_var_data", "trace_py_var", "trace_depth", "log_silent",
    "log_sync", "log_v", "use_stat_allocator", "use_nfef_allocator",
    "use_temp_allocator", "use_sfrl_allocator", "use_cuda_host_allocator",
)


def _load_state():
    spec = importlib.util.spec_from_file_location("_runtime_state_test", STATE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_flags():
    values = {name: index for index, name in enumerate(FLAG_NAMES)}
    values["float32_matmul_precision"] = "highest"
    return SimpleNamespace(**values)


def test_state_module_load_does_not_import_native_runtime(monkeypatch):
    original_import = builtins.__import__
    loaded_before = {
        name for name in sys.modules
        if name == "jittor" or name.startswith("jittor.")
    }

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"jittor", "jittor_utils", "numpy"}:
            raise AssertionError("runtime state imported bootstrap dependency: " + name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = _load_state()
    flags = _fake_flags()
    context = module.RuntimeContext(flags)
    state = module.RuntimeState(context)
    assert state.context is context
    assert context._flags is flags
    assert state.sync_run == flags.sync_run
    assert {
        name for name in sys.modules
        if name == "jittor" or name.startswith("jittor.")
    } == loaded_before


def test_injected_flags_are_live_through_context_and_state():
    module = _load_state()
    flags = _fake_flags()
    context = module.RuntimeContext(flags)
    state = module.RuntimeState(context)
    for name in FLAG_NAMES:
        assert getattr(context, name) == getattr(flags, name)
        assert getattr(state, name) == getattr(flags, name)
        changed = "high" if name == "float32_matmul_precision" else 900
        setattr(flags, name, changed)
        assert getattr(context, name) == changed
        assert getattr(state, name) == changed
        with pytest.raises(AttributeError):
            setattr(context, name, changed)
        with pytest.raises(AttributeError):
            setattr(state, name, changed)
    with pytest.raises(AttributeError):
        state.context = context


def test_snapshot_is_detached_from_flags_and_other_snapshots():
    module = _load_state()
    flags = _fake_flags()
    context = module.RuntimeContext(flags)
    snapshot = context.snapshot()
    assert snapshot == vars(flags)
    assert set(snapshot) == set(FLAG_NAMES)
    flags.sync_run = 999
    flags.float32_matmul_precision = "high"
    assert snapshot["sync_run"] == 0
    assert snapshot["float32_matmul_precision"] == "highest"
    assert context.snapshot()["sync_run"] == 999
    assert context.snapshot()["float32_matmul_precision"] == "high"
    snapshot["sync_run"] = -5
    assert context.sync_run == 999
    assert context.snapshot()["sync_run"] == 999


def test_device_id_falls_back_to_cpu_without_native_device_field():
    module = _load_state()
    flags = _fake_flags()
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


def test_native_snapshot_contains_python_values_without_allocating_tensors():
    import jittor as jt

    held_before = jt.liveness_info()["hold_vars"]
    snapshot = jt.runtime.context.snapshot()
    for name, value in snapshot.items():
        if name == "float32_matmul_precision":
            assert isinstance(value, str)
        else:
            assert type(value) is int, name
    assert jt.liveness_info()["hold_vars"] == held_before
