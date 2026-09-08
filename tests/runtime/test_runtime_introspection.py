"""Public observations: host service contracts plus two native integration probes."""
import builtins
import ast
from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def services(monkeypatch):
    package = ModuleType("_introspection_services_test")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "python/jittor/_runtime")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    owner = importlib.import_module(package.__name__ + ".introspection")
    state = importlib.import_module(package.__name__ + ".state")
    flags = SimpleNamespace(use_cuda=0, device_id=-1, use_stat_allocator=0,
                            exec_called=11, stat_allocator_total_alloc_call=2,
                            stat_allocator_total_alloc_byte=256,
                            stat_allocator_total_free_call=1,
                            stat_allocator_total_free_byte=128,
                            compile_options={"nested": [1, {"flag": 2}]},
                            cuda_archs=[80], cc_path="/compiler")
    @contextmanager
    def scope(**changes):
        before = {key: getattr(flags, key) for key in changes}
        try:
            for key, value in changes.items():
                setattr(flags, key, value)
            yield
        finally:
            for key, value in before.items():
                setattr(flags, key, value)
    runtime = state.RuntimeState(state.RuntimeContext(flags), scope)
    config = state.StartupConfig(flags)
    counts = {"cpu": 1, "cuda": 2}
    observed = []
    def count(name):
        observed.append(name)
        value = counts[name]
        if isinstance(value, Exception):
            raise value
        return value
    core = SimpleNamespace(known_backends=lambda: ("cpu", "cuda", "acl"),
                           registered_backends=lambda: tuple(counts),
                           backend_device_count=count,
                           number_of_hold_vars=lambda: 3,
                           number_of_lived_vars=lambda: 5,
                           number_of_lived_ops=lambda: 4)
    def library(name, *, load):
        assert load is False, "introspection must never load a library"
        if name != "cutt":
            raise ValueError(name)
        return owner.Capability(name, "library", owner.CapabilityState.UNPROBED,
                                "a loader exists but has not run")
    capability = SimpleNamespace(
        accelerator=lambda name: owner.Capability(name, "accelerator", owner.CapabilityState.DISABLED,
                                                 "hardware present but build disabled", {"build_enabled": False}),
        libraries=lambda: ("cutt",), library=library)
    api = owner.Introspection(capability, config, runtime, core)
    return SimpleNamespace(owner=owner, api=api, runtime=runtime, config=config,
                           flags=flags, core=core, counts=counts, observed=observed)


def test_host_module_import_has_no_bootstrap_dependency(monkeypatch, services):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split(".", 1)[0] not in {"jittor", "jittor_core", "jittor_utils", "numpy"}, name
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    importlib.reload(services.owner)
    assert services.api.counters.exec_calls == 11


def test_host_root_composes_existing_services_and_declares_public_stub():
    root = Path(__file__).resolve().parents[2] / "python/jittor"
    tree = ast.parse((root / "__init__.py").read_text())
    node = next(item for item in tree.body if isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name) and item.target.id == "introspection")
    assert node.annotation.id == "_Introspection"
    assert isinstance(node.value, ast.Call)
    assert node.value.func.id == "_Introspection"
    assert [value.id for value in node.value.args] == ["capability", "config", "runtime", "core"]
    declaration = next(line for line in (root / "__init__.pyi").read_text().splitlines()
                       if line.startswith("introspection:"))
    stub = ast.parse(declaration).body[0]
    assert isinstance(stub, ast.AnnAssign) and stub.annotation.id == "_Introspection"


def test_host_policy_is_live_nested_readonly_and_scoped(services):
    api, flags = services.api, services.flags
    old = api.policy.snapshot()
    with services.runtime.scope(use_cuda=1, device_id=1):
        assert api.policy.runtime.use_cuda == 1
        assert api.policy.runtime["device_id"] == 1
        assert old.runtime["use_cuda"] == 0
    assert api.policy.runtime.use_cuda == 0
    with pytest.raises(TypeError):
        api.policy.runtime.compile_options["nested"][1]["flag"] = 9
    assert flags.compile_options["nested"][1]["flag"] == 2
    assert api.policy.startup.cuda_archs == (80,)
    with pytest.raises(AttributeError):
        api.policy.runtime.exec_called
    with pytest.raises(KeyError):
        api.policy.startup["use_cuda"]


def test_host_observation_namespaces_reject_assignment_and_deletion(services):
    api = services.api
    for obj, name in [(api, "policy"), (api.policy, "runtime"),
                      (api.policy.runtime, "use_cuda"), (api.policy.startup, "cc_path"),
                      (api.capabilities, "backend"), (api.counters, "live_vars")]:
        with pytest.raises(AttributeError):
            setattr(obj, name, None)
        with pytest.raises(AttributeError):
            delattr(obj, name)


def test_host_devices_use_named_backend_not_current_policy(services):
    api = services.api
    assert services.flags.use_cuda == 0
    inventory = api.capabilities.devices("cuda")
    assert inventory.capability.enabled
    assert inventory.count == 2
    assert inventory.devices == (("cuda", 0), ("cuda", 1))
    assert services.observed == ["cuda"]
    assert services.flags.device_id == -1
    assert api.capabilities.devices("cpu").count == 1
    assert not api.capabilities.backend("acl").enabled
    assert api.capabilities.backend("acl").disabled
    with pytest.raises(TypeError):
        bool(inventory)
    with pytest.raises(ValueError):
        api.capabilities.backend("bogus")


@pytest.mark.parametrize("failure", [RuntimeError("driver refused query"), -1])
def test_host_failed_device_query_is_not_empty_inventory(services, failure):
    services.counts["cuda"] = failure
    result = services.api.capabilities.devices("cuda")
    assert result.capability.failed
    assert result.count is None
    assert result.capability.reason
    with pytest.raises(TypeError):
        bool(result.capability)


def test_host_library_observation_preserves_unprobed_and_cannot_load(services):
    capability = services.api.capabilities
    assert capability.libraries() == ("cutt",)
    assert capability.library("cutt").unprobed
    with pytest.raises(TypeError):
        capability.library("cutt", load=True)


def test_host_counters_are_live_detached_and_do_not_execute(services):
    api, flags = services.api, services.flags
    old = api.counters.snapshot()
    assert old.exec_calls == 11
    assert old.allocator == (False, 2, 256, 1, 128)
    assert (old.held_vars, old.live_vars, old.live_ops) == (3, 5, 4)
    flags.exec_called = 12
    flags.stat_allocator_total_alloc_byte = 512
    services.core.number_of_lived_vars = lambda: 9
    assert api.counters.exec_calls == 12
    assert api.counters.allocator.allocated_bytes == 512
    assert api.counters.live_vars == 9
    assert old.exec_calls == 11 and old.live_vars == 5
    assert services.observed == []
    with pytest.raises(AttributeError):
        old.exec_calls = 1


def test_native_public_api_scope_and_backend_queries():
    import jittor as jt
    api = jt.introspection
    assert "introspection" in jt.__all__
    assert api.capabilities.backend("cpu").enabled
    assert api.capabilities.devices("cpu").count == 1
    old = api.policy.runtime.no_grad
    with jt.runtime.scope(no_grad=not old):
        assert api.policy.runtime.no_grad == (not old)
    assert api.policy.runtime.no_grad == old
    assert api.policy.startup.cc_path == jt.config.cc_path


def test_native_counter_observation_does_not_submit_lazy_graph():
    import jittor as jt
    with jt.runtime.scope(lazy_execution=1, auto_flush_ops=0):
        x = jt.array([1., 2.])
        y = x + 1
        before = jt.introspection.counters.snapshot()
        after = jt.introspection.counters.snapshot()
        assert after == before
        assert before.live_vars > 0 and before.live_ops > 0
        y.sync()
        assert jt.introspection.counters.exec_calls > before.exec_calls
        assert list(y.numpy()) == [2., 3.]
