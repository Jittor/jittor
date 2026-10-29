"""DDP hook namespace publication participates in installation rollback."""
import types

import pytest

from jittor.compat.torch import ddp_comm_hooks as helper
from jittor.compat.torch.context import ModuleRegistry
from jittor.compat.torch import fidelity
from jittor.compat.transaction import InstallTransaction


_PREFIX = "torch.distributed.algorithms.ddp_comm_hooks"
_SYMBOLS = {
    "default_hooks": ("fp16_compress_hook", "bf16_compress_hook",
                      "fp16_compress_wrapper", "bf16_compress_wrapper"),
    "powerSGD_hook": ("powerSGD_hook", "batched_powerSGD_hook", "PowerSGDState"),
}


def _registry(existing):
    algorithms = types.ModuleType("torch.distributed.algorithms")
    placeholder = object()
    algorithms.ddp_comm_hooks = placeholder
    modules = {algorithms.__name__: algorithms}
    originals = {}
    if existing:
        package = types.ModuleType(_PREFIX)
        algorithms.ddp_comm_hooks = package
        modules[_PREFIX] = package
        for suffix, names in _SYMBOLS.items():
            module = types.ModuleType(_PREFIX + "." + suffix)
            setattr(package, suffix, module)
            modules[module.__name__] = module
            for name in names:
                value = object()
                setattr(module, name, value)
                originals[(module, name)] = value
    registry = ModuleRegistry(types.ModuleType("isolated-owner"), modules=modules,
                              native_backend=object())
    return registry, algorithms, modules, originals


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("phase", ["after_first_symbol", "after_default_bindings", "after_complete"])
def test_fault_restores_publications_parent_attributes_and_symbols(monkeypatch, existing, phase):
    registry, algorithms, modules, originals = _registry(existing)
    before_modules = dict(modules)
    before_published = dict(registry._published)
    before_parent = algorithms.ddp_comm_hooks
    before_fidelity = dict(fidelity._REGISTRY)
    package = modules.get(_PREFIX)
    assert package is None or not hasattr(package, "__path__")
    transaction = InstallTransaction("ddp-hook-publication-fault")
    transaction.acquire()
    try:
        set_attribute = helper.set_attr
        register = helper.register_api_bindings

        def inject_attr(target, name, value):
            set_attribute(target, name, value)
            if phase == "after_first_symbol" and name == "fp16_compress_hook":
                raise RuntimeError("injected hook failure")

        def inject_register(module, name, symbols, level, detail):
            register(module, name, symbols, level, detail)
            if phase == "after_default_bindings" and name.endswith(".default_hooks"):
                raise RuntimeError("injected hook failure")

        monkeypatch.setattr(helper, "set_attr", inject_attr)
        monkeypatch.setattr(helper, "register_api_bindings", inject_register)
        with pytest.raises(RuntimeError, match="injected hook failure"):
            helper.install_ddp_comm_hooks(registry.module_map, algorithms)
            raise RuntimeError("injected hook failure")
        transaction.rollback()
    finally:
        transaction.release()
    assert modules == before_modules
    assert registry._published == before_published
    assert algorithms.ddp_comm_hooks is before_parent
    assert fidelity._REGISTRY == before_fidelity
    if package is not None:
        assert not hasattr(package, "__path__")
    for (module, name), value in originals.items():
        assert getattr(module, name) is value


def test_no_transaction_install_is_repeatable_with_stable_module_identities():
    registry, algorithms, modules, _ = _registry(False)
    helper.install_ddp_comm_hooks(registry.module_map, algorithms)
    first = {key: modules[key] for key in (_PREFIX, _PREFIX + ".default_hooks",
                                          _PREFIX + ".powerSGD_hook")}
    helper.install_ddp_comm_hooks(registry.module_map, algorithms)
    assert all(modules[key] is value for key, value in first.items())
    assert algorithms.ddp_comm_hooks is first[_PREFIX]
    assert first[_PREFIX].default_hooks is first[_PREFIX + ".default_hooks"]
    assert first[_PREFIX].powerSGD_hook is first[_PREFIX + ".powerSGD_hook"]
    assert first[_PREFIX + ".default_hooks"].fp16_compress_hook is helper.fp16_compress_hook


def test_two_installs_in_same_transaction_rollback_to_original():
    registry, algorithms, modules, _ = _registry(False)
    before_modules, before_parent = dict(modules), algorithms.ddp_comm_hooks
    before_published = dict(registry._published)
    transaction = InstallTransaction("ddp-hook-repeat-rollback")
    transaction.acquire()
    try:
        helper.install_ddp_comm_hooks(registry.module_map, algorithms)
        first = modules[_PREFIX]
        helper.install_ddp_comm_hooks(registry.module_map, algorithms)
        assert modules[_PREFIX] is first
        transaction.rollback()
    finally:
        transaction.release()
    assert modules == before_modules
    assert registry._published == before_published
    assert algorithms.ddp_comm_hooks is before_parent
