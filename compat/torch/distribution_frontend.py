"""Torch distribution owners delegating mathematics to native distributions."""

import inspect
from types import ModuleType

from .distribution_adapters import DistributionAdapterState
from .api_delegates import bind_delegates
from .context import get_install_context
from .distribution_api import DISTRIBUTION_FUNCTIONS


def make_distribution_frontend(native, target):
    module = ModuleType("torch.distributions")
    module.__package__ = "torch.distributions"
    module.__path__ = []
    module._native_distribution_module = native
    state = DistributionAdapterState(native, target)
    module._distribution_adapter_state = state
    bind_delegates(get_install_context(target), "distribution_functions", {
        name: getattr(native, name) for name in DISTRIBUTION_FUNCTIONS
        if name in native.__all__
    })

    for name in native.__all__:
        value = getattr(native, name)
        if isinstance(value, type) and issubclass(value, native.Distribution):
            value = state.adapt(value)
        elif isinstance(value, ModuleType):
            child = ModuleType("torch.distributions." + name)
            for attribute in dir(value):
                if not attribute.startswith("__"):
                    setattr(child, attribute, getattr(value, attribute))
            value = child
        elif inspect.isfunction(value):
            value = DISTRIBUTION_FUNCTIONS[name]
        setattr(module, name, value)
    module.__all__ = list(native.__all__)
    return module
