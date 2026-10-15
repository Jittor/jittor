"""Torch serialization composed from module-owned codecs and readers."""
from .portable import load, save
from .safetensors import _install_safetensors_shim as _install_safetensors_shim
from ..fidelity import Fidelity, register_api_bindings
from .security import add_safe_globals, get_safe_globals, clear_safe_globals, safe_globals


def install(ctx):
    g = ctx.target_namespace
    g.save = save
    g.load = load
    g._vj_pickle_load = load
    g._vj_pickle_save = save
    module = ctx.registry.ensure("torch.serialization")
    module.add_safe_globals = add_safe_globals
    module.get_safe_globals = get_safe_globals
    module.clear_safe_globals = clear_safe_globals
    module.safe_globals = safe_globals
    module.load = load
    module.save = save
    register_api_bindings(module, "torch.serialization",
        ("add_safe_globals", "get_safe_globals", "clear_safe_globals", "safe_globals"),
        Fidelity.APPROXIMATE, "Explicit user allowlist for supported restricted pickle readers")
    register_api_bindings(g, "torch", ("save", "load"), Fidelity.APPROXIMATE,
        "Portable tensor pickle and supported Torch zip storages; restricted "
        "unpickling is the default, native-only formats require explicit unsafe opt-in")


__all__ = ["load", "save", "install"]
