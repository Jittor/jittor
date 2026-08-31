"""Make a vLLM source checkout run on this backend through the torch shim.

vLLM expects two things from its environment that a source checkout on this
backend cannot provide: a compiled C extension (``vllm._C`` and friends) and
the ``torch.ops._C`` namespace that extension registers. Both are answered
here -- the modules as empty stand-ins, the operators from Jittor's own public
primitives in :mod:`.custom_ops`.

Nothing runs unless vLLM is actually imported: the work is registered as a
module patch and fires when ``vllm.envs`` executes, which is early enough that
the operators exist before any layer binds one.

This package is staged inside the repository for convenience while both sides
are moving. It talks to Jittor only through public APIs and to vLLM only
through the module patcher, so moving it out later is a directory move plus a
``jittor.module_patches`` entry point -- nothing here has to change.
"""

import sys
import types

from jittor.compat.module_patcher import register_module_patch

from . import custom_ops

# The compiled bundles vLLM tries to import. Being importable-but-empty is what
# tells vLLM its kernels are present, which is the question that leads it to
# ask which ones this build supports -- see the capability probes.
_EXTENSION_MODULES = (
    "vllm._C",
    "vllm._moe_C",
    "vllm._vllm_fa2_C",
    "vllm._vllm_fa3_C",
    "vllm._C_stable_libtorch",
    "vllm.cumem_allocator",
)

# vLLM's earliest import, so patching it puts the operators in place before any
# model is built. The package itself is a backstop for import orders that reach
# it first.
_TRIGGERS = ("vllm.envs", "vllm")

_installed = False


def install(module=None):
    """Answer vLLM's compiled-extension surface. Idempotent; safe to re-call."""

    global _installed
    if _installed:
        return False
    import torch

    for name in _EXTENSION_MODULES:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    custom_ops.register(torch)
    _installed = True
    return True


def register():
    """Arm :func:`install` to run when vLLM is imported. Returns the triggers."""

    for trigger in _TRIGGERS:
        register_module_patch(trigger, install)
    return _TRIGGERS
