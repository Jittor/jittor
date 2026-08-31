"""Make a vLLM source checkout run on this backend through the torch shim.

vLLM expects two things from its environment that a source checkout on this
backend cannot provide: a compiled C extension (``vllm._C`` and friends) and
the ``torch.ops._C`` namespace that extension registers. Both are answered
here -- the modules as empty stand-ins, the operators from Jittor's own public
primitives in :mod:`.custom_ops`. :mod:`.flash_attn` does the same for the
flash-attention wheel the attention backend expects to import.

:mod:`.layers` and the attention patch in :mod:`.flash_attn` are a
different kind of work: vLLM's own layers, pointed at Jittor's fused
primitives instead of kernels this backend does not have.

Nothing runs unless vLLM is actually imported, and then it runs *before* vLLM
does. That timing is the whole difficulty: vLLM reaches for the compiled
extension from inside its own package body, so anything that waits for a module
to finish executing is already too late. A finder is consulted before the body
runs, so this arms itself from the first import of anything under ``vllm``.

This package is staged inside the repository for convenience while both sides
are moving. It talks to Jittor only through public APIs, so moving it out later
is a directory move plus a ``jittor.module_patches`` entry point -- nothing
here has to change.
"""

import importlib.abc
import importlib.machinery
import sys
import types

from jittor.compat.module_patcher import register_module_patch

from . import custom_ops, flash_attn, layers

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

_installed = False


def install():
    """Answer vLLM's compiled-extension surface. Idempotent; safe to re-call."""

    global _installed
    if _installed:
        return False
    import torch

    # vLLM decides what it may use from `torch.__version__`. This shim keeps
    # Jittor's own version there on purpose -- torch *is* jittor here, and the
    # API level it implements is `torch.__torch_version__`. vLLM reads only the
    # former, so say it here, where the decision to run vLLM has already been
    # made, rather than changing what every other user of the shim sees.
    api_version = getattr(torch, "__torch_version__", None)
    if api_version is not None:
        torch.__version__ = api_version

    for name in _EXTENSION_MODULES:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    custom_ops.register(torch)
    flash_attn.install()
    _installed = True
    return True


class _AlreadyPublished(importlib.abc.Loader):
    """Hand back a module :func:`install` has just put in the module table."""

    def create_module(self, spec):
        return sys.modules[spec.name]

    def exec_module(self, module):
        pass


class _ArmOnFirstImport(importlib.abc.MetaPathFinder):
    """Install vLLM compatibility the first time anything under ``vllm`` loads.

    Returning ``None`` hands the import back to the normal finders, which is
    what should happen for vLLM's own modules. The one exception is a name
    :func:`install` has just published -- the machinery has already looked past
    the module table for this import and would not look again.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "vllm" and not fullname.startswith("vllm."):
            return None
        install()
        if fullname in sys.modules:
            return importlib.machinery.ModuleSpec(fullname, _AlreadyPublished())
        return None


def register():
    """Arm vLLM compatibility. Nothing runs until vLLM is actually imported.

    Two mechanisms, because the work happens at two different moments. The
    compiled-extension surface has to exist *before* vLLM's package body runs,
    which a finder can do. The layer patches need vLLM's classes to exist, so
    they run *after* the module defining them executes -- what the module
    patcher is for.

    Idempotent, and reports the state it establishes rather than whether this
    particular call did the work -- callers record it in a status report that
    has to read the same every time it is taken.
    """

    if not any(isinstance(finder, _ArmOnFirstImport) for finder in sys.meta_path):
        sys.meta_path.insert(0, _ArmOnFirstImport())
    for patches in (layers.PATCHES, flash_attn.PATCHES):
        for path, patch in patches.items():
            register_module_patch(path, patch)
    return True
