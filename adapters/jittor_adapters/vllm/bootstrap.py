"""Make a vLLM source checkout run on this backend through the torch shim.

vLLM expects two things from its environment that a source checkout on this
backend cannot provide: a compiled C extension (``vllm._C`` and friends) and
the ``torch.ops._C`` namespace that extension registers. Both are answered
here -- the modules as empty stand-ins, the operators from Jittor's own public
primitives in :mod:`.custom_ops`. :mod:`.flash_attn` does the same for the
flash-attention wheel the attention backend expects to import.

:mod:`.layers` and the attention patch in :mod:`.flash_attn` are a
different kind of work: vLLM's own layers, pointed at Jittor's fused
primitives instead of kernels this backend does not have. :mod:`.backend`
completes that substitution by declaring the paged cache layout those
primitives read.

Nothing runs unless vLLM is actually imported, and then it runs *before* vLLM
does. That timing is the whole difficulty: vLLM reaches for the compiled
extension from inside its own package body, so anything that waits for a module
to finish executing is already too late. A finder is consulted before the body
runs, so this arms itself from the first import of anything under ``vllm``.

This independently installed adapter uses the public Jittor primitives and
compatibility hook contracts. Its entry point arms the pre-import finder and
registers post-import layer callbacks without importing vLLM.
"""

import importlib.abc
import importlib.machinery
import sys
import types

from jittor.compat.module_patcher import register_module_patch
from jittor.compat.transaction import TransactionConflict, owned_runtime_hook, active_transaction, set_attr


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


@owned_runtime_hook("vllm.activation")
def install():
    """Answer vLLM's compiled-extension surface. Idempotent; safe to re-call."""

    global _installed
    if _installed:
        return False
    import torch
    from . import custom_ops, flash_attn

    # vLLM decides what it may use from `torch.__version__`, and needs the
    # torch API level there rather than Jittor's own version.
    #
    # This used to be `torch.__version__ = api_version`. `torch` IS jittor, so
    # that statement rewrote the framework's version number for every user in
    # the process -- an adapter restricted to public APIs silently changing
    # what `jittor.__version__` says. Ask the compatibility layer to report the
    # API level instead: the decision, and its reversal, belong to the layer
    # that owns both numbers. Adapters must not assign to torch/jittor
    # attributes at all; tests/structure/test_vllm_compat_structure.py enforces
    # that.
    report = getattr(torch, "compat_report_torch_api_version", None)
    if report is not None:
        report(True)

    for name in _EXTENSION_MODULES:
        if name not in sys.modules:
            active_transaction().replace_module(sys.modules, name, types.ModuleType(name))
    custom_ops.register(torch)
    flash_attn.install()
    set_attr(sys.modules[__name__], "_installed", True)
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


def arm(transaction=None, *, register_callback=None):
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

    from . import backend, layers, flash_attn
    registrar = register_module_patch if register_callback is None else register_callback

    if not any(isinstance(finder, _ArmOnFirstImport) for finder in sys.meta_path):
        finder = _ArmOnFirstImport()
        index = 0
        sys.meta_path.insert(index, finder)
        if transaction is not None:
            def restore_finder(f=finder, i=index):
                # Owner-aware, like the permissive and module-patch finders.
                # This used to be ``remove(f) if f in sys.meta_path else None``:
                # when another actor had already dropped or replaced the entry,
                # rollback reported success and left whatever they installed in
                # place, so the one failure mode the ledger exists to surface --
                # a concurrent external replacement -- was the one it hid.
                if i >= len(sys.meta_path) or sys.meta_path[i] is not f:
                    raise TransactionConflict(
                        "vLLM arming finder moved or replaced externally"
                    )
                sys.meta_path.pop(i)
            transaction.record_undo(restore_finder)
    for patches in (backend.PATCHES, layers.PATCHES, flash_attn.PATCHES):
        for path, patch in patches.items():
            registrar(path, patch)
    return True
