"""Whether ``jt.vmap`` has installed its process-wide entry-point patches.

``install_batching_patches`` replaces ``jt.Var.matmul``, ``jt.matmul``,
``jt.grad`` and the elementwise entries the first time ``jt.vmap`` runs, and it
deliberately never restores them: a later call that does carry a batched
argument must still dispatch.

Several structure contracts compare those entry points with the objects the
alias-composition layer published -- the native binding ``_var_matmul``, the
core ``grad``. That is a statement about the publication *before* any runtime
patch, and it is checked by the structure-only gate, which imports Jittor and
stops there. A smoke worker that also ran a vmap test cannot make it, and used
to fail for a reason that has nothing to do with the contract.
"""

import importlib


def batching_patches_installed():
    try:
        module = importlib.import_module("jittor.vmap")
    except ImportError:
        return False
    return bool(getattr(module, "_PATCHED", False))


def skip_reason():
    return ("harness-limited: jt.vmap installed its process-wide entry-point "
            "patches before this check ran; the pre-patch publication is "
            "pinned by a structure-only run")
