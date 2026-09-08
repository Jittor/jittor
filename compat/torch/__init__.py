"""Install the independent Torch frontend over Jittor's native runtime.

The implementation is split by ``torch.*`` family under :mod:`installers`.
This module owns only public compatibility re-exports and deterministic install
composition. Historical ``jittor.torch_compat`` imports resolve to this same
package object through the central compatibility alias registry.
"""

from __future__ import absolute_import

import sys
from typing import Tuple

from .._aliases import _torch_namespace as _torch_namespace_snapshot
from .context import InstallContext, InstallReport, InstallStepError, ModuleRegistry
from .contracts import InstallStep
from ..transaction import InstallTransaction, active_transaction, _MISSING
from .functional import (
    _diff,
    _isin,
    _repeat_interleave,
    _torch_norm_impl,
    _torch_where_select,
    _trapz,
)
from .grad import (
    _amp_passthrough_decorator,
    _AutocastContext,
    _clip_grad_norm_device,
    _GradDecoratorCtx,
    _GradScaler,
)
from .lr_scheduler import _install_lr_scheduler
from .nested import (
    _NestedTensor,
    _rebuild_nested_tensor,
    _rebuild_var_from_numpy,
    _TorchSize,
    _torch_make_parameter,
    _torch_prune_leaf_registry,
    _torch_register_leaf,
)
from . import optimizers as optimizers_owner
from .optimizers import _install_optimizers
from .serialization import _install_safetensors_shim
from .types import (
    _device_is_cpu,
    _device_is_cuda,
    _dtype_to_str,
    _make_cpu_resident,
    _make_cuda_resident,
    _make_dtypes,
    _mark_cpu_like,
    _var_has_cpu_residency_hint,
    _var_is_cpu_resident,
    device,
    dtype,
)
from .installers import (
    autograd as autograd_installer,
    compiler,
    core,
    cuda,
    data,
    distributed,
    distributions,
    nn,
    numerical,
    tensor,
    utilities,
)
from . import serialization
from ..diagnostics import EXPECTED, swallowed


_COMPAT_PUBLIC_SYMBOLS = (
    _diff,
    _isin,
    _repeat_interleave,
    _torch_norm_impl,
    _torch_where_select,
    _trapz,
    _amp_passthrough_decorator,
    _AutocastContext,
    _clip_grad_norm_device,
    _GradDecoratorCtx,
    _GradScaler,
    _install_lr_scheduler,
    _NestedTensor,
    _rebuild_nested_tensor,
    _rebuild_var_from_numpy,
    _TorchSize,
    _torch_make_parameter,
    _torch_prune_leaf_registry,
    _torch_register_leaf,
    _install_optimizers,
    _install_safetensors_shim,
    _device_is_cpu,
    _device_is_cuda,
    _dtype_to_str,
    _make_cpu_resident,
    _make_cuda_resident,
    _make_dtypes,
    _mark_cpu_like,
    _var_has_cpu_residency_hint,
    _var_is_cpu_resident,
    device,
    dtype,
)


def _install_optim_and_schedulers(context):
    module = context.jittor_module
    _install_optimizers(module, context.registry)
    _install_lr_scheduler(module, context.registry)


def _install_serialization(context):
    serialization.install(context)


def _install_optional_safetensors(context):
    _install_safetensors_shim(context.registry)


def _install_optional_vllm(context):
    """Arm vLLM compatibility, which fires only if vLLM is imported later."""

    from ..module_patcher import install_module_patches
    return install_module_patches(
        transaction=active_transaction(context),
        expected_entry_points=("jittor_vllm",),
        entry_point_names=("jittor_vllm",),
    )


_REQUIRED_STEPS: Tuple[InstallStep, ...] = (
    ("core", core.install),
    ("tensor.base", tensor.install),
    ("tensor.methods", tensor.install_methods),
    ("nn", nn.install),
    ("optim", _install_optim_and_schedulers),
    ("autograd", autograd_installer.install),
    ("cuda", cuda.install),
    ("distributed", distributed.install),
    ("core.extended", core.install_misc),
    ("serialization", _install_serialization),
    ("utilities", utilities.install),
    ("utilities.runtime-knobs", utilities.install_runtime_knobs),
    ("data", data.install),
    ("distributions", distributions.install),
    ("compiler", compiler.install),
    ("numerical", numerical.install),
    ("numerical.signal", numerical.install_signal),
    ("autograd.module-keys", autograd_installer.install_parity),
    ("nn.module-keys", nn.install_parity),
    ("optim.module-keys", optimizers_owner.install_module_keys),
    ("distributions.module-keys", distributions.install_parity),
    ("compiler.module-keys", compiler.install_parity),
    ("numerical.module-keys", numerical.install_parity),
    ("utilities.module-keys", utilities.install_parity),
)

_OPTIONAL_STEPS: Tuple[InstallStep, ...] = (
    ("optional.torchmetrics", utilities.install_torchmetrics),
    ("optional.transformers", utilities.install_transformers),
    ("optional.tensordict", autograd_installer.install_tensordict),
    ("optional.safetensors", _install_optional_safetensors),
    ("optional.flash-attn", utilities.install_flash),
    ("optional.vllm", _install_optional_vllm),
)

_NAMESPACE_TRANSACTION = "_torch_namespace_transaction"


def _same_namespace(left, right):
    return left.keys() == right.keys() and all(
        left[name] is module for name, module in right.items()
    )


def _restore_namespace(snapshot, expected):
    from ..transaction import TransactionConflict
    missing = object()
    conflicts = []
    for name in snapshot.keys() | expected.keys():
        old, new = snapshot.get(name, missing), expected.get(name, missing)
        if old is new:
            continue
        current = sys.modules.get(name, missing)
        if current is old:
            continue  # An explicit child entry already restored this slot.
        if current is not new:
            conflicts.append(name)
            continue
        if old is missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old
    if conflicts:
        raise TransactionConflict("Torch namespace owner lost: " + ", ".join(conflicts))


def _abandon(transaction, context):
    """Give up the process lock and the ledger handle, however install ended.

    Both have to go on every exit, including the one where ``rollback()`` itself
    raises ``TransactionConflict`` because another actor took over a value. That
    path used to run neither: the class-level RLock stayed acquired, so the next
    install from any other thread blocked forever, and the dead transaction
    stayed in ``context.state`` where the runtime write helpers still found it.
    """
    context.state.pop("_install_transaction", None)
    transaction.release()


def install(torch, strict=True, parent_transaction=None):
    """Install once on the explicit Torch target and return that target."""

    from .namespace import TorchNamespace
    if not isinstance(torch, TorchNamespace):
        raise RuntimeError(
            "Torch installation requires an independent TorchNamespace; "
            "install(jittor) is no longer supported. Use shim.activate() and import torch.")

    from .tensor_state import (
        compatibility_owner, bind_tensor_state, snapshot_tensor_state,
        record_tensor_state_changes,
    )
    torch = compatibility_owner(torch)

    if getattr(torch, "_compat_native_composition_in_progress", False):
        raise RuntimeError(
            "cannot activate Torch compatibility while native Jittor "
            "composition is in progress"
        )

    transaction = InstallTransaction("torch.install")
    if parent_transaction is not None and parent_transaction.state != "open":
        raise RuntimeError("parent install transaction must be open")
    context = InstallContext.for_module(torch, strict=strict)
    if context.complete:
        from .._aliases import torch_namespace_owned

        if not torch_namespace_owned(torch):
            raise RuntimeError(
                "completed Torch compatibility graph was changed after install"
            )
        return torch

    transaction.acquire()
    context.state["_install_transaction"] = transaction

    pending = context.state.pop(_NAMESPACE_TRANSACTION, None)
    if pending is not None:
        current = _torch_namespace_snapshot()
        if not _same_namespace(current, pending["before"]):
            context.state[_NAMESPACE_TRANSACTION] = pending
            _abandon(transaction, context)
            raise RuntimeError(
                "torch namespace changed after a failed compatibility install"
            )
        # Required-step markers are rolled back, so retry reconstructs their
        # modules and types. Republishing the abandoned graph here would make
        # fresh owners collide with stale modules from the failed attempt.
        before = pending["before"]
    else:
        before = _torch_namespace_snapshot()

    root_attrs_before = dict(vars(torch))
    var_type = getattr(torch, "Var", None)
    var_attrs_before = (
        dict(vars(var_type))
        if var_type is not None and hasattr(var_type, "__dict__")
        else None
    )
    namespace_after = [before]
    transaction.record_undo(lambda: _restore_namespace(before, namespace_after[0]))

    tensor_state_before = None
    tensor_state = None
    markers_before = dict(context.markers)
    published_before = dict(context.registry._published)
    try:
        try:
            tensor_state = bind_tensor_state(
                context.native_backend, context.target_namespace, transaction,
                state=context.state.get("_tensor_state"),
            )
            previous_tensor_state = context.state.get("_tensor_state", _MISSING)
            context.state["_tensor_state"] = tensor_state
            transaction.record(context.state, "_tensor_state", previous_tensor_state, tensor_state)
        finally:
            # Binding owns its own attribute ledger. Do not record the same
            # changes again when collecting subsequent installer mutations.
            root_attrs_before = dict(vars(torch))
        tensor_state_before = snapshot_tensor_state(tensor_state)
        for step, installer in _REQUIRED_STEPS:
            context.run_required(step, installer)
        for step, installer in _OPTIONAL_STEPS:
            context.run_optional(step, installer)
        from .native_api import install as install_native_delegates
        install_native_delegates(context)
        from .api_manifest import register_public_apis
        register_public_apis(context)
        context.mark_complete()
    except BaseException as exc:
        swallowed("torch/__init__.py install: for step, installer in _REQUIRED_STEPS:", exc)
        if tensor_state_before is not None:
            record_tensor_state_changes(transaction, tensor_state, tensor_state_before)
        # A reverted step is not complete: retry must rebuild its bindings and
        # registrations instead of skipping work that the ledger just undid.
        transaction.record_mapping_diffs(context.markers, markers_before)
        transaction.record_mapping_diffs(context.registry._published, published_before)
        transaction.record_object_diffs(torch, root_attrs_before)
        if var_attrs_before is not None:
            transaction.record_object_diffs(var_type, var_attrs_before)
        namespace_after[0] = _torch_namespace_snapshot()
        context.state[_NAMESPACE_TRANSACTION] = {
            "before": before,
        }
        setattr(torch, InstallContext.COMPLETE_ATTR, False)
        try:
            transaction.rollback()
        finally:
            _abandon(transaction, context)
        raise
    context.state.pop(_NAMESPACE_TRANSACTION, None)
    try:
        namespace_after[0] = _torch_namespace_snapshot()
        if parent_transaction is not None:
            if tensor_state_before is not None:
                record_tensor_state_changes(transaction, tensor_state, tensor_state_before)
            transaction.record_mapping_diffs(context.markers, markers_before)
            transaction.record_mapping_diffs(context.registry._published, published_before)
            transaction.record_object_diffs(torch, root_attrs_before)
            if var_attrs_before is not None:
                transaction.record_object_diffs(var_type, var_attrs_before)
            parent_transaction.adopt(transaction)
        transaction.commit()
    finally:
        _abandon(transaction, context)
    # A module tree can now contain torch-authored classes, which register
    # parameters by nn.Parameter rather than by assignment. Nothing has to be
    # switched on for that: the marker that tells the two apart is attached by
    # this layer's own ``torch.tensor`` and read off the value. There used to be
    # a `_core_api._torch_registration_semantics = True` here, which made the
    # meaning of `module.x = var` in the kernel depend on whether this import had
    # run (see ``jittor._runtime.core_api._is_plain_tensor``).
    return torch


__all__ = [
    "InstallContext",
    "InstallReport",
    "InstallStepError",
    "ModuleRegistry",
    "device",
    "dtype",
    "install",
]
