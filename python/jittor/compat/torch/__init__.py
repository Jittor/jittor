"""Install the canonical ``import jittor as torch`` compatibility surface.

The implementation is split by ``torch.*`` family under :mod:`installers`.
This module owns only public compatibility re-exports and deterministic install
composition. Historical ``jittor.torch_compat`` imports resolve to this same
package object through the central compatibility alias registry.
"""

from __future__ import absolute_import

import sys

from .._aliases import _torch_namespace as _torch_namespace_snapshot
from .context import InstallContext, InstallReport, InstallStepError, ModuleRegistry
from ..transaction import InstallTransaction
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
    autograd,
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
    from ..vllm import register

    register()
    # Registering fills the patch table; the finder that consults it has to be
    # live before vLLM is imported. Entry points stay out of it -- scanning
    # them here would drag unrelated adapters into every import of the shim.
    install_module_patches(load_entry_points=False)


_REQUIRED_STEPS = (
    ("core", core.install),
    ("tensor.base", tensor.install),
    ("tensor.methods", tensor.install_methods),
    ("nn", nn.install),
    ("optim", _install_optim_and_schedulers),
    ("autograd", autograd.install),
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
    ("autograd.module-keys", autograd.install_parity),
    ("nn.module-keys", nn.install_parity),
    ("optim.module-keys", optimizers_owner.install_module_keys),
    ("distributions.module-keys", distributions.install_parity),
    ("compiler.module-keys", compiler.install_parity),
    ("numerical.module-keys", numerical.install_parity),
    ("utilities.module-keys", utilities.install_parity),
)

_OPTIONAL_STEPS = (
    ("optional.torchmetrics", utilities.install_torchmetrics),
    ("optional.transformers", utilities.install_transformers),
    ("optional.tensordict", autograd.install_tensordict),
    ("optional.safetensors", _install_optional_safetensors),
    ("optional.flash-attn", utilities.install_flash),
    ("optional.vllm", _install_optional_vllm),
)

_NAMESPACE_TRANSACTION = "_torch_namespace_transaction"


def _same_namespace(left, right):
    return left.keys() == right.keys() and all(
        left[name] is module for name, module in right.items()
    )


def _restore_namespace(snapshot):
    for name in tuple(sys.modules):
        if name == "torch" or name.startswith("torch."):
            sys.modules.pop(name, None)
    sys.modules.update(snapshot)


def install(torch, strict=True):
    """Install the Torch surface once and return the canonical Jittor module."""

    if getattr(torch, "_compat_native_composition_in_progress", False):
        raise RuntimeError(
            "cannot activate Torch compatibility while native Jittor "
            "composition is in progress"
        )

    transaction = InstallTransaction("torch.install")
    transaction.acquire()
    context = InstallContext.for_module(torch, strict=strict)
    if context.complete:
        from .._aliases import torch_namespace_owned

        if not torch_namespace_owned(torch):
            raise RuntimeError(
                "completed Torch compatibility graph was changed after install"
            )
        transaction.release()
        return torch

    pending = context.state.pop(_NAMESPACE_TRANSACTION, None)
    if pending is not None:
        current = _torch_namespace_snapshot()
        if not _same_namespace(current, pending["before"]):
            context.state[_NAMESPACE_TRANSACTION] = pending
            transaction.release()
            raise RuntimeError(
                "torch namespace changed after a failed compatibility install"
            )
        _restore_namespace(pending["staged"])
        before = pending["before"]
    else:
        before = _torch_namespace_snapshot()

    transaction.record_undo(lambda: _restore_namespace(before))

    try:
        for step, installer in _REQUIRED_STEPS:
            context.run_required(step, installer)
        for step, installer in _OPTIONAL_STEPS:
            context.run_optional(step, installer)
        context.mark_complete()
    except EXPECTED as exc:
        swallowed("torch/__init__.py install: for step, installer in _REQUIRED_STEPS:", exc)
        staged = _torch_namespace_snapshot()
        _restore_namespace(before)
        context.state[_NAMESPACE_TRANSACTION] = {
            "before": before,
            "staged": staged,
        }
        setattr(torch, InstallContext.COMPLETE_ATTR, False)
        transaction.rollback()
        transaction.release()
        raise
    context.state.pop(_NAMESPACE_TRANSACTION, None)
    transaction.commit()
    transaction.release()
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
