"""Post-core control surface for activating the Torch shim runtime."""

from __future__ import absolute_import

import os

from jittor.compat._aliases import torch_namespace_owned

from .preflight import is_truthy
from ..diagnostics import EXPECTED, swallowed


def _strict_bootstrap(value):
    if value is None:
        return is_truthy(os.environ.get("JITTOR_TORCH_STRICT_BOOTSTRAP"))
    return bool(value)


def _logger(root_module):
    try:
        return getattr(getattr(root_module, "compiler", None), "LOG", None)
    except (AttributeError, TypeError) as exc:
        swallowed("shim/control.py _logger: return getattr(getattr(root_module, 'compiler', None), ...", exc)
        return None


def _warn(root_module, message):
    try:
        logger = _logger(root_module)
        if logger is not None:
            logger.w(message)
    except EXPECTED as exc:
        swallowed("shim/control.py _warn: logger = _logger(root_module)", exc)


def enable_runtime(root_module, preflight_result=None, strict=None):
    """Compatibility wrapper around the single public activation function."""

    strict = _strict_bootstrap(strict)
    state = getattr(root_module, "_torch_shim_runtime_state", None)
    if state and state.get("installed") and not torch_namespace_owned(root_module):
        raise RuntimeError(
            "cannot re-activate the Jittor Torch shim over a changed Torch "
            "module graph"
        )
    prepared = preflight_result or getattr(root_module, "_compat_preflight_result", None)
    try:
        from .runtime import enable

        return enable(
            inference=is_truthy(os.environ.get("JITTOR_TORCH_INFERENCE")),
            verbose=False,
            strict=strict,
            _root_module=root_module,
            _preflight_result=prepared,
        )
    except EXPECTED as error:
        swallowed("shim/control.py enable_runtime: activate(", error)
        from jittor.compat.torch.context import InstallStepError

        if isinstance(error, InstallStepError):
            raise
        if strict:
            raise RuntimeError("torch shim bootstrap failed") from error
        _warn(root_module, "torch_shim bootstrap skipped: %s" % error)
        return None
