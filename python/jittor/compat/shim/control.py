"""Post-core control surface for activating the Torch shim runtime."""

from __future__ import absolute_import

import os
import sys

from jittor.compat._aliases import torch_namespace_owned

from .preflight import is_truthy, prepare_import_environment, project_runtime_root
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


def _apply_external_patches(root_module, state):
    from jittor.compat.integrations import apply_external_runtime_patches

    report = apply_external_runtime_patches(logger=_logger(root_module))
    state["external_patches"] = report
    result = state.get("result")
    if isinstance(result, dict):
        result["module_patches"] = report.get("module_patches")
        result["external_backends"] = report.get("external_backends")
        result["integrations"] = report
    return report


def enable_runtime(root_module, preflight_result=None, strict=None):
    strict = _strict_bootstrap(strict)
    state = getattr(root_module, "_torch_shim_runtime_state", None)
    if state is None:
        state = {"installed": False, "result": None, "external_patches": None}
        root_module._torch_shim_runtime_state = state
    if state["installed"]:
        if not torch_namespace_owned(root_module):
            raise RuntimeError(
                "cannot re-enable the Jittor Torch shim over a changed Torch "
                "module graph"
            )
        _apply_external_patches(root_module, state)
        sys.modules["torch"] = root_module
        return state["result"]

    prepared = preflight_result or getattr(
        root_module, "_compat_preflight_result", None
    )
    project = getattr(prepared, "project_root", "") or os.environ.get(
        "JITTOR_TORCH_PROJECT_ROOT"
    )
    if not project:
        entry = sys.argv[0] if sys.argv else ""
        project = (
            os.path.dirname(os.path.abspath(entry))
            if entry and entry not in ("-c", "-m") and os.path.isfile(entry)
            else os.getcwd()
        )
    runtime_root = (
        getattr(prepared, "runtime_root", "")
        or os.environ.get("JITTOR_TORCH_RUNTIME_ROOT")
        or os.fspath(project_runtime_root(project))
    )
    result = None
    try:
        prepare_import_environment(
            project_root=project,
            runtime_root=runtime_root,
            force=True,
            configure_cuda=False,
        )
        from .runtime import enable

        result = enable(
            project_root=project,
            runtime_root=runtime_root,
            auto_scan_extensions=True,
            build_extensions=True,
            local_home=True,
            configure_cuda=False,
            inference=is_truthy(os.environ.get("JITTOR_TORCH_INFERENCE")),
            verbose=False,
            strict=strict,
        )
    except EXPECTED as error:
        swallowed("shim/control.py enable_runtime: prepare_import_environment(", error)
        from jittor.compat.torch.context import InstallStepError

        if isinstance(error, InstallStepError):
            raise
        if strict:
            raise RuntimeError("torch shim bootstrap failed") from error
        _warn(root_module, "torch_shim bootstrap skipped: %s" % error)
        return None
    if not bool(getattr(root_module, "_torch_compat_install_complete", False)):
        error = RuntimeError(
            "torch shim bootstrap did not complete the required compatibility graph"
        )
        if strict:
            raise RuntimeError("torch shim bootstrap failed") from error
        _warn(root_module, "torch_shim bootstrap skipped: %s" % error)
        return None
    if sys.modules.get("torch") is not root_module:
        error = RuntimeError("torch shim bootstrap did not claim the torch namespace")
        if strict:
            raise RuntimeError("torch shim bootstrap failed") from error
        _warn(root_module, "torch_shim bootstrap skipped: %s" % error)
        return None
    state["installed"] = True
    state["result"] = result
    if isinstance(result, dict):
        state["external_patches"] = result.get("integrations")
    return result


class TorchShimFlagsProxy:
    def __init__(self, inner, root_module, strict=None):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_root_module", root_module)
        object.__setattr__(self, "_strict", _strict_bootstrap(strict))
        object.__setattr__(self, "_torch_shim", 0)

    def __getattr__(self, name):
        # Only reached on a miss, and every flag read is a miss, so this is the
        # hot path: a decode step reads jt.flags thousands of times. `self._inner`
        # is in the instance dict, so plain attribute access finds it without
        # recursing back here -- no need for the slower object.__getattribute__.
        if name == "torch_shim":
            return self._torch_shim
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        if name == "torch_shim":
            object.__setattr__(self, "_torch_shim", int(bool(value)))
            if value:
                enable_runtime(
                    object.__getattribute__(self, "_root_module"),
                    preflight_result=getattr(
                        object.__getattribute__(self, "_root_module"),
                        "_compat_preflight_result",
                        None,
                    ),
                    strict=object.__getattribute__(self, "_strict"),
                )
            return
        setattr(object.__getattribute__(self, "_inner"), name, value)

    def __repr__(self):
        return repr(object.__getattribute__(self, "_inner"))


def wrap_flags(root_module, core_flags, strict=None):
    current = getattr(root_module, "flags", None)
    if isinstance(current, TorchShimFlagsProxy):
        object.__setattr__(current, "_strict", _strict_bootstrap(strict))
        return current
    wrapped = TorchShimFlagsProxy(core_flags, root_module, strict=strict)
    root_module.flags = wrapped
    return wrapped
