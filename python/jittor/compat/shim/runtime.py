"""Orchestration for enabling the canonical Jittor Torch shim."""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any, List, NamedTuple, Optional, Sequence, Union

from .build import (
    _deploy_torch_shim, _extension_from_user_item, _preload_jittor_cores,
    _pythonpath_for_child, _write_build_sitecustomize, build_extension_dirs,
)
from .discovery import (
    NativeExtension, _dedupe_extensions, _log, _pythonpath_extension_roots,
    scan_extension_dirs,
)
from .preflight import (
    _ensure_dir, append_sys_path, configure_torch_math_flags, is_truthy,
    jittor_python_root, prepare_import_environment, prepend_sys_path,
)
from jittor.compat._aliases import torch_namespace_claimable, torch_namespace_owned
from jittor.compat.torch.namespace import independent_torch_namespace
from ..diagnostics import EXPECTED, swallowed
from ..transaction import ActivationTransaction


class ActivationStatus(NamedTuple):
    phase: str
    active: bool
    result: Optional[dict]
    error: Optional[str]


def _runtime_state(root_module):
    state = getattr(root_module, "_torch_shim_runtime_state", None)
    if state is None:
        state = {
            "phase": "inactive",
            "installed": False,
            "result": None,
            "external_patches": None,
            "error": None,
            "runtime_configured": False,
        }
        root_module._torch_shim_runtime_state = state
    return state


def activation_status(root_module=None):
    """Return an immutable snapshot of process-wide Torch shim activation."""

    root = root_module or sys.modules.get("jittor")
    state = getattr(root, "_torch_shim_runtime_state", None) if root else None
    if state is None:
        return ActivationStatus("inactive", False, None, None)
    phase = state.get("phase") or (
        "active" if state.get("installed") else "inactive"
    )
    return ActivationStatus(
        phase,
        phase == "active" and bool(state.get("installed")),
        state.get("result"),
        state.get("error"),
    )


def _activate_once(
    project_root: Optional[Union[str, os.PathLike]] = None,
    runtime_root: Optional[Union[str, os.PathLike]] = None,
    import_paths: Optional[Sequence[Union[str, os.PathLike]]] = None,
    extension_dirs: Optional[Sequence[Union[NativeExtension, str, os.PathLike]]] = None,
    auto_scan_extensions: bool = True,
    build_extensions: bool = True,
    max_scan_depth: int = 5,
    local_home: bool = True,
    configure_cuda: bool = True,
    inference: bool = False,
    verbose: Optional[bool] = None,
    strict: Optional[bool] = None,
    _root_module=None,
    _preflight_result=None,
    _composition=False,
    _transaction=None,
    independent_namespace=False,
):
    """Enable Jittor-backed ``import torch`` for the current Python process.

    It sets project-local cache directories, deploys the torch shim into that
    runtime, registers Jittor as ``torch`` in-process, scans local native
    extension projects, and builds setuptools extensions through Jittor's
    ``torch.utils.cpp_extension`` facade.

    Typical use in a torch-oriented project entrypoint::

        from jittor.compat.shim import activate
        activate(project_root=__file__)
        import torch

    For pure evaluation/metrics scripts, pass ``inference=True`` to enable
    Jittor no-grad mode for the process.

    Dependency rule for Jittor-backed runs: do not install upstream PyTorch
    environment files as-is. Install Jittor plus the project's non-torch Python
    dependencies, install torch-dependent pure-Python packages with
    ``pip install --no-deps`` when needed, and keep local C++/CUDA extension
    source trees in the project so this bootstrap can build them through the
    shim. Prebuilt wheels compiled against PyTorch/libtorch ABI are not
    compatible with the Jittor backend unless a matching shim implementation
    exists.
    """

    jittor_root = _root_module or sys.modules.get("jittor")
    if not torch_namespace_claimable(jittor_root):
        raise RuntimeError(
            "cannot enable the Jittor Torch shim over a preloaded Torch "
            "module graph"
        )

    verbose = (not is_truthy(os.environ.get("JITTOR_TORCH_QUIET"))) if verbose is None else bool(verbose)
    strict_bootstrap = (
        is_truthy(os.environ.get("JITTOR_TORCH_STRICT_BOOTSTRAP"))
        if strict is None
        else bool(strict)
    )
    jittor_root.autograd.set_policy(
        jittor_root.autograd.EXPLICIT_REQUIRES_GRAD
    )
    if _composition:
        jt = jittor_root
        configure_torch_math_flags(jt)
        from jittor.compat import torch as torch_compat
        transaction = ActivationTransaction("shim.composition")
        transaction.acquire()
        try:
            torch_compat.install(jt, strict=strict_bootstrap)
            published = independent_torch_namespace(jt) if independent_namespace else jt
            if independent_namespace:
                jt._torch_compat_install_context.registry._published["torch"] = published
            transaction.publish_module(sys.modules, "torch", published)
            transaction.commit()
        except EXPECTED:
            transaction.rollback()
            raise
        finally:
            transaction.release()
        return {
            "torch": published,
            "runtime_root": getattr(_preflight_result, "runtime_root", ""),
            "shim_site": "",
            "extensions": [],
            "built": [],
            "preloaded": [],
            "module_patches": None,
            "external_backends": None,
            "integrations": {},
        }
    prepared = _preflight_result
    if not bool(getattr(prepared, "active", False)):
        prepared = prepare_import_environment(
            project_root=project_root or pathlib.Path.cwd(),
            runtime_root=runtime_root,
            force=True,
            local_home=local_home,
            configure_cuda=configure_cuda,
        )
    project_dir = pathlib.Path(prepared.project_root)
    runtime = pathlib.Path(prepared.runtime_root)

    shim_site = _ensure_dir(runtime / "site-packages",
                            "the deployed torch shim this process imports")
    jt_python = jittor_python_root()
    _deploy_torch_shim(shim_site)
    _write_build_sitecustomize(shim_site)

    # Only the two directories this layer owns go in front of the standard
    # library: the deployed shim (which is what makes `import torch` resolve
    # here) and Jittor's own package root. The project directory used to be
    # inserted at sys.path[0] as well, ahead of both -- so a project holding a
    # `types.py` or a `copy.py` shadowed the stdlib for the whole process,
    # Jittor's own imports included, from the moment enable() ran. A project
    # only needs to be importable; it goes on the end.
    if _transaction is None:
        prepend_sys_path(shim_site); prepend_sys_path(jt_python)
        append_sys_path(project_dir)
    else:
        _transaction.mutate_path(sys.path, os.fspath(shim_site), prepend=True)
        _transaction.mutate_path(sys.path, os.fspath(jt_python), prepend=True)
        _transaction.mutate_path(sys.path, os.fspath(project_dir), prepend=False)
    for p in import_paths or ():
        pp = pathlib.Path(p)
        if not pp.is_absolute():
            pp = project_dir / pp
        if _transaction is None:
            append_sys_path(pp.resolve())
        else:
            _transaction.mutate_path(sys.path, os.fspath(pp.resolve()), prepend=False)

    import jittor as jt
    if jittor_root is not None and jt is not jittor_root:
        raise RuntimeError("Torch shim activation changed the Jittor root module")
    configure_torch_math_flags(jt)
    if inference:
        if _transaction is None:
            jt.flags.no_grad = 1
        else:
            _transaction.mutate_flag(jt.flags, "no_grad", 1)
    from jittor.compat import torch as torch_compat
    torch_compat.install(jt, strict=strict_bootstrap)
    published = independent_torch_namespace(jt) if independent_namespace else jt
    if independent_namespace:
        jt._torch_compat_install_context.registry._published["torch"] = published
    if _transaction is None:
        sys.modules["torch"] = published
    else:
        _transaction.publish_module(sys.modules, "torch", published)
    try:
        from jittor.compat.shim.cpp_extension.torch_utils import install_cpp_extension
        install_cpp_extension(
            getattr(jt, "utils", None),
            registry=jt._torch_compat_install_context.registry,
        )
    except EXPECTED as exc:
        swallowed("shim/runtime.py enable: from jittor.compat.shim.cpp_extension.torch_utils impor...", exc)
        if strict_bootstrap:
            raise
        pass
    preloaded = _preload_jittor_cores(verbose=verbose)

    scanned: List[NativeExtension] = []
    if auto_scan_extensions:
        scanned.extend(scan_extension_dirs(project_root=project_dir, max_depth=max_scan_depth))
        py_roots = _pythonpath_extension_roots(project_dir, runtime)
        if py_roots:
            scanned.extend(scan_extension_dirs(roots=py_roots, max_depth=max_scan_depth))
    for item in extension_dirs or ():
        if not isinstance(item, NativeExtension):
            item_path = pathlib.Path(os.fspath(item)).expanduser()
            if not item_path.is_absolute():
                item = project_dir / item_path
        ext = _extension_from_user_item(item)
        if ext is not None:
            scanned.append(ext)
    scanned = _dedupe_extensions(scanned)

    for ext in scanned:
        if _transaction is None:
            append_sys_path(ext.root)
        else:
            _transaction.mutate_path(sys.path, os.fspath(ext.root), prepend=False)

    child_paths: List[Union[str, os.PathLike]] = [shim_site, jt_python, project_dir]
    child_paths += [ext.root for ext in scanned]
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = _pythonpath_for_child(child_paths)

    built: List[str] = []
    skip_build = is_truthy(os.environ.get("JITTOR_TORCH_SKIP_EXT_BUILD"))
    if build_extensions and not skip_build:
        buildable = [ext for ext in scanned if ext.setup_py]
        if buildable:
            built = build_extension_dirs(
                buildable,
                env=child_env,
                force=is_truthy(os.environ.get("JITTOR_TORCH_FORCE_EXT_BUILD")),
                verbose=verbose,
            )
    elif skip_build:
        _log(verbose, "skip extension build by environment")

    _log(verbose, "runtime: %s" % runtime)
    if scanned:
        _log(verbose, "native extensions: %s" % ", ".join(ext.root for ext in scanned))
    from jittor.compat.integrations import apply_external_runtime_patches

    integration_report = apply_external_runtime_patches(
        logger=getattr(getattr(jt, "compiler", None), "LOG", None),
        transaction=_transaction,
    )

    return {
        "torch": published,
        "runtime_root": os.fspath(runtime),
        "shim_site": os.fspath(shim_site),
        "extensions": scanned,
        "built": built,
        "preloaded": preloaded,
        "module_patches": integration_report.get("module_patches"),
        "external_backends": integration_report.get("external_backends"),
        "integrations": integration_report,
    }


def activate(
    project_root: Optional[Union[str, os.PathLike]] = None,
    runtime_root: Optional[Union[str, os.PathLike]] = None,
    import_paths: Optional[Sequence[Union[str, os.PathLike]]] = None,
    extension_dirs: Optional[Sequence[Union[NativeExtension, str, os.PathLike]]] = None,
    auto_scan_extensions: bool = True,
    build_extensions: bool = True,
    max_scan_depth: int = 5,
    local_home: bool = True,
    configure_cuda: bool = True,
    inference: bool = False,
    verbose: Optional[bool] = None,
    strict: Optional[bool] = None,
    _root_module: Any = None,
    _preflight_result: Any = None,
    _composition: bool = False,
    independent_namespace: bool = False,
):
    """Activate Torch compatibility exactly once for this process.

    Repeated calls return the original result and never rescan extensions or
    reapply integration patches. Use :func:`activation_status` for inspection.
    """

    root = _root_module or sys.modules.get("jittor")
    if root is None:
        raise RuntimeError("import jittor before activating Torch compatibility")
    state = _runtime_state(root)
    already_installed = bool(state.get("installed"))
    if state.get("installed"):
        if not torch_namespace_owned(root):
            raise RuntimeError(
                "cannot re-activate the Jittor Torch shim over a changed Torch "
                "module graph"
            )
        if _composition or state.get("runtime_configured"):
            return state.get("result")
    if state.get("phase") == "activating":
        raise RuntimeError("recursive Jittor Torch shim activation")

    state.update(phase="activating", error=None)
    transaction = ActivationTransaction("shim.activate")
    transaction.acquire()
    try:
        result = _activate_once(
            project_root=project_root,
            runtime_root=runtime_root,
            import_paths=import_paths,
            extension_dirs=extension_dirs,
            auto_scan_extensions=auto_scan_extensions,
            build_extensions=build_extensions,
            max_scan_depth=max_scan_depth,
            local_home=local_home,
            configure_cuda=configure_cuda,
            inference=inference,
            verbose=verbose,
            strict=strict,
            _root_module=root,
            _preflight_result=_preflight_result,
            _composition=_composition,
            _transaction=transaction,
            independent_namespace=independent_namespace,
        )
    except EXPECTED as exc:
        transaction.rollback()
        transaction.release()
        state.update(
            phase="active" if already_installed else "failed",
            installed=already_installed,
            error=str(exc),
        )
        raise
    transaction.commit()
    transaction.release()
    state.update(
        phase="active",
        installed=True,
        result=result,
        external_patches=(
            result.get("integrations") if isinstance(result, dict) else None
        ),
        error=None,
        runtime_configured=not _composition,
    )
    return result


# Historical spelling; identity makes the canonical implementation observable.
enable = activate
