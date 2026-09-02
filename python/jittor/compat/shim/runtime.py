"""Orchestration for enabling the canonical Jittor Torch shim."""

from __future__ import annotations

import os
import pathlib
import sys
from typing import List, Optional, Sequence, Union

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
from jittor.compat._aliases import torch_namespace_claimable
from ..diagnostics import EXPECTED, swallowed

def enable(
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
):
    """Enable Jittor-backed ``import torch`` for the current Python process.

    It sets project-local cache directories, deploys the torch shim into that
    runtime, registers Jittor as ``torch`` in-process, scans local native
    extension projects, and builds setuptools extensions through Jittor's
    ``torch.utils.cpp_extension`` facade.

    Typical use in a torch-oriented project entrypoint::

        from jittor.compat.shim import enable as _enable_torch_shim
        _enable_torch_shim(project_root=__file__)
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

    jittor_root = sys.modules.get("jittor")
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
    prepend_sys_path(shim_site)
    prepend_sys_path(jt_python)
    append_sys_path(project_dir)
    for p in import_paths or ():
        pp = pathlib.Path(p)
        if not pp.is_absolute():
            pp = project_dir / pp
        append_sys_path(pp.resolve())

    import jittor as jt
    configure_torch_math_flags(jt)
    if inference:
        jt.flags.no_grad = 1
    from jittor.compat import torch as torch_compat
    torch_compat.install(jt, strict=strict_bootstrap)
    sys.modules["torch"] = jt
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
        append_sys_path(ext.root)

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
        logger=getattr(getattr(jt, "compiler", None), "LOG", None)
    )

    return {
        "torch": jt,
        "runtime_root": os.fspath(runtime),
        "shim_site": os.fspath(shim_site),
        "extensions": scanned,
        "built": built,
        "preloaded": preloaded,
        "module_patches": integration_report.get("module_patches"),
        "external_backends": integration_report.get("external_backends"),
        "integrations": integration_report,
    }
