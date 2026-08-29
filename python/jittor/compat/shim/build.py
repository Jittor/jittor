"""Build and staleness handling for discovered native extensions."""

from __future__ import annotations

import ctypes
import glob
import importlib.machinery
import hashlib
import os
import pathlib
import subprocess
import sys
from typing import List, Optional, Sequence, Tuple, Union

from .discovery import (
    NativeExtension, _extension_from_root, _log,
)
from .preflight import (
    _ensure_dir,
    _prepend_env_path,
    project_dir as _project_dir,
)

def _deploy_torch_shim(target: pathlib.Path) -> None:
    from jittor.compat.shim import deploy

    deploy.deploy(os.fspath(target))


def _write_build_sitecustomize(target: pathlib.Path) -> None:
    """Install a tiny sitecustomize for extension-build child processes.

    Some editable/package installs add internal package directories to
    ``sys.path``.  If a directory such as ``.../site-packages/tvm_ffi`` is on the
    path, stdlib imports like ``import dataclasses`` can resolve to
    ``tvm_ffi/dataclasses`` during ``setup.py`` startup.  Keep the fix local to
    Jittor's runtime shim site so the user's conda environment is untouched.
    """

    path = target / "sitecustomize.py"
    path.write_text(
        """
import importlib.util as _iu
import os as _os
import sys as _sys
import sysconfig as _sc

def _jt_drop_internal_package_paths():
    _bad = (
        _os.path.sep + "site-packages" + _os.path.sep + "tvm_ffi",
    )
    _sys.path[:] = [
        _p for _p in _sys.path
        if not any(_p.endswith(_s) for _s in _bad)
    ]

def _jt_preload_stdlib_dataclasses():
    _stdlib = _sc.get_path("stdlib")
    if not _stdlib:
        return
    _path = _os.path.join(_stdlib, "dataclasses.py")
    if not _os.path.isfile(_path):
        return
    _cur = _sys.modules.get("dataclasses")
    _cur_file = _os.path.abspath(str(getattr(_cur, "__file__", ""))) if _cur else ""
    if _cur_file == _os.path.abspath(_path):
        return
    _spec = _iu.spec_from_file_location("dataclasses", _path)
    if _spec is None or _spec.loader is None:
        return
    _mod = _iu.module_from_spec(_spec)
    _sys.modules["dataclasses"] = _mod
    _spec.loader.exec_module(_mod)

_jt_drop_internal_package_paths()
_jt_preload_stdlib_dataclasses()
""".lstrip(),
        encoding="utf-8",
    )


def _pythonpath_for_child(paths: Sequence[Union[str, os.PathLike]]) -> str:
    out: List[str] = []
    for p in paths:
        s = os.fspath(p)
        if s and s not in out:
            out.append(s)
    existing = os.environ.get("PYTHONPATH")
    if existing:
        for p in existing.split(os.pathsep):
            if p and p not in out:
                out.append(p)
    return os.pathsep.join(out)


def _preload_jittor_cores(verbose: bool) -> List[str]:
    loaded: List[str] = []
    try:
        import jittor as jt  # noqa: F401
        from jittor import compiler
        search_root = pathlib.Path(compiler.cache_path).parent
    except Exception:
        return loaded
    for name in ("jit_utils_core", "jittor_core"):
        # If the interpreter already imported this core, that exact file is the
        # only one that may be preloaded. The cache holds one build per
        # configuration -- a CPU one and a CUDA one, both for this Python -- and
        # picking by shortest path lands on the CPU build while the CUDA one is
        # already loaded. Two copies of the runtime's static state then free the
        # same exit-handler block twice and the process aborts with "double free
        # or corruption" once every test has already passed. Re-loading the file
        # that is in use is a no-op, which is the point.
        imported = sys.modules.get(name)
        origin = getattr(imported, "__file__", None) if imported is not None else None
        if origin:
            hits = [origin]
        else:
            hits = glob.glob(
                os.path.join(os.fspath(search_root), "**", name + ".*.so"),
                recursive=True)
            hits = _matching_abi(name, hits)
            hits.sort(key=lambda p: len(p))
        for so in hits[:1]:
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                loaded.append(so)
                _prepend_env_path(
                    os.environ, "LD_LIBRARY_PATH", pathlib.Path(so).parent
                )
            except Exception as e:
                _log(verbose, "could not preload %s: %s" % (so, e))
    return loaded


def _matching_abi(name: str, hits: Sequence[str]) -> List[str]:
    """Keep only builds this interpreter can load.

    The cache can hold a core built for another Python -- anything that shells
    out to the wrong ``python3`` leaves one behind. ``ctypes.CDLL`` loads it
    without complaint, and the process then carries two copies of the runtime's
    static state; at shutdown they free the same exit-handler block twice and
    the process aborts with "double free or corruption" after every test has
    already passed.
    """
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    return [
        hit for hit in hits
        if os.path.basename(hit)[len(name):] in suffixes
    ]


def _extension_from_user_item(item: Union[NativeExtension, str, os.PathLike]) -> Optional[NativeExtension]:
    if isinstance(item, NativeExtension):
        return item
    root = _project_dir(item)
    ext = _extension_from_root(root)
    if ext is not None:
        return ext
    setup_py = root / "setup.py"
    if setup_py.is_file():
        return NativeExtension(root=os.fspath(root), setup_py=os.fspath(setup_py), reason="explicit setup.py")
    return None


def _extension_outputs(root: str) -> List[str]:
    outputs: List[str] = []
    for cur, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {"build", "__pycache__", ".git"}]
        for name in files:
            if name.endswith((".so", ".pyd", ".dylib")):
                outputs.append(os.path.join(cur, name))
    outputs.sort()
    return outputs


def _extension_inputs(ext: NativeExtension) -> List[str]:
    inputs = [p for p in (ext.setup_py, ext.pyproject_toml, ext.cmake_lists) if p]
    inputs += list(ext.sources)
    return inputs


def _newest_mtime(paths: Sequence[str]) -> Optional[float]:
    if not paths:
        return None
    newest_input = 0.0
    for path in paths:
        try:
            newest_input = max(newest_input, os.path.getmtime(path))
        except OSError:
            return None
    return newest_input


def _same_file_contents(a: str, b: str) -> bool:
    try:
        if os.path.getsize(a) != os.path.getsize(b):
            return False
        with open(a, "rb") as fa, open(b, "rb") as fb:
            while True:
                ca = fa.read(1024 * 1024)
                cb = fb.read(1024 * 1024)
                if ca != cb:
                    return False
                if not ca:
                    return True
    except OSError:
        return False


def _setuptools_build_dirs(
    ext_root: pathlib.Path,
    env: Optional[dict] = None,
) -> Tuple[pathlib.Path, pathlib.Path]:
    env = env or {}
    build_root = pathlib.Path(
        env.get("JITTOR_TORCH_EXTENSIONS_DIR")
        or os.environ.get("JITTOR_TORCH_EXTENSIONS_DIR")
        or (pathlib.Path.home() / ".cache" / "jittor_torch_extensions")
    ).expanduser().resolve()
    digest = hashlib.sha256(os.fspath(ext_root).encode("utf-8")).hexdigest()[:16]
    base = build_root / "setuptools" / ext_root.name / digest
    return base / "temp", base / "lib"


def _setuptools_build_root(ext_root: pathlib.Path, env: Optional[dict] = None) -> pathlib.Path:
    build_temp, _build_lib = _setuptools_build_dirs(ext_root, env)
    return build_temp.parents[3]


def _cached_setuptools_outputs_current(
    ext: NativeExtension,
    inputs: Sequence[str],
    cpp_ext,
    env: Optional[dict],
) -> bool:
    if not ext.setup_py:
        return False
    ext_root = pathlib.Path(ext.root).resolve()
    _build_temp, build_lib = _setuptools_build_dirs(ext_root, env)
    if not build_lib.is_dir():
        return False
    cached_outputs = _extension_outputs(os.fspath(build_lib))
    if not cached_outputs:
        return False
    newest_input = _newest_mtime(inputs)
    if newest_input is None and inputs:
        return False

    valid_cached: List[Tuple[str, str]] = []
    for cached in cached_outputs:
        if not cpp_ext.output_matches_toolchain(cached):
            continue
        try:
            rel = pathlib.Path(cached).resolve().relative_to(build_lib)
        except ValueError:
            return False
        source_out = ext_root / rel
        if not source_out.is_file():
            return False
        if newest_input is not None and os.path.getmtime(os.fspath(source_out)) < newest_input:
            return False
        valid_cached.append((os.fspath(source_out), cached))

    if not valid_cached:
        return False

    for source_out, cached in valid_cached:
        if cpp_ext.output_matches_toolchain(source_out):
            continue
        if not _same_file_contents(source_out, cached):
            return False
        cpp_ext.write_toolchain_stamp(
            source_out,
            {"root": ext.root, "mirrored_from": cached},
        )
    return True


def _source_outputs_current(
    ext: NativeExtension,
    outputs: Sequence[str],
    inputs: Sequence[str],
    cpp_ext,
) -> bool:
    if not outputs:
        return False
    for path in outputs:
        if not cpp_ext.output_matches_toolchain(path):
            return False
    newest_input = _newest_mtime(inputs)
    if newest_input is None:
        return not inputs
    newest_output = 0.0
    for path in outputs:
        try:
            newest_output = max(newest_output, os.path.getmtime(path))
        except OSError:
            return False
    return newest_output >= newest_input


def _needs_build(ext: NativeExtension, env: Optional[dict] = None) -> bool:
    outputs = [
        p for p in _extension_outputs(ext.root)
        if os.path.sep + "build" + os.path.sep not in p
    ]
    inputs = _extension_inputs(ext)
    try:
        from jittor.compat.shim import cpp_extension as _cpp_ext
    except Exception:
        return True
    if _cached_setuptools_outputs_current(ext, inputs, _cpp_ext, env):
        return False
    return not _source_outputs_current(ext, outputs, inputs, _cpp_ext)


def _build_timeout_seconds() -> Optional[float]:
    raw = os.environ.get("JITTOR_TORCH_EXT_BUILD_TIMEOUT", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def build_extension_dirs(
    extensions: Sequence[Union[NativeExtension, str, os.PathLike]],
    env: Optional[dict] = None,
    force: bool = False,
    verbose: bool = True,
) -> List[str]:
    """Build setuptools native extension directories with the active Python."""

    built: List[str] = []
    for raw in extensions:
        ext = _extension_from_user_item(raw)
        if ext is None:
            continue
        if not ext.setup_py:
            _log(verbose, "skip native extension without setup.py: %s" % ext.root)
            continue
        child_env = (env or os.environ.copy()).copy()
        ext_root = pathlib.Path(ext.root).resolve()
        build_temp_path, build_lib_path = _setuptools_build_dirs(ext_root, child_env)
        build_root = _setuptools_build_root(ext_root, child_env)
        child_env["JITTOR_TORCH_EXTENSIONS_DIR"] = os.fspath(build_root)
        if not force and not _needs_build(ext, env=child_env):
            _log(verbose, "extension up-to-date: %s" % ext.root)
            continue
        _log(verbose, "build_ext: %s" % ext.root)
        build_temp = _ensure_dir(build_temp_path)
        build_lib = _ensure_dir(build_lib_path)
        child_env["JITTOR_TORCH_EXTENSIONS_DIR"] = os.fspath(build_root)
        cmd = [
            sys.executable,
            os.path.basename(ext.setup_py),
            "build_ext",
            "--inplace",
            "--build-temp",
            os.fspath(build_temp),
            "--build-lib",
            os.fspath(build_lib),
        ]
        try:
            subprocess.run(
                cmd,
                cwd=ext.root,
                env=child_env,
                check=True,
                timeout=_build_timeout_seconds(),
            )
        except subprocess.TimeoutExpired:
            if force or _needs_build(ext, env=child_env):
                raise
            _log(verbose, "build_ext timed out after usable outputs were produced: %s" % ext.root)
        try:
            from jittor.compat.shim import cpp_extension as _cpp_ext
            for path in _extension_outputs(ext.root):
                if os.path.sep + "build" + os.path.sep not in path:
                    _cpp_ext.write_toolchain_stamp(path, {"root": ext.root})
        except Exception:
            pass
        built.append(ext.root)
    return built
