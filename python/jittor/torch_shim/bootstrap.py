from __future__ import annotations

import ctypes
import dataclasses
import glob
import importlib.util
import os
import pathlib
import subprocess
import sys
import ast
import hashlib
from typing import Iterable, List, Optional, Sequence, Tuple, Union


_TRUTHY = {"1", "true", "yes", "on"}
_NATIVE_SUFFIXES = (".cu", ".cuh", ".cpp", ".cc", ".cxx")
_PRUNE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "build",
    "dist",
    ".eggs",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".jittor_torch_runtime",
}


@dataclasses.dataclass(frozen=True)
class NativeExtension:
    """A local project directory that appears to build a native Python module."""

    root: str
    setup_py: Optional[str] = None
    pyproject_toml: Optional[str] = None
    cmake_lists: Optional[str] = None
    sources: Tuple[str, ...] = ()
    reason: str = ""

    @property
    def build_backend(self) -> str:
        return "setuptools" if self.setup_py else "unknown"


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print("[jittor.torch_shim] " + message)


def _is_truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in _TRUTHY


def _as_path(path: Optional[Union[str, os.PathLike]]) -> pathlib.Path:
    if path is None:
        return pathlib.Path.cwd()
    return pathlib.Path(os.fspath(path)).expanduser()


def _project_dir(path: Optional[Union[str, os.PathLike]]) -> pathlib.Path:
    p = _as_path(path)
    if p.exists() and p.is_file():
        return p.parent.resolve()
    # __file__ can be relative before the file exists in some launchers.
    if str(p).endswith(".py"):
        return p.parent.resolve()
    return p.resolve()


def _jittor_python_root() -> pathlib.Path:
    # .../python/jittor/torch_shim/bootstrap.py -> .../python
    return pathlib.Path(__file__).resolve().parents[2]


def _prepend_sys_path(path: Union[str, os.PathLike], after: Optional[Union[str, os.PathLike]] = None) -> None:
    s = os.fspath(path)
    if not s:
        return
    if s in sys.path:
        sys.path.remove(s)
    if after is not None:
        marker = os.fspath(after)
        if marker in sys.path:
            sys.path.insert(sys.path.index(marker) + 1, s)
            return
    sys.path.insert(0, s)


def _prepend_env_path(name: str, paths: Iterable[Union[str, os.PathLike]]) -> None:
    current = [p for p in os.environ.get(name, "").split(os.pathsep) if p]
    out: List[str] = []
    for p in paths:
        s = os.fspath(p)
        if s and s not in out:
            out.append(s)
    for p in current:
        if p not in out:
            out.append(p)
    os.environ[name] = os.pathsep.join(out)


def _read_text(path: pathlib.Path, limit: int = 256 * 1024) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return f.read(limit)
    except OSError:
        return ""


def _rel_depth(root: pathlib.Path, path: pathlib.Path) -> int:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return 999999
    return len(rel.parts)


def _native_sources(root: pathlib.Path, max_depth: int) -> Tuple[str, ...]:
    found: List[str] = []
    for cur, dirs, files in os.walk(os.fspath(root)):
        cur_path = pathlib.Path(cur)
        dirs[:] = [
            d for d in dirs
            if d not in _PRUNE_DIRS and not d.startswith(".jittor")
            and _rel_depth(root, cur_path / d) <= max_depth
        ]
        if _rel_depth(root, cur_path) > max_depth:
            continue
        for name in files:
            if name.endswith(_NATIVE_SUFFIXES):
                found.append(os.fspath(cur_path / name))
    found.sort()
    return tuple(found)


def _setup_has_torch_extension_signal(text: str) -> bool:
    signals = (
        "torch.utils.cpp_extension",
        "CUDAExtension",
        "CppExtension",
        "BuildExtension",
        ".cu",
        ".cpp",
        ".cc",
    )
    return any(s in text for s in signals)


def _literal_string_list(node: ast.AST) -> Optional[List[str]]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    out: List[str] = []
    for item in node.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            out.append(item.value)
            continue
        return None
    return out


def _setup_declared_sources(setup_py: pathlib.Path) -> Tuple[str, ...]:
    text = _read_text(setup_py)
    if not text:
        return ()
    try:
        tree = ast.parse(text, filename=os.fspath(setup_py))
    except SyntaxError:
        return ()
    root = setup_py.parent
    found: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "sources":
                continue
            values = _literal_string_list(kw.value)
            if values is None:
                continue
            for value in values:
                if value.endswith(_NATIVE_SUFFIXES):
                    found.append(os.fspath((root / value).resolve()))
    return tuple(dict.fromkeys(found))


def _pyproject_has_native_signal(text: str) -> bool:
    signals = (
        "torch.utils.cpp_extension",
        "CUDAExtension",
        "CppExtension",
        "scikit-build",
        "cmake",
        "pybind11",
        "cuda",
    )
    low = text.lower()
    return any(s.lower() in low for s in signals)


def _cmake_has_native_signal(text: str) -> bool:
    signals = (
        "cuda",
        "pybind11_add_module",
        "python_add_library",
        "add_library",
        "torch",
    )
    low = text.lower()
    return any(s.lower() in low for s in signals)


def _candidate_roots(root: pathlib.Path, max_depth: int) -> List[pathlib.Path]:
    roots: List[pathlib.Path] = []
    if not root.exists():
        return roots
    for cur, dirs, files in os.walk(os.fspath(root)):
        cur_path = pathlib.Path(cur)
        dirs[:] = [
            d for d in dirs
            if d not in _PRUNE_DIRS and not d.startswith(".jittor")
            and _rel_depth(root, cur_path / d) <= max_depth
        ]
        if _rel_depth(root, cur_path) > max_depth:
            continue
        names = set(files)
        if {"setup.py", "pyproject.toml", "CMakeLists.txt"} & names:
            roots.append(cur_path)
    roots.sort(key=lambda p: (len(p.parts), os.fspath(p)))
    return roots


def _extension_from_root(root: pathlib.Path, max_source_depth: int = 4) -> Optional[NativeExtension]:
    setup_py = root / "setup.py"
    pyproject = root / "pyproject.toml"
    cmake = root / "CMakeLists.txt"
    setup_sources = _setup_declared_sources(setup_py) if setup_py.is_file() else ()
    sources = setup_sources or _native_sources(root, max_source_depth)
    reasons: List[str] = []

    setup_path = None
    if setup_py.is_file():
        text = _read_text(setup_py)
        if _setup_has_torch_extension_signal(text) or sources:
            setup_path = os.fspath(setup_py)
            reasons.append("setup.py")

    pyproject_path = None
    if pyproject.is_file():
        text = _read_text(pyproject)
        if _pyproject_has_native_signal(text) or sources:
            pyproject_path = os.fspath(pyproject)
            reasons.append("pyproject.toml")

    cmake_path = None
    if cmake.is_file():
        text = _read_text(cmake)
        if _cmake_has_native_signal(text) or sources:
            cmake_path = os.fspath(cmake)
            reasons.append("CMakeLists.txt")

    if not (setup_path or pyproject_path or cmake_path):
        return None
    if sources:
        reasons.append("%d native source(s)" % len(sources))
    return NativeExtension(
        root=os.fspath(root),
        setup_py=setup_path,
        pyproject_toml=pyproject_path,
        cmake_lists=cmake_path,
        sources=sources,
        reason=", ".join(reasons),
    )


def _dedupe_extensions(items: Iterable[NativeExtension]) -> List[NativeExtension]:
    seen = set()
    out: List[NativeExtension] = []
    for ext in sorted(items, key=lambda e: (len(pathlib.Path(e.root).parts), e.root)):
        key = os.path.realpath(ext.root)
        if key in seen:
            continue
        ext_path = pathlib.Path(key)
        if not ext.setup_py:
            skip = False
            for parent in out:
                if not parent.setup_py:
                    continue
                try:
                    ext_path.relative_to(os.path.realpath(parent.root))
                except ValueError:
                    continue
                skip = True
                break
            if skip:
                continue
        seen.add(key)
        out.append(ext)
    return out


def scan_extension_dirs(
    roots: Optional[Sequence[Union[str, os.PathLike]]] = None,
    project_root: Optional[Union[str, os.PathLike]] = None,
    max_depth: int = 5,
) -> List[NativeExtension]:
    """Scan local project trees for native extension build directories.

    The scanner is intentionally generic: it recognizes setuptools projects that
    use ``torch.utils.cpp_extension`` as well as pyproject/CMake native-module
    signals. Only setuptools ``setup.py`` projects are built automatically by
    :func:`enable`; other results are reported for callers that want to build
    them explicitly.
    """

    base_roots = list(roots or [])
    if not base_roots:
        base_roots = [project_root or pathlib.Path.cwd()]

    found: List[NativeExtension] = []
    for raw in base_roots:
        root = _project_dir(raw)
        if (root / "setup.py").is_file() or (root / "pyproject.toml").is_file() or (root / "CMakeLists.txt").is_file():
            ext = _extension_from_root(root)
            if ext is not None:
                found.append(ext)
        for cand in _candidate_roots(root, max_depth=max_depth):
            ext = _extension_from_root(cand)
            if ext is not None:
                found.append(ext)
    return _dedupe_extensions(found)


def _is_relative_to(path: pathlib.Path, parent: pathlib.Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _pythonpath_extension_roots(project_dir: pathlib.Path, runtime: pathlib.Path) -> List[pathlib.Path]:
    """Return explicit PYTHONPATH entries worth scanning for native extensions.

    TRELLIS-style projects keep torch-extension dependencies as sibling source
    trees on PYTHONPATH.  Scanning those explicit entries lets ``import jittor as
    torch`` build unmodified external packages in place, without vendoring their
    sources into Jittor.  Broad locations such as site-packages, the Jittor source
    tree and runtime/cache directories are intentionally skipped.
    """

    raw_paths: List[str] = []
    for item in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if item and item not in raw_paths:
            raw_paths.append(item)

    jt_root = _jittor_python_root().resolve()
    prefixes = []
    for raw_prefix in (sys.prefix, getattr(sys, "base_prefix", "")):
        if raw_prefix:
            try:
                prefixes.append(pathlib.Path(raw_prefix).resolve())
            except OSError:
                pass
    roots: List[pathlib.Path] = []
    for raw in raw_paths:
        try:
            p = pathlib.Path(raw).expanduser().resolve()
        except OSError:
            continue
        if not p.is_dir():
            continue
        parts = set(p.parts)
        if "site-packages" in parts or "dist-packages" in parts:
            continue
        if any(p == prefix or _is_relative_to(p, prefix) for prefix in prefixes):
            continue
        if p == jt_root or _is_relative_to(p, jt_root):
            continue
        if p == runtime or _is_relative_to(p, runtime):
            continue
        if p == project_dir:
            continue
        roots.append(p)
    return sorted(dict.fromkeys(roots), key=lambda x: (len(x.parts), os.fspath(x)))


def _ensure_dir(path: Union[str, os.PathLike]) -> pathlib.Path:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _set_env_dir(name: str, path: pathlib.Path, override: bool = False) -> None:
    if override or not os.environ.get(name):
        os.environ[name] = os.fspath(path)
    _ensure_dir(os.environ[name])


def _configure_torch_math_flags(jt) -> None:
    if _is_truthy(os.environ.get("JITTOR_TORCH_KEEP_FAST_MATH")):
        return
    extra = "--fmad=false --prec-div=true --prec-sqrt=true"
    env_flags = os.environ.get("nvcc_flags", "")
    for tok in extra.split():
        if tok not in env_flags.split():
            env_flags = (env_flags + " " + tok).strip()
    os.environ["nvcc_flags"] = env_flags
    try:
        flags = getattr(getattr(jt, "compiler", None), "flags", None)
        cur = getattr(flags, "nvcc_flags", None)
        if isinstance(cur, str):
            cur = cur.replace("--use_fast_math", "")
            for tok in extra.split():
                if tok not in cur:
                    cur += " " + tok
            flags.nvcc_flags = cur
    except Exception:
        pass


def _best_jtcuda(real_home: Optional[str]) -> Optional[pathlib.Path]:
    candidates: List[str] = []
    if os.environ.get("JTCUDA"):
        candidates.append(os.environ["JTCUDA"])
    homes = []
    if real_home:
        homes.append(real_home)
    if os.environ.get("HOME") and os.environ["HOME"] not in homes:
        homes.append(os.environ["HOME"])
    for home in homes:
        candidates.extend(glob.glob(os.path.join(home, ".cache", "jittor", "jtcuda", "cuda*_linux")))
    valid = []
    for c in candidates:
        p = pathlib.Path(c)
        if (p / "bin" / "nvcc").is_file():
            valid.append(p)
    if not valid:
        return None
    valid.sort(key=lambda p: (("cuda12.2" not in p.name), os.fspath(p)))
    return valid[0]


def _configure_cuda(real_home: Optional[str], verbose: bool) -> None:
    jtcuda = _best_jtcuda(real_home)
    if jtcuda is None:
        return
    os.environ.setdefault("JTCUDA", os.fspath(jtcuda))
    os.environ.setdefault("nvcc_path", os.fspath(jtcuda / "bin" / "nvcc"))
    os.environ.setdefault("CUDA_HOME", os.fspath(jtcuda))
    _prepend_env_path("PATH", [jtcuda / "bin"])
    _prepend_env_path("LD_LIBRARY_PATH", [jtcuda / "lib64"])
    _log(verbose, "CUDA toolkit: %s" % jtcuda)


def _configure_runtime_driver_lib(runtime: pathlib.Path) -> None:
    """Expose an unversioned libcuda.so for Triton's build helper.

    Driver packages often install only ``libcuda.so.1`` for x86_64, while Triton
    links its small runtime helper with ``-lcuda``.  Keep the compatibility
    symlink inside Jittor's runtime instead of touching system or dependency
    directories.
    """

    candidates = (
        pathlib.Path("/lib/x86_64-linux-gnu/libcuda.so.1"),
        pathlib.Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
    )
    src = next((p for p in candidates if p.is_file()), None)
    if src is None:
        return
    lib_dir = _ensure_dir(runtime / "lib")
    dst = lib_dir / "libcuda.so"
    try:
        if dst.exists() or dst.is_symlink():
            cur = pathlib.Path(os.readlink(dst)) if dst.is_symlink() else None
            if cur == src:
                _prepend_env_path("LD_LIBRARY_PATH", [lib_dir])
                os.environ.setdefault("TRITON_LIBCUDA_PATH", os.fspath(lib_dir))
                return
            dst.unlink()
        dst.symlink_to(src)
    except OSError:
        return
    _prepend_env_path("LD_LIBRARY_PATH", [lib_dir])
    os.environ.setdefault("TRITON_LIBCUDA_PATH", os.fspath(lib_dir))


def _deploy_torch_shim(target: pathlib.Path) -> None:
    deploy_py = _jittor_python_root() / "jittor" / "torch_shim" / "deploy.py"
    spec = importlib.util.spec_from_file_location("_jittor_torch_deploy", os.fspath(deploy_py))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load torch shim deploy module from %s" % deploy_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.deploy(os.fspath(target))


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
        hits = glob.glob(os.path.join(os.fspath(search_root), "**", name + ".*.so"), recursive=True)
        hits.sort(key=lambda p: len(p))
        for so in hits[:1]:
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                loaded.append(so)
                _prepend_env_path("LD_LIBRARY_PATH", [pathlib.Path(so).parent])
            except Exception as e:
                _log(verbose, "could not preload %s: %s" % (so, e))
    return loaded


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


def _needs_build(ext: NativeExtension) -> bool:
    outputs = [
        p for p in _extension_outputs(ext.root)
        if os.path.sep + "build" + os.path.sep not in p
    ]
    if not outputs:
        return True
    try:
        from jittor.torch_shim import cpp_extension as _cpp_ext
        for path in outputs:
            if not _cpp_ext.output_matches_toolchain(path):
                return True
    except Exception:
        return True
    inputs = [p for p in (ext.setup_py, ext.pyproject_toml, ext.cmake_lists) if p]
    inputs += list(ext.sources)
    if not inputs:
        return False
    newest_input = 0.0
    for path in inputs:
        try:
            newest_input = max(newest_input, os.path.getmtime(path))
        except OSError:
            return True
    newest_output = 0.0
    for path in outputs:
        try:
            newest_output = max(newest_output, os.path.getmtime(path))
        except OSError:
            return True
    return newest_output < newest_input


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
        if not force and not _needs_build(ext):
            _log(verbose, "extension up-to-date: %s" % ext.root)
            continue
        _log(verbose, "build_ext: %s" % ext.root)
        child_env = (env or os.environ.copy()).copy()
        ext_root = pathlib.Path(ext.root).resolve()
        build_root = pathlib.Path(
            child_env.get("JITTOR_TORCH_EXTENSIONS_DIR")
            or os.environ.get("JITTOR_TORCH_EXTENSIONS_DIR")
            or (pathlib.Path.home() / ".cache" / "jittor_torch_extensions")
        ).expanduser().resolve()
        digest = hashlib.sha256(os.fspath(ext_root).encode("utf-8")).hexdigest()[:16]
        build_temp = _ensure_dir(build_root / "setuptools" / ext_root.name / digest / "temp")
        build_lib = _ensure_dir(build_root / "setuptools" / ext_root.name / digest / "lib")
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
        subprocess.run(cmd, cwd=ext.root, env=child_env, check=True)
        try:
            from jittor.torch_shim import cpp_extension as _cpp_ext
            for path in _extension_outputs(ext.root):
                if os.path.sep + "build" + os.path.sep not in path:
                    _cpp_ext.write_toolchain_stamp(path, {"root": ext.root})
        except Exception:
            pass
        built.append(ext.root)
    return built


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
):
    """Enable Jittor-backed ``import torch`` for the current Python process.

    This is a general project bootstrap, not a gaussian-splatting adapter. It
    sets project-local cache directories, deploys the torch shim into that local
    runtime, registers Jittor as ``torch`` in-process, scans local native
    extension projects, and builds setuptools extensions through Jittor's
    ``torch.utils.cpp_extension`` facade.

    Typical use in a torch-oriented project entrypoint::

        from jittor.torch_shim import enable as _enable_torch_shim
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

    verbose = (not _is_truthy(os.environ.get("JITTOR_TORCH_QUIET"))) if verbose is None else bool(verbose)
    project_dir = _project_dir(project_root or pathlib.Path.cwd())
    real_home = os.environ.get("REAL_HOME") or os.environ.get("HOME")
    runtime = pathlib.Path(
        os.environ.get("JITTOR_TORCH_RUNTIME_ROOT")
        or os.fspath(runtime_root or (project_dir / ".cache" / "jittor_torch"))
    ).expanduser().resolve()
    _ensure_dir(runtime)

    os.environ.setdefault("REAL_HOME", real_home or os.fspath(pathlib.Path.home()))
    if local_home and not _is_truthy(os.environ.get("JITTOR_TORCH_KEEP_HOME")):
        os.environ["HOME"] = os.environ.get("JITTOR_TORCH_HOME", os.fspath(runtime / "home"))
        _ensure_dir(os.environ["HOME"])

    _set_env_dir("JITTOR_HOME", runtime / "jittor_cache")
    _set_env_dir("TORCH_HOME", runtime / "torch_home")
    _set_env_dir("JITTOR_TORCH_EXTENSIONS_DIR", runtime / "torch_extensions")
    _set_env_dir("TORCH_EXTENSIONS_DIR", runtime / "torch_extensions")
    _set_env_dir("TMPDIR", runtime / "tmp", override=not _is_truthy(os.environ.get("JITTOR_TORCH_KEEP_TMPDIR")))
    _set_env_dir("XDG_CACHE_HOME", runtime / "xdg_cache")
    _set_env_dir("CUDA_CACHE_PATH", runtime / "cuda_cache")
    _set_env_dir("TRITON_HOME", runtime / "triton_home")
    _set_env_dir("TRITON_CACHE_DIR", runtime / "triton_home" / "cache")
    _set_env_dir("TRITON_OVERRIDE_DIR", runtime / "triton_home" / "override")
    _set_env_dir("TRITON_DUMP_DIR", runtime / "triton_home" / "dump")
    _set_env_dir("PIP_CACHE_DIR", runtime / "pip_cache")
    flex_cache = runtime / "flex_gemm" / "autotune_cache.json"
    os.environ.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH", os.fspath(flex_cache))
    _ensure_dir(pathlib.Path(os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"]).parent)

    if configure_cuda:
        _configure_cuda(real_home, verbose=verbose)
    _configure_runtime_driver_lib(runtime)

    for name, value in (
        ("DISABLE_MULTIPROCESSING", "1"),
        ("use_cutt", "0"),
        ("use_cutlass", "0"),
        ("use_nccl", "0"),
        ("use_mkl", "0"),
    ):
        os.environ.setdefault(name, value)

    shim_site = _ensure_dir(runtime / "site-packages")
    jt_python = _jittor_python_root()
    _deploy_torch_shim(shim_site)
    _write_build_sitecustomize(shim_site)

    _prepend_sys_path(shim_site)
    _prepend_sys_path(jt_python)
    _prepend_sys_path(project_dir)
    for p in import_paths or ():
        pp = pathlib.Path(p)
        if not pp.is_absolute():
            pp = project_dir / pp
        _prepend_sys_path(pp.resolve())

    import jittor as jt
    _configure_torch_math_flags(jt)
    if inference:
        jt.flags.no_grad = 1
    try:
        from jittor import torch_compat
        torch_compat.install(jt)
    except Exception:
        pass
    sys.modules["torch"] = jt
    try:
        from jittor.torch_shim.cpp_extension.torch_utils import install_cpp_extension
        install_cpp_extension(getattr(jt, "utils", None))
    except Exception:
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

    for ext in reversed(scanned):
        _prepend_sys_path(ext.root, after=project_dir)

    child_paths: List[Union[str, os.PathLike]] = [shim_site, jt_python, project_dir]
    child_paths += [ext.root for ext in scanned]
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = _pythonpath_for_child(child_paths)

    built: List[str] = []
    skip_build = _is_truthy(os.environ.get("JITTOR_TORCH_SKIP_EXT_BUILD"))
    if build_extensions and not skip_build:
        buildable = [ext for ext in scanned if ext.setup_py]
        if buildable:
            built = build_extension_dirs(
                buildable,
                env=child_env,
                force=_is_truthy(os.environ.get("JITTOR_TORCH_FORCE_EXT_BUILD")),
                verbose=verbose,
            )
    elif skip_build:
        _log(verbose, "skip extension build by environment")

    _log(verbose, "runtime: %s" % runtime)
    if scanned:
        _log(verbose, "native extensions: %s" % ", ".join(ext.root for ext in scanned))

    return {
        "torch": jt,
        "runtime_root": os.fspath(runtime),
        "shim_site": os.fspath(shim_site),
        "extensions": scanned,
        "built": built,
        "preloaded": preloaded,
    }
