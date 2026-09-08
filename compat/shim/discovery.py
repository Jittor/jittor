"""Discovery of local native extension source trees.

This module performs no Jittor import at module load time.
"""

from __future__ import annotations

import ast
import dataclasses
import os
import pathlib
import sys
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from .preflight import jittor_python_root, project_dir as _project_dir
from ..diagnostics import swallowed

_TRUTHY = {"1", "true", "yes", "on"}
_NATIVE_SUFFIXES = (".cu", ".cuh", ".cpp", ".cc", ".cxx")
_PRUNE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".cache",
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
        print("[jittor.compat.shim] " + message)


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
    from jittor.compat.external_backend import external_backend_for_source_root

    if external_backend_for_source_root(root) is not None:
        return None
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

    from jittor.compat.external_backend import load_external_backend_entry_points

    load_external_backend_entry_points()
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

    Some projects keep torch-extension dependencies as sibling source trees on
    PYTHONPATH. Scanning those explicit entries lets ``import jittor as torch``
    build unmodified external packages in place. Broad locations such as
    site-packages, the Jittor source tree, and runtime directories are skipped.
    """

    raw_paths: List[str] = []
    for item in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if item and item not in raw_paths:
            raw_paths.append(item)

    jt_root = jittor_python_root().resolve()
    prefixes = []
    for raw_prefix in (sys.prefix, getattr(sys, "base_prefix", "")):
        if raw_prefix:
            try:
                prefixes.append(pathlib.Path(raw_prefix).resolve())
            except OSError as exc:
                swallowed("shim/discovery.py _pythonpath_extension_roots: prefixes.append(pathlib.Path(raw_prefix).resolve())", exc)
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
