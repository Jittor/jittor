"""Optional native flash-attn backend for the Jittor torch shim.

The bundled ``flash_attn`` package is a numerically correct Jittor SDPA
fallback.  This helper lets projects keep a fused CUDA implementation outside
the Jittor repository while still using the normal ``import flash_attn`` API:

* import an already installed ``flashattn_jittor`` style Python package;
* discover a source tree in the active project;
* build a setup.py project through the Jittor torch-extension shim;
* or compile sources listed in a small JSON manifest through
  ``torch.utils.cpp_extension.load``.

No real PyTorch/libtorch package is imported or linked here.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import pathlib
import sys
import glob
from types import ModuleType
from typing import List, Optional, Sequence, Tuple


_TRUTHY = {"1", "true", "yes", "on"}
_FALSEY = {"0", "false", "no", "off"}

_SRC_ENVS = (
    "JITTOR_FLASH_ATTN_JITTOR_SRC",
    "FLASHATTN_JITTOR_SRC",
    "FLASH_ATTN_JITTOR_SRC",
    "FLASHATTNJITTOR_SRC",
)
_MODULE_ENV = "JITTOR_FLASH_ATTN_JITTOR_MODULE"
_MANIFEST_NAMES = (
    "flashattn_jittor.json",
    "flash_attn_jittor.json",
    "jittor_flashattn.json",
)
_DEFAULT_MODULE_NAMES = (
    "flashattn_jittor",
    "flash_attn_jittor",
    "flashattnjittor",
    "flashattn_jittor_cuda",
    "flash_attn_jittor_cuda",
)
_PUBLIC_FUNCS = (
    "flash_attn_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_kvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
)
_HOOK_NAMES = (
    "load_jittor_flash_attn",
    "build_jittor_flash_attn",
    "load_flashattn_jittor",
    "build_flashattn_jittor",
)
_SUBMODULE_ATTRS = (
    "_C",
    "cuda",
    "ops",
    "flashattn_jittor_cuda",
    "flash_attn_jittor_cuda",
)
_RELATIVE_SOURCE_DIRS = (
    "flashattn_jittor",
    "flash_attn_jittor",
    "flashattnjittor",
    "flash-attention-jittor",
    "third_party/flashattn_jittor",
    "third_party/flash_attn_jittor",
    "third_party/flash-attention-jittor",
    "extern/flashattn_jittor",
    "extern/flash_attn_jittor",
    "extensions/flashattn_jittor",
    "extensions/flash_attn_jittor",
)
_SOURCE_ROOT_NAMES = set(_DEFAULT_MODULE_NAMES + ("flash-attention-jittor",))

_UNSET = object()
_BACKEND = _UNSET
_BACKEND_NAME = "math"
_LAST_ERROR: Optional[str] = None
_LOADING = False


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in _TRUTHY


def _falsey(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in _FALSEY


def enabled() -> bool:
    value = os.environ.get("JITTOR_FLASH_ATTN_JITTOR")
    if value is None:
        value = os.environ.get("JITTOR_FLASHATTN_JITTOR")
    return not _falsey(value)


def required() -> bool:
    return (
        _truthy(os.environ.get("JITTOR_FLASH_ATTN_JITTOR_REQUIRED"))
        or _truthy(os.environ.get("JITTOR_FLASHATTN_JITTOR_REQUIRED"))
    )


def _verbose() -> bool:
    return _truthy(os.environ.get("JITTOR_FLASH_ATTN_JITTOR_VERBOSE"))


def _log(message: str) -> None:
    if _verbose():
        print("[jittor.flashattn_jittor] " + message)


def _split_env_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    out: List[str] = []
    for item in value.replace(",", os.pathsep).split(os.pathsep):
        item = item.strip()
        if item and item not in out:
            out.append(item)
    return out


def _module_names() -> List[str]:
    names = _split_env_list(os.environ.get(_MODULE_ENV))
    for name in _DEFAULT_MODULE_NAMES:
        if name not in names:
            names.append(name)
    return names


def _project_roots() -> List[pathlib.Path]:
    roots: List[pathlib.Path] = []
    for name in (
        "JITTOR_FLASH_ATTN_JITTOR_PROJECT_ROOT",
        "JITTOR_TORCH_PROJECT_ROOT",
        "TRELLIS2_ROOT",
        "TRELLIS_ROOT",
    ):
        value = os.environ.get(name)
        if value:
            roots.append(pathlib.Path(value).expanduser())

    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 and argv0 not in ("-c", "-m"):
        path = pathlib.Path(argv0).expanduser()
        if path.suffix == ".py":
            roots.append(path.parent)

    try:
        roots.append(pathlib.Path.cwd())
    except OSError:
        pass

    runtime = os.environ.get("JITTOR_TORCH_RUNTIME_ROOT")
    if runtime:
        p = pathlib.Path(runtime).expanduser()
        if p.name == "jittor_torch" and p.parent.name == ".cache":
            roots.append(p.parent.parent)

    out: List[pathlib.Path] = []
    seen = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        key = os.fspath(resolved)
        if key not in seen:
            seen.add(key)
            out.append(resolved)
    return out


def candidate_source_roots() -> List[str]:
    roots: List[Tuple[pathlib.Path, bool]] = []
    for name in _SRC_ENVS:
        for raw in _split_env_list(os.environ.get(name)):
            roots.append((pathlib.Path(raw).expanduser(), True))

    for base in _project_roots():
        roots.append((base, False))
        for rel in _RELATIVE_SOURCE_DIRS:
            roots.append((base / rel, False))

    out: List[str] = []
    seen = set()
    for root, explicit in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        key = os.fspath(resolved)
        if key in seen or not resolved.is_dir():
            continue
        if _looks_like_source_root(resolved, explicit=explicit):
            seen.add(key)
            out.append(key)
    return out


def explicit_source_roots() -> List[str]:
    out: List[str] = []
    seen = set()
    for name in _SRC_ENVS:
        for raw in _split_env_list(os.environ.get(name)):
            root = pathlib.Path(raw).expanduser()
            try:
                resolved = root.resolve()
            except OSError:
                continue
            key = os.fspath(resolved)
            if key in seen or not resolved.is_dir():
                continue
            if _looks_like_source_root(resolved, explicit=True):
                seen.add(key)
                out.append(key)
    return out


def _looks_like_source_root(root: pathlib.Path, explicit: bool = False) -> bool:
    if any((root / name).is_file() for name in _MANIFEST_NAMES):
        return True
    if explicit and (root / "setup.py").is_file():
        return True
    if (root / "build_jittor.py").is_file():
        return True
    if (root / "__init__.py").is_file() and root.name in _SOURCE_ROOT_NAMES:
        return True
    for name in _DEFAULT_MODULE_NAMES:
        if (root / name / "__init__.py").is_file():
            return True
    if root.name in _SOURCE_ROOT_NAMES and (root / "setup.py").is_file():
        return True
    return False


def _prepend_sys_path(path: pathlib.Path) -> None:
    text = os.fspath(path)
    if not text:
        return
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)


def _add_source_to_sys_path(root: pathlib.Path) -> None:
    _prepend_sys_path(root)
    _prepend_sys_path(root.parent)


def _has_public_api(mod: object) -> bool:
    return any(callable(getattr(mod, name, None)) for name in _PUBLIC_FUNCS)


def _select_backend(mod: ModuleType, allow_hooks: bool = True) -> Optional[ModuleType]:
    if _has_public_api(mod):
        return mod

    for attr in _SUBMODULE_ATTRS:
        sub = getattr(mod, attr, None)
        if isinstance(sub, ModuleType) and _has_public_api(sub):
            return sub

    if not allow_hooks:
        return None

    for hook_name in _HOOK_NAMES:
        hook = getattr(mod, hook_name, None)
        if not callable(hook):
            continue
        result = _call_hook(hook)
        selected = _coerce_backend(result)
        if selected is not None:
            return selected

    return None


def _coerce_backend(obj: object) -> Optional[ModuleType]:
    if isinstance(obj, ModuleType):
        return _select_backend(obj, allow_hooks=False) or (obj if _has_public_api(obj) else None)
    return None


def _call_hook(hook):
    build_root = _default_build_root("hooks")
    kwargs = {
        "build_root": build_root,
        "verbose": _verbose(),
    }
    try:
        sig = inspect.signature(hook)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        accepted = {}
        params = sig.parameters
        has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
        for key, value in kwargs.items():
            if has_var_kw or key in params:
                accepted[key] = value
        return hook(**accepted)
    try:
        return hook(**kwargs)
    except TypeError:
        return hook()


def _try_import_module(name: str) -> Optional[ModuleType]:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        _remember_error("import %s failed: %s" % (name, exc))
        return None
    selected = _select_backend(mod)
    if selected is None:
        _remember_error("module %s has no flash_attn entry points" % name)
        return None
    return selected


def _import_from_known_modules() -> Optional[ModuleType]:
    for name in _module_names():
        mod = _try_import_module(name)
        if mod is not None:
            return mod
    return None


def _local_module_names(root: pathlib.Path) -> List[str]:
    names: List[str] = []
    for name in _module_names():
        if root.name == name and (root / "__init__.py").is_file():
            names.append(name)
        if (root / name / "__init__.py").is_file():
            names.append(name)
    return list(dict.fromkeys(names))


def _import_local_modules(root: pathlib.Path) -> Optional[ModuleType]:
    for name in _local_module_names(root):
        mod = _try_import_module(name)
        if mod is not None:
            return mod
    return None


def _manifest_paths(root: pathlib.Path) -> List[pathlib.Path]:
    return [root / name for name in _MANIFEST_NAMES if (root / name).is_file()]


def _expand_paths(root: pathlib.Path, items: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in items:
        p = pathlib.Path(raw).expanduser()
        if not p.is_absolute():
            p = root / p
        text = os.fspath(p)
        if any(ch in text for ch in "*?[]"):
            for hit in sorted(glob.glob(text, recursive=True)):
                out.append(os.fspath(pathlib.Path(hit).resolve()))
        else:
            out.append(os.fspath(p.resolve()))
    return list(dict.fromkeys(out))


def _default_build_root(*parts: str) -> str:
    root = os.environ.get("JITTOR_TORCH_EXTENSIONS_DIR")
    if root is None:
        runtime = os.environ.get("JITTOR_TORCH_RUNTIME_ROOT")
        if runtime:
            root = os.path.join(runtime, "torch_extensions")
        else:
            try:
                import jittor as jt
                root = os.path.join(jt.flags.cache_path, "torch_extensions")
            except Exception:
                root = os.path.join(os.path.expanduser("~"), ".cache", "jittor_torch_extensions")
    path = os.path.join(os.path.abspath(os.path.expanduser(root)), *parts)
    os.makedirs(path, exist_ok=True)
    return path


def _manifest_build_dir(root: pathlib.Path, manifest: pathlib.Path, module_name: str) -> str:
    digest_key = os.fspath(root.resolve()) + "|" + os.fspath(manifest.resolve())
    digest = hashlib.sha256(digest_key.encode("utf-8")).hexdigest()[:16]
    safe_name = module_name.replace(".", "_")
    return _default_build_root("flashattn_jittor", safe_name, digest)


def _load_manifest(root: pathlib.Path, manifest: pathlib.Path) -> Optional[ModuleType]:
    try:
        with manifest.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        _remember_error("read manifest %s failed: %s" % (manifest, exc))
        return None

    sources = data.get("sources") or data.get("source_files")
    if not sources:
        _remember_error("manifest %s has no sources" % manifest)
        return None
    if isinstance(sources, str):
        sources = [sources]
    module_name = (
        data.get("module")
        or data.get("name")
        or os.environ.get(_MODULE_ENV)
        or "flashattn_jittor_cuda"
    )
    if not isinstance(module_name, str) or not module_name:
        _remember_error("manifest %s has invalid module name" % manifest)
        return None

    include_items = data.get("include_dirs") or data.get("extra_include_paths") or []
    if isinstance(include_items, str):
        include_items = [include_items]
    include_dirs = []
    for rel in ("include", "csrc", "src"):
        p = root / rel
        if p.is_dir():
            include_dirs.append(os.fspath(p.resolve()))
    include_dirs.extend(_expand_paths(root, include_items))
    include_dirs = list(dict.fromkeys(include_dirs))

    build_dir = data.get("build_directory") or data.get("build_dir")
    if build_dir:
        build_path = pathlib.Path(build_dir).expanduser()
        if not build_path.is_absolute():
            build_path = root / build_path
        build_dir = os.fspath(build_path.resolve())
        os.makedirs(build_dir, exist_ok=True)
    else:
        build_dir = _manifest_build_dir(root, manifest, module_name)

    extra_cflags = list(data.get("extra_cflags") or data.get("cflags") or [])
    extra_cuda_cflags = list(data.get("extra_cuda_cflags") or data.get("cuda_cflags") or [])
    extra_ldflags = list(data.get("extra_ldflags") or data.get("ldflags") or [])

    _log("compile %s from %s" % (module_name, manifest))
    try:
        from jittor.torch_shim.cpp_extension.torch_utils import load

        mod = load(
            name=module_name.split(".")[-1],
            sources=_expand_paths(root, sources),
            extra_include_paths=include_dirs,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=extra_ldflags,
            build_directory=build_dir,
            verbose=_verbose(),
        )
    except Exception as exc:
        _remember_error("compile manifest %s failed: %s" % (manifest, exc))
        return None

    selected = _select_backend(mod)
    if selected is None:
        _remember_error("compiled module %s has no flash_attn entry points" % module_name)
        return None
    return selected


def _load_build_jittor(root: pathlib.Path) -> Optional[ModuleType]:
    path = root / "build_jittor.py"
    if not path.is_file():
        return None
    name = "_jittor_flashattn_build_" + hashlib.sha256(os.fspath(root).encode("utf-8")).hexdigest()[:16]
    try:
        spec = importlib.util.spec_from_file_location(name, os.fspath(path))
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load build_jittor.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    except Exception as exc:
        _remember_error("load %s failed: %s" % (path, exc))
        return None
    selected = _select_backend(mod)
    if selected is None:
        _remember_error("%s has no flashattn_jittor build hook" % path)
    return selected


def _setup_child_env(root: pathlib.Path) -> dict:
    env = os.environ.copy()
    paths: List[str] = []
    runtime = env.get("JITTOR_TORCH_RUNTIME_ROOT")
    if runtime:
        paths.append(os.path.join(runtime, "site-packages"))
    try:
        jittor_python = pathlib.Path(__file__).resolve().parents[2]
        paths.append(os.fspath(jittor_python))
    except Exception:
        pass
    paths.append(os.fspath(root))
    paths.append(os.fspath(root.parent))
    existing = env.get("PYTHONPATH")
    if existing:
        paths.extend(p for p in existing.split(os.pathsep) if p)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    return env


def _build_setup_py(root: pathlib.Path) -> bool:
    if not (root / "setup.py").is_file():
        return False
    try:
        from jittor.torch_shim import bootstrap

        built = bootstrap.build_extension_dirs(
            [os.fspath(root)],
            env=_setup_child_env(root),
            force=_truthy(os.environ.get("JITTOR_FLASH_ATTN_JITTOR_FORCE_BUILD")),
            verbose=_verbose(),
        )
        importlib.invalidate_caches()
        return bool(built) or True
    except Exception as exc:
        _remember_error("build setup.py %s failed: %s" % (root, exc))
        return False


def _load_from_source_root(raw_root: str) -> Optional[ModuleType]:
    root = pathlib.Path(raw_root).expanduser().resolve()
    _add_source_to_sys_path(root)

    for manifest in _manifest_paths(root):
        mod = _load_manifest(root, manifest)
        if mod is not None:
            return mod

    mod = _import_local_modules(root)
    if mod is not None:
        return mod

    mod = _load_build_jittor(root)
    if mod is not None:
        return mod

    if _build_setup_py(root):
        mod = _import_from_known_modules()
        if mod is not None:
            return mod
    return None


def _remember_error(message: str) -> None:
    global _LAST_ERROR
    _LAST_ERROR = message
    _log(message)


def load_backend(force: bool = False) -> Optional[ModuleType]:
    """Return the optional native flashattn_jittor backend module, if available."""
    global _BACKEND, _BACKEND_NAME, _LAST_ERROR, _LOADING
    if not enabled():
        _BACKEND = None
        _BACKEND_NAME = "disabled"
        return None
    if _BACKEND is not _UNSET and not force:
        return _BACKEND
    if _LOADING:
        return None

    _LOADING = True
    _LAST_ERROR = None
    try:
        explicit_roots = explicit_source_roots()
        for root in explicit_roots:
            mod = _load_from_source_root(root)
            if mod is not None:
                _BACKEND = mod
                _BACKEND_NAME = "%s:%s" % (getattr(mod, "__name__", "flashattn_jittor"), root)
                return mod

        mod = _import_from_known_modules()
        if mod is not None:
            _BACKEND = mod
            _BACKEND_NAME = getattr(mod, "__name__", "flashattn_jittor")
            return mod

        for root in candidate_source_roots():
            if root in explicit_roots:
                continue
            mod = _load_from_source_root(root)
            if mod is not None:
                _BACKEND = mod
                _BACKEND_NAME = "%s:%s" % (getattr(mod, "__name__", "flashattn_jittor"), root)
                return mod

        _BACKEND = None
        _BACKEND_NAME = "math"
        if _LAST_ERROR is None:
            _LAST_ERROR = "no flashattn_jittor source or module found"
        else:
            _LAST_ERROR = "no flashattn_jittor source or module found; last error: " + _LAST_ERROR
        return None
    finally:
        _LOADING = False


def is_available() -> bool:
    return load_backend() is not None


def backend_name() -> str:
    if _BACKEND is _UNSET:
        load_backend()
    return _BACKEND_NAME


def last_error() -> Optional[str]:
    if _BACKEND is _UNSET:
        load_backend()
    return _LAST_ERROR
