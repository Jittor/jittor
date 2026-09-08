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
import os
import pathlib
import subprocess
import sys
import threading
from types import ModuleType
from typing import List, Optional, Tuple

from jittor.compat.external_backend import (
    ExternalBackend,
    ExternalBackendSpec,
    register_external_backend,
)
# External diagnostics stay absolute for direct package-spec loading when the
# environment-epoch hook is re-executed. The implementation children below use
# this package's own search path, without relying on its outer parent chain.
from jittor.compat.diagnostics import EXPECTED, swallowed


_TRUTHY = {"1", "true", "yes", "on"}
_FALSEY = {"0", "false", "no", "off"}

_MODULE_ENV = "JITTOR_FLASH_ATTN_JITTOR_MODULE"
_READONLY_BORROW_ATTR = "_jittor_torch_ext_readonly_borrow"
_MISSING_ATTR = object()
_BACKEND_SPEC = ExternalBackendSpec(
    name="flash-attn",
    source_envs=(
        "JITTOR_FLASH_ATTN_JITTOR_SRC",
        "FLASHATTN_JITTOR_SRC",
        "FLASH_ATTN_JITTOR_SRC",
        "FLASHATTNJITTOR_SRC",
    ),
    module_env=_MODULE_ENV,
    module_names=(
        "flashattn_jittor",
        "flash_attn_jittor",
        "flashattnjittor",
        "flashattn_jittor_cuda",
        "flash_attn_jittor_cuda",
    ),
    public_functions=(
        "flash_attn_func",
        "flash_attn_qkvpacked_func",
        "flash_attn_kvpacked_func",
        "flash_attn_varlen_func",
        "flash_attn_varlen_qkvpacked_func",
        "flash_attn_varlen_kvpacked_func",
    ),
    hook_names=(
        "load_jittor_flash_attn",
        "build_jittor_flash_attn",
        "load_flashattn_jittor",
        "build_flashattn_jittor",
    ),
    manifest_names=(
        "flashattn_jittor.json",
        "flash_attn_jittor.json",
        "jittor_flashattn.json",
    ),
    relative_source_dirs=(
        "flashattn_jittor",
        "flash_attn_jittor",
        "flashattnjittor",
        "flash-attention-jittor",
        "flash-attention",
        "third_party/flashattn_jittor",
        "third_party/flash_attn_jittor",
        "third_party/flash-attention-jittor",
        "third_party/flash-attention",
        "extern/flashattn_jittor",
        "extern/flash_attn_jittor",
        "extern/flash-attention",
        "extensions/flashattn_jittor",
        "extensions/flash_attn_jittor",
        "extensions/flash-attention",
    ),
    source_root_names=("flash-attention-jittor", "flash-attention"),
    project_root_envs=(
        "JITTOR_FLASH_ATTN_JITTOR_PROJECT_ROOT",
        "JITTOR_TORCH_PROJECT_ROOT",
    ),
    submodule_attrs=("_C", "cuda", "ops", "flashattn_jittor_cuda", "flash_attn_jittor_cuda"),
    environment_names=(
        "JITTOR_FLASH_ATTN_JITTOR",
        "JITTOR_FLASHATTN_JITTOR",
        "JITTOR_FLASH_ATTN_JITTOR_PROJECT_ROOT",
        "JITTOR_TORCH_PROJECT_ROOT",
        "JITTOR_FLASH_ATTN_HEAD_DIMS",
        "FLASH_ATTN_HEAD_DIMS",
        "JITTOR_FLASH_ATTN_DTYPES",
        "FLASH_ATTN_DTYPES",
        "JITTOR_FLASH_ATTN_FORCE_BUILD",
        "JITTOR_FLASH_ATTN_JITTOR_FORCE_BUILD",
        "JITTOR_FLASH_ATTN_DIRECT_ADAPTER",
        "JITTOR_FLASH_ATTN_DIRECT_PACKED",
        "JITTOR_FLASH_ATTN_FUSED_PACKED_SPLIT",
        "JITTOR_HOME",
        "JTCUDA",
        "CUDA_HOME",
        "nvcc_path",
        "JITTOR_TORCH_RUNTIME_ROOT",
        "JITTOR_TORCH_EXTENSIONS_DIR",
        "TORCH_EXTENSIONS_DIR",
        "TORCH_CUDA_ARCH_LIST",
        "CC",
        "CXX",
    ),
    default_module_name="flashattn_jittor_cuda",
    build_namespace="flashattn_jittor",
    force_build_env="JITTOR_FLASH_ATTN_JITTOR_FORCE_BUILD",
    source_predicates=(lambda root: _looks_like_official_flash_attention(root),),
)
_EXTERNAL_BACKEND = register_external_backend(
    ExternalBackend(
        _BACKEND_SPEC,
        log=lambda message: _remember_error(message),
        verbose=lambda: _verbose(),
        build_root=lambda *parts: _default_build_root("flashattn_jittor", *parts),
        setup_builder=lambda root: _build_setup_backend(root),
        special_source_loader=lambda root: _load_official_flash_attention(root),
    )
)
_SRC_ENVS = _BACKEND_SPEC.source_envs
_MANIFEST_NAMES = _BACKEND_SPEC.manifest_names
_DEFAULT_MODULE_NAMES = _BACKEND_SPEC.module_names
_PUBLIC_FUNCS = _BACKEND_SPEC.public_functions
_HOOK_NAMES = _BACKEND_SPEC.hook_names
_SUBMODULE_ATTRS = _BACKEND_SPEC.submodule_attrs
_RELATIVE_SOURCE_DIRS = _BACKEND_SPEC.relative_source_dirs
_SOURCE_ROOT_NAMES = set(_BACKEND_SPEC.source_root_names + _BACKEND_SPEC.module_names)

_UNSET = object()
_BACKEND = _UNSET
_BACKEND_NAME = "math"
_BACKEND_CONFIG_KEY = None
_BACKEND_LOAD_GENERATION = 0
_BACKEND_PUBLICATION_TOKEN = None
_LAST_ERROR: Optional[str] = None
_LOADING = False
_BACKEND_LOAD_LOCK = threading.RLock()
_BORROW_INPUTS_CACHE = None
_PACKED_SPLIT_STATS = {
    "qkv_cuda": 0,
    "kv_cuda": 0,
    "fallback": 0,
    "error": 0,
}


from .official_codegen import (
    _official_stub_source,
    _official_packed_source,
)

from .official_build import (
    _looks_like_official_flash_attention,
    _official_build_dir,
    _official_packed_build_dir,
    _official_import_identity,
    _ensure_official_cutlass,
    _official_dropout_backward_supported,
    _official_head_dims,
    _official_dtypes,
    _official_sources,
    _official_compiled_specs,
    _official_flags,
    _load_official_packed_flash_attention,
    _load_official_flash_attention,
)

from .adapter import (
    _window_size_pair,
    _flashattn_result,
    _dtype_name,
    _native_supported_dtype,
    _float32_cast_target,
    _maybe_cast_float32_tensor,
    _mark_readonly_borrow,
    _restore_readonly_borrow,
    _make_official_backend,
)

from .packed import (
    _packed_split_enabled,
    _is_cuda_jittor_var,
    _split_qkvpacked_cuda,
    _split_kvpacked_cuda,
    _direct_packed_enabled,
)

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
    return _EXTERNAL_BACKEND.module_names()


def _project_roots() -> List[pathlib.Path]:
    return _EXTERNAL_BACKEND.project_roots()


def candidate_source_roots() -> List[str]:
    return _EXTERNAL_BACKEND.source_roots()


def explicit_source_roots() -> List[str]:
    return _EXTERNAL_BACKEND.source_roots(explicit_only=True)


def _looks_like_source_root(root: pathlib.Path, explicit: bool = False) -> bool:
    return _EXTERNAL_BACKEND.looks_like_source_root(root, explicit=explicit)


def _has_public_api(mod: object) -> bool:
    return _EXTERNAL_BACKEND.has_public_api(mod)


def _select_backend(mod: ModuleType, allow_hooks: bool = True) -> Optional[ModuleType]:
    return _EXTERNAL_BACKEND.select_backend(mod, allow_hooks=allow_hooks)


def _import_from_known_modules() -> Optional[ModuleType]:
    return _EXTERNAL_BACKEND.import_installed()


def _import_local_modules(root: pathlib.Path) -> Optional[ModuleType]:
    return _EXTERNAL_BACKEND.import_local(root)


def _manifest_paths(root: pathlib.Path) -> List[pathlib.Path]:
    return _EXTERNAL_BACKEND.manifest_paths(root)


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
            except EXPECTED as exc:
                swallowed("shim/backends/flash_attention.py _default_build_root: import jittor as jt", exc)
                root = os.path.join(os.path.expanduser("~"), ".cache", "jittor_torch_extensions")
    path = os.path.join(os.path.abspath(os.path.expanduser(root)), *parts)
    os.makedirs(path, exist_ok=True)
    return path


_OFFICIAL_FLASH_ATTN_HEAD_DIMS = ["32", "64", "96", "128", "192", "256"]
_OFFICIAL_FLASH_ATTN_DTYPES = ["fp16", "bf16"]


def _torch_ext_borrow_inputs_enabled() -> bool:
    """Mirror the C++ extension borrow-input gate.

    flash-attn marks q/k/v as readonly-borrow itself after explicit
    materialization.  This helper only detects the extension-wide borrow mode so
    we can skip redundant Python tagging in that unsafe opt-in configuration.
    """
    global _BORROW_INPUTS_CACHE
    state = (
        os.environ.get("JITTOR_TORCH_EXT_SYNC_BOUNDARY"),
        os.environ.get("JITTOR_TORCH_EXT_COPY_INPUTS"),
        os.environ.get("JITTOR_TORCH_EXT_UNSAFE_BORROW_INPUTS"),
        os.environ.get("JITTOR_TORCH_EXT_BORROW_INPUTS"),
    )
    if _BORROW_INPUTS_CACHE is not None and _BORROW_INPUTS_CACHE[0] == state:
        return _BORROW_INPUTS_CACHE[1]
    enabled = not (
        _truthy(state[0]) or _truthy(state[1])
        or _falsey(state[2]) or _falsey(state[3])
    ) and (_truthy(state[2]) or _truthy(state[3]))
    _BORROW_INPUTS_CACHE = (state, enabled)
    return enabled


def _load_manifest(root: pathlib.Path, manifest: pathlib.Path) -> Optional[ModuleType]:
    return _EXTERNAL_BACKEND.load_manifest(root, manifest)


def _load_build_jittor(root: pathlib.Path) -> Optional[ModuleType]:
    return _EXTERNAL_BACKEND.load_build_script(root)


def _setup_child_env(root: pathlib.Path) -> dict:
    env = os.environ.copy()
    paths: List[str] = []
    runtime = env.get("JITTOR_TORCH_RUNTIME_ROOT")
    if runtime:
        paths.append(os.path.join(runtime, "site-packages"))
    try:
        import jittor as jt

        jittor_python = pathlib.Path(jt.__file__).resolve().parents[1]
        paths.append(os.fspath(jittor_python))
    except OSError as exc:
        swallowed("shim/backends/flash_attention.py _setup_child_env: import jittor as jt", exc)
    paths.append(os.fspath(root))
    paths.append(os.fspath(root.parent))
    existing = env.get("PYTHONPATH")
    if existing:
        paths.extend(p for p in existing.split(os.pathsep) if p)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    return env


def _build_setup_backend(root: pathlib.Path) -> bool:
    if not (root / "setup.py").is_file():
        return False
    try:
        from jittor.compat.shim import bootstrap

        built = bootstrap.build_extension_dirs(
            [os.fspath(root)],
            env=_setup_child_env(root),
            force=_truthy(os.environ.get("JITTOR_FLASH_ATTN_JITTOR_FORCE_BUILD")),
            verbose=_verbose(),
        )
        importlib.invalidate_caches()
        return bool(built) or True
    except EXPECTED as exc:
        swallowed("shim/backends/flash_attention.py _build_setup_backend: from jittor.compat.shim import bootstrap", exc)
        _remember_error("build setup.py %s failed: %s" % (root, exc))
        return False


def _build_setup_py(root: pathlib.Path) -> bool:
    return _EXTERNAL_BACKEND.build_setup(root)


def _load_from_source_root(raw_root: str) -> Optional[ModuleType]:
    return _EXTERNAL_BACKEND.load_source_root(raw_root)


def _remember_error(message: str) -> None:
    global _LAST_ERROR
    _LAST_ERROR = message
    _log(message)


_BACKEND_ENV_NAMES = tuple(
    dict.fromkeys(_SRC_ENVS + (_MODULE_ENV,) + _BACKEND_SPEC.environment_names)
)


_BACKEND_ENV_EPOCH_STATE_ATTR = "_jittor_flashattn_backend_env_epoch_state_v1"
_BACKEND_ENV_EPOCH_PROBE = "jittor.flashattn.backend_env_epoch_probe"
_BACKEND_MODULE_STATE_ATTR = "_jittor_flashattn_backend_module_state_v1"


def _install_backend_environment_epoch_hook():
    """Install one process-wide watcher for backend-related environment writes."""
    names = frozenset(
        dict.fromkeys(_BACKEND_ENV_NAMES + _EXTERNAL_BACKEND.environment_names())
    )
    byte_names = frozenset(os.fsencode(name) for name in names)
    state = getattr(sys, _BACKEND_ENV_EPOCH_STATE_ATTR, None)
    if isinstance(state, dict) and state.get("version") == 1:
        if state.get("names") != names or state.get("byte_names") != byte_names:
            state["names"] = names
            state["byte_names"] = byte_names
            state["epoch"] += 1
        return state if state.get("active") and state.get("reliable") else None

    state = {
        "version": 1,
        "epoch": 0,
        "names": names,
        "byte_names": byte_names,
        "active": False,
        "reliable": True,
    }
    setattr(sys, _BACKEND_ENV_EPOCH_STATE_ATTR, state)

    def audit_hook(event, args):
        try:
            if event == _BACKEND_ENV_EPOCH_PROBE:
                state["active"] = True
                return
            if event not in ("os.putenv", "os.unsetenv") or not args:
                return
            name = args[0]
            if ((isinstance(name, bytes) and name in state["byte_names"])
                    or (isinstance(name, str) and name in state["names"])):
                state["epoch"] += 1
        except EXPECTED as exc:
            # An audit hook exception would abort the environment write. Mark
            # the token unusable and leave the write itself untouched.
            swallowed("shim/backends/flash_attention.py audit_hook: if event == _BACKEND_ENV_EPOCH_PROBE:", exc)
            state["reliable"] = False

    # Audit hooks cannot be removed. Keep the state on sys so module reloads
    # reuse this hook rather than installing duplicate watchers.
    state["hook"] = audit_hook
    try:
        sys.addaudithook(audit_hook)
        sys.audit(_BACKEND_ENV_EPOCH_PROBE)
    except EXPECTED as exc:
        swallowed("shim/backends/flash_attention.py _install_backend_environment_epoch_hook: sys.addaudithook(audit_hook)", exc)
    return state if state["active"] else None


_BACKEND_ENV_EPOCH_STATE = _install_backend_environment_epoch_hook()


def _next_backend_module_incarnation() -> int:
    state = getattr(sys, _BACKEND_MODULE_STATE_ATTR, None)
    if not isinstance(state, dict) or state.get("version") != 1:
        state = {"version": 1, "incarnation": 0}
        setattr(sys, _BACKEND_MODULE_STATE_ATTR, state)
    state["incarnation"] += 1
    return int(state["incarnation"])


_BACKEND_MODULE_INCARNATION = _next_backend_module_incarnation()


def backend_environment_epoch() -> Optional[int]:
    """Return a cheap invalidation token, or None when audit hooks are unavailable."""
    state = _install_backend_environment_epoch_hook()
    if state is None or not state.get("reliable"):
        return None
    return int(state["epoch"])


def invalidate_backend_environment() -> None:
    """Invalidate cached backend selection after a non-os.environ config change."""
    if _BACKEND_ENV_EPOCH_STATE is not None:
        _BACKEND_ENV_EPOCH_STATE["epoch"] += 1


def backend_cache_token() -> Optional[Tuple[int, int, int]]:
    """Return the process-local backend identity used by inference fast paths."""
    epoch = backend_environment_epoch()
    if epoch is None:
        return None
    return (_BACKEND_MODULE_INCARNATION, _BACKEND_LOAD_GENERATION, epoch)


def backend_publication_token(backend: Optional[ModuleType]) -> Optional[Tuple[int, int, int]]:
    """Return the token under which *backend* was published by this loader."""
    if backend is None or backend is not _BACKEND:
        return None
    return _BACKEND_PUBLICATION_TOKEN


def _backend_environment_key() -> Tuple[Tuple[str, Optional[str]], ...]:
    return tuple(
        dict.fromkeys(
            tuple((name, os.environ.get(name)) for name in _BACKEND_ENV_NAMES)
            + _EXTERNAL_BACKEND.environment_key()
        )
    )


def _stable_backend_environment_key():
    """Capture an environment snapshot and its matching audit epoch."""
    for _ in range(8):
        epoch_before = backend_environment_epoch()
        key = _backend_environment_key()
        epoch_after = backend_environment_epoch()
        if epoch_before == epoch_after:
            return key, epoch_after
    return key, None


def _backend_config_key() -> Tuple[object, ...]:
    return (
        _backend_environment_key(),
        tuple(candidate_source_roots()),
    )


def backend_capability_miss(backend: Optional[ModuleType], head_dim: int,
                            dtype: str) -> Optional[str]:
    if backend is None:
        return "no_backend"
    if not getattr(backend, "_flashattn_jittor_official", False):
        return None
    dims = {int(x) for x in getattr(backend, "_flashattn_jittor_head_dims", ())}
    dtypes = set(getattr(backend, "_flashattn_jittor_dtypes", ()))
    if dims and int(head_dim) not in dims:
        return "backend_head_dim"
    expected_dtype = {"float16": "fp16", "bfloat16": "bf16"}.get(str(dtype))
    if dtypes and expected_dtype is not None and expected_dtype not in dtypes:
        return "backend_dtype"
    return None


def _merge_capability_env_list(primary: str, fallback: str, item: object) -> None:
    raw = os.environ.get(primary) or os.environ.get(fallback)
    if not raw:
        os.environ[primary] = str(item)
        return
    if raw.strip().lower() in ("all", "full", "*"):
        return
    items = [part.strip() for part in raw.replace(";", ",").split(",")
             if part.strip()]
    if str(item) not in items:
        items.append(str(item))
        os.environ[primary] = ",".join(items)


def _ensure_capability_compile_env(head_dim: int, dtype: str) -> None:
    _merge_capability_env_list(
        "JITTOR_FLASH_ATTN_HEAD_DIMS", "FLASH_ATTN_HEAD_DIMS", int(head_dim))
    dtype_name = str(dtype).strip().lower()
    if dtype_name in ("float16", "fp16", "half"):
        compile_dtype = "fp16"
    elif dtype_name in ("bfloat16", "bf16"):
        compile_dtype = "bf16"
    else:
        return
    _merge_capability_env_list(
        "JITTOR_FLASH_ATTN_DTYPES", "FLASH_ATTN_DTYPES", compile_dtype)


def load_backend_for(head_dim: int, dtype: str) -> Tuple[Optional[ModuleType], Optional[str]]:
    """Load a backend containing the requested official kernel capability."""
    # Capability env, build digest, source selection, module metadata and cache
    # key all consume the same process-global environment. Keep the entire
    # transaction under the loader lock so concurrent first-use requests cannot
    # publish a partially expanded or internally inconsistent backend.
    with _BACKEND_LOAD_LOCK:
        _ensure_capability_compile_env(head_dim, dtype)
        backend = load_backend()
        miss = backend_capability_miss(backend, head_dim, dtype)
        if miss in ("backend_head_dim", "backend_dtype"):
            # Official build directories include dims/dtypes in their digest,
            # so a forced reload incrementally builds the expanded module.
            backend = load_backend(force=True)
            miss = backend_capability_miss(backend, head_dim, dtype)
        return backend, miss


def load_backend(force: bool = False) -> Optional[ModuleType]:
    # Extension compilation and sys.modules replacement are process-global.
    # Other threads wait for the first loader; same-thread recursive hooks use
    # the RLock and retain the existing _LOADING recursion guard below.
    with _BACKEND_LOAD_LOCK:
        return _load_backend_locked(force)


def _load_backend_locked(force: bool = False) -> Optional[ModuleType]:
    """Return the optional native flashattn_jittor backend module, if available."""
    global _BACKEND, _BACKEND_NAME, _BACKEND_CONFIG_KEY
    global _BACKEND_LOAD_GENERATION, _BACKEND_PUBLICATION_TOKEN
    global _LAST_ERROR, _LOADING
    if not enabled():
        _BACKEND = None
        _BACKEND_NAME = "disabled"
        # Do not bind a miss to a snapshot captured after the enabled check;
        # another thread may have re-enabled the backend in between.
        _BACKEND_CONFIG_KEY = None
        _BACKEND_PUBLICATION_TOKEN = None
        return None
    if _BACKEND is not _UNSET and not force:
        cached_env_key = (
            _BACKEND_CONFIG_KEY[0] if _BACKEND_CONFIG_KEY is not None else None
        )
        environment_key, environment_epoch = _stable_backend_environment_key()
        if environment_key == cached_env_key:
            if _BACKEND is not None:
                if environment_epoch is not None:
                    _BACKEND_PUBLICATION_TOKEN = (
                        _BACKEND_MODULE_INCARNATION,
                        _BACKEND_LOAD_GENERATION,
                        environment_epoch,
                    )
                return _BACKEND
            # A failed lookup also tracks auto-discovered source roots, so a
            # source tree appearing later in the process invalidates the miss.
            if _backend_config_key() == _BACKEND_CONFIG_KEY:
                return None
        force = True
    if _LOADING:
        return None

    _LOADING = True
    _BACKEND_LOAD_GENERATION += 1
    load_environment_epoch = backend_environment_epoch()
    _BACKEND_PUBLICATION_TOKEN = None
    _LAST_ERROR = None
    load_completed = False
    try:
        explicit_roots = explicit_source_roots()
        for root in explicit_roots:
            mod = _load_from_source_root(root)
            if mod is not None:
                _BACKEND = mod
                _BACKEND_NAME = "%s:%s" % (getattr(mod, "__name__", "flashattn_jittor"), root)
                _LAST_ERROR = None
                load_completed = True
                return mod

        mod = _import_from_known_modules()
        if mod is not None:
            _BACKEND = mod
            _BACKEND_NAME = getattr(mod, "__name__", "flashattn_jittor")
            _LAST_ERROR = None
            load_completed = True
            return mod

        for root in candidate_source_roots():
            if root in explicit_roots:
                continue
            mod = _load_from_source_root(root)
            if mod is not None:
                _BACKEND = mod
                _BACKEND_NAME = "%s:%s" % (getattr(mod, "__name__", "flashattn_jittor"), root)
                _LAST_ERROR = None
                load_completed = True
                return mod

        _BACKEND = None
        _BACKEND_NAME = "math"
        if _LAST_ERROR is None:
            _LAST_ERROR = "no flashattn_jittor source or module found"
        else:
            _LAST_ERROR = "no flashattn_jittor source or module found; last error: " + _LAST_ERROR
        load_completed = True
        return None
    finally:
        try:
            if not load_completed:
                _BACKEND_CONFIG_KEY = None
                _BACKEND_PUBLICATION_TOKEN = None
            else:
                config_key = _backend_config_key()
                _, publication_epoch = _stable_backend_environment_key()
            if load_completed and (load_environment_epoch is not None
                    and publication_epoch != load_environment_epoch):
                # A concurrent backend configuration write raced the build. Do
                # not associate the module with config it may not have consumed.
                _BACKEND_CONFIG_KEY = None
                _BACKEND_PUBLICATION_TOKEN = None
            elif load_completed:
                _BACKEND_CONFIG_KEY = config_key
                if _BACKEND is not None and publication_epoch is not None:
                    _BACKEND_PUBLICATION_TOKEN = (
                        _BACKEND_MODULE_INCARNATION,
                        _BACKEND_LOAD_GENERATION,
                        publication_epoch,
                    )
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
