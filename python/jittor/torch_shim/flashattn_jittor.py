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
import subprocess
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
_READONLY_BORROW_ATTR = "_jittor_torch_ext_readonly_borrow"
_MISSING_ATTR = object()
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
)
_SOURCE_ROOT_NAMES = set(_DEFAULT_MODULE_NAMES + ("flash-attention-jittor", "flash-attention"))

_UNSET = object()
_BACKEND = _UNSET
_BACKEND_NAME = "math"
_LAST_ERROR: Optional[str] = None
_LOADING = False
_BORROW_INPUTS_CACHE = None


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
    if _looks_like_official_flash_attention(root):
        return True
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


def _looks_like_official_flash_attention(root: pathlib.Path) -> bool:
    return (
        (root / "csrc" / "flash_attn" / "flash_api.cpp").is_file()
        and (root / "csrc" / "flash_attn" / "src" / "flash.h").is_file()
    )


def _official_build_dir(root: pathlib.Path) -> str:
    digest_key = os.fspath(root.resolve())
    try:
        head = subprocess.check_output(
            ["git", "-C", os.fspath(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if head:
            digest_key += "|" + head
    except Exception:
        pass
    digest_key += "|head_dims=" + ",".join(_official_head_dims(root))
    digest_key += "|dtypes=" + ",".join(_official_dtypes())
    digest_key += "|missing_forward_stubs=1"
    digest = hashlib.sha256(digest_key.encode("utf-8")).hexdigest()[:16]
    return _default_build_root("flashattn_jittor", "official_flash_attn", digest)


def _ensure_official_cutlass(root: pathlib.Path) -> bool:
    cutlass_h = root / "csrc" / "cutlass" / "include" / "cutlass" / "cutlass.h"
    if cutlass_h.is_file():
        return True
    gitmodules = root / ".gitmodules"
    if (root / ".git").exists() and gitmodules.is_file():
        try:
            _log("initializing official flash-attn CUTLASS submodule")
            subprocess.run(
                ["git", "-C", os.fspath(root), "submodule", "update", "--init", "csrc/cutlass"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:
            _remember_error("initialize official flash-attn CUTLASS failed: %s" % exc)
            return False
    if not cutlass_h.is_file():
        _remember_error("official flash-attn CUTLASS headers missing: %s" % cutlass_h)
        return False
    return True


_OFFICIAL_FLASH_ATTN_HEAD_DIMS = ["32", "64", "96", "128", "192", "256"]
_OFFICIAL_FLASH_ATTN_DTYPES = ["fp16", "bf16"]


def _official_head_dims(root: pathlib.Path) -> List[str]:
    raw = os.environ.get("JITTOR_FLASH_ATTN_HEAD_DIMS") or os.environ.get("FLASH_ATTN_HEAD_DIMS")
    if raw:
        if raw.strip().lower() in ("all", "full", "*"):
            dims = list(_OFFICIAL_FLASH_ATTN_HEAD_DIMS)
        else:
            dims = [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]
    else:
        # TRELLIS.2 uses 128-wide attention heads. Other official kernels are
        # covered by generated runtime stubs unless explicitly requested.
        dims = ["128"]
    src_dir = root / "csrc" / "flash_attn" / "src"
    out = []
    for dim in dims:
        if not dim.isdigit():
            continue
        if (src_dir / ("flash_fwd_hdim%s_fp16_sm80.cu" % dim)).is_file():
            out.append(dim)
    return out or ["128"]


def _official_dtypes() -> List[str]:
    raw = os.environ.get("JITTOR_FLASH_ATTN_DTYPES") or os.environ.get("FLASH_ATTN_DTYPES")
    if raw:
        if raw.strip().lower() in ("all", "full", "*"):
            dtypes = list(_OFFICIAL_FLASH_ATTN_DTYPES)
        else:
            dtypes = [item.strip().lower() for item in raw.replace(";", ",").split(",") if item.strip()]
    else:
        dtypes = list(_OFFICIAL_FLASH_ATTN_DTYPES)
    out = [dt for dt in dtypes if dt in ("fp16", "bf16")]
    return out or ["fp16", "bf16"]


def _official_forward_sources(root: pathlib.Path) -> List[str]:
    src_dir = root / "csrc" / "flash_attn" / "src"
    sources: List[pathlib.Path] = [root / "csrc" / "flash_attn" / "flash_api.cpp"]
    for prefix in ("flash_fwd", "flash_fwd_split"):
        for dim in _official_head_dims(root):
            for dtype in _official_dtypes():
                for causal in ("", "_causal"):
                    path = src_dir / ("%s_hdim%s_%s%s_sm80.cu" % (prefix, dim, dtype, causal))
                    if path.is_file():
                        sources.append(path)
                    else:
                        _remember_error("official flash-attn source missing: %s" % path)
    return [os.fspath(p.resolve()) for p in sources]


def _official_compiled_forward_specs(root: pathlib.Path) -> set:
    src_dir = root / "csrc" / "flash_attn" / "src"
    specs = set()
    for dim in _official_head_dims(root):
        for dtype in _official_dtypes():
            for causal_suffix, causal in (("", False), ("_causal", True)):
                fwd = src_dir / ("flash_fwd_hdim%s_%s%s_sm80.cu" % (dim, dtype, causal_suffix))
                split = src_dir / ("flash_fwd_split_hdim%s_%s%s_sm80.cu" % (dim, dtype, causal_suffix))
                if fwd.is_file():
                    specs.add(("fwd", dtype, dim, causal))
                if split.is_file():
                    specs.add(("split", dtype, dim, causal))
    return specs


def _official_stub_source(build_dir: str, root: pathlib.Path) -> str:
    path = pathlib.Path(build_dir) / "flashattn_jittor_bwd_stubs.cu"
    compiled = _official_compiled_forward_specs(root)
    fwd_lines = []
    split_lines = []
    bwd_lines = []
    for dtype in _OFFICIAL_FLASH_ATTN_DTYPES:
        ctype = "cutlass::half_t" if dtype == "fp16" else "cutlass::bfloat16_t"
        for dim in _OFFICIAL_FLASH_ATTN_HEAD_DIMS:
            for causal in (False, True):
                cbool = "true" if causal else "false"
                if ("fwd", dtype, dim, causal) not in compiled:
                    fwd_lines.append("JT_FLASHATTN_FWD_STUB(%s, %s, %s)" % (ctype, dim, cbool))
                if ("split", dtype, dim, causal) not in compiled:
                    split_lines.append("JT_FLASHATTN_SPLIT_FWD_STUB(%s, %s, %s)" % (ctype, dim, cbool))
                bwd_lines.append("JT_FLASHATTN_BWD_STUB(%s, %s, %s)" % (ctype, dim, cbool))

    body = r'''
#include <stdexcept>
#include <cuda_runtime.h>
#include "namespace_config.h"
#include <cutlass/numeric_types.h>
#include "flash.h"

namespace FLASH_NAMESPACE {
template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_(Flash_fwd_params&, cudaStream_t) {
    throw std::runtime_error("flashattn_jittor official backend was built without this forward kernel; set JITTOR_FLASH_ATTN_HEAD_DIMS=all or include the requested head dimension");
}

template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params&, cudaStream_t) {
    throw std::runtime_error("flashattn_jittor official backend was built without this split forward kernel; set JITTOR_FLASH_ATTN_HEAD_DIMS=all or include the requested head dimension");
}

template<typename T, int Headdim, bool Is_causal>
void run_mha_bwd_(Flash_bwd_params&, cudaStream_t) {
    throw std::runtime_error("flashattn_jittor official backend supports forward inference only; backward is not implemented");
}

#define JT_FLASHATTN_FWD_STUB(DTYPE, HDIM, CAUSAL) \
template void run_mha_fwd_<DTYPE, HDIM, CAUSAL>(Flash_fwd_params&, cudaStream_t);

#define JT_FLASHATTN_SPLIT_FWD_STUB(DTYPE, HDIM, CAUSAL) \
template void run_mha_fwd_splitkv_dispatch<DTYPE, HDIM, CAUSAL>(Flash_fwd_params&, cudaStream_t);

#define JT_FLASHATTN_BWD_STUB(DTYPE, HDIM, CAUSAL) \
template void run_mha_bwd_<DTYPE, HDIM, CAUSAL>(Flash_bwd_params&, cudaStream_t);

%s
%s
%s
#undef JT_FLASHATTN_FWD_STUB
#undef JT_FLASHATTN_SPLIT_FWD_STUB
#undef JT_FLASHATTN_BWD_STUB
} // namespace FLASH_NAMESPACE
''' % ("\n".join(fwd_lines), "\n".join(split_lines), "\n".join(bwd_lines))
    try:
        old = path.read_text(encoding="utf-8")
    except OSError:
        old = None
    if old != body:
        path.write_text(body, encoding="utf-8")
    return os.fspath(path)


def _official_flags() -> Tuple[List[str], List[str]]:
    common = [
        "-O3",
        "-std=c++17",
        "-DFLASHATTENTION_DISABLE_BACKWARD",
        "-DFLASHATTENTION_DISABLE_DROPOUT",
        "-DFLASHATTENTION_DISABLE_ALIBI",
        "-DFLASHATTENTION_DISABLE_SOFTCAP",
    ]
    cuda = [
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-DFLASHATTENTION_DISABLE_BACKWARD",
        "-DFLASHATTENTION_DISABLE_DROPOUT",
        "-DFLASHATTENTION_DISABLE_ALIBI",
        "-DFLASHATTENTION_DISABLE_SOFTCAP",
    ]
    return common, cuda


def _window_size_pair(window_size, window_size_left=-1, window_size_right=-1) -> Tuple[int, int]:
    if window_size is not None:
        try:
            return int(window_size[0]), int(window_size[1])
        except Exception:
            pass
    return int(window_size_left), int(window_size_right)


def _flashattn_result(result, return_attn_probs: bool = False):
    if return_attn_probs:
        return result[0], result[1], result[2]
    return result[0]


def _dtype_name(x) -> str:
    return str(getattr(x, "dtype", ""))


def _native_supported_dtype(x) -> bool:
    return _dtype_name(x) in ("float16", "bfloat16")


def _float32_cast_target():
    raw = (os.environ.get("JITTOR_FLASH_ATTN_CAST_FLOAT32") or "").strip().lower()
    if raw in ("1", "true", "yes", "on", "bf16", "bfloat16"):
        return "bfloat16"
    if raw in ("fp16", "float16", "half"):
        return "float16"
    return None


def _maybe_cast_float32_tensor(x, target: Optional[str]):
    if target and _dtype_name(x) == "float32":
        return x.to(target)
    return x


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


def _mark_readonly_borrow(*tensors):
    if _torch_ext_borrow_inputs_enabled():
        return []
    saved = []
    for tensor in tensors:
        if tensor is None:
            continue
        try:
            old_value = getattr(tensor, _READONLY_BORROW_ATTR)
        except AttributeError:
            old_value = _MISSING_ATTR
        except Exception:
            continue
        try:
            setattr(tensor, _READONLY_BORROW_ATTR, True)
        except Exception:
            continue
        saved.append((tensor, old_value))
    return saved


def _restore_readonly_borrow(saved) -> None:
    for tensor, old_value in reversed(saved):
        try:
            if old_value is _MISSING_ATTR:
                delattr(tensor, _READONLY_BORROW_ATTR)
            else:
                setattr(tensor, _READONLY_BORROW_ATTR, old_value)
        except Exception:
            pass


def _make_official_backend(low_level: ModuleType, root: pathlib.Path) -> ModuleType:
    mod = ModuleType("flashattn_jittor_official")
    mod.__file__ = os.fspath(root)
    mod._flashattn_jittor_official = True
    mod._flashattn_jittor_low_level = low_level
    mod._flashattn_jittor_head_dims = tuple(_official_head_dims(root))
    mod._flashattn_jittor_dtypes = tuple(_official_dtypes())
    low_fwd = low_level.fwd
    low_varlen_fwd = low_level.varlen_fwd

    def _check_args(dropout_p, softcap, alibi_slopes, return_attn_probs):
        if dropout_p not in (0, 0.0, None) and float(dropout_p) != 0.0:
            raise RuntimeError("flashattn_jittor official backend supports dropout_p=0 only")
        if softcap not in (0, 0.0, None) and float(softcap) > 0.0:
            raise RuntimeError("flashattn_jittor official backend does not support softcap")
        if alibi_slopes is not None:
            raise RuntimeError("flashattn_jittor official backend does not support alibi_slopes")
        if return_attn_probs:
            raise RuntimeError("flashattn_jittor official backend does not support return_attn_probs with dropout disabled")

    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                        window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                        deterministic=False, return_attn_probs=False, *args, **kwargs):
        _check_args(dropout_p, softcap, alibi_slopes, return_attn_probs)
        if not (_native_supported_dtype(q) and _native_supported_dtype(k) and _native_supported_dtype(v)):
            target = _float32_cast_target()
            if target and _dtype_name(q) == _dtype_name(k) == _dtype_name(v) == "float32":
                q0 = q
                q = _maybe_cast_float32_tensor(q, target)
                k = _maybe_cast_float32_tensor(k, target)
                v = _maybe_cast_float32_tensor(v, target)
                return flash_attn_func(q, k, v, dropout_p, softmax_scale, causal,
                                       window_size, softcap, alibi_slopes,
                                       deterministic, return_attn_probs,
                                       *args, **kwargs).to(q0.dtype)
            return None
        wl, wr = _window_size_pair(kwargs.get("window_size", window_size))
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        saved = _mark_readonly_borrow(q, k, v, alibi_slopes)
        try:
            result = low_fwd(q, k, v, None, alibi_slopes, 0.0,
                             float(softmax_scale), bool(causal), wl, wr,
                             0.0, bool(return_attn_probs), None)
        finally:
            _restore_readonly_borrow(saved)
        return _flashattn_result(result, return_attn_probs)

    def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None,
                                  causal=False, window_size=(-1, -1), softcap=0.0,
                                  alibi_slopes=None, deterministic=False,
                                  return_attn_probs=False, *args, **kwargs):
        return flash_attn_func(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2],
                               dropout_p, softmax_scale, causal, window_size,
                               softcap, alibi_slopes, deterministic,
                               return_attn_probs, *args, **kwargs)

    def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None,
                                 causal=False, window_size=(-1, -1), softcap=0.0,
                                 alibi_slopes=None, deterministic=False,
                                 return_attn_probs=False, *args, **kwargs):
        return flash_attn_func(q, kv[:, :, 0], kv[:, :, 1],
                               dropout_p, softmax_scale, causal, window_size,
                               softcap, alibi_slopes, deterministic,
                               return_attn_probs, *args, **kwargs)

    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                               max_seqlen_q, max_seqlen_k,
                               dropout_p=0.0, softmax_scale=None, causal=False,
                               window_size=(-1, -1), softcap=0.0,
                               alibi_slopes=None, deterministic=False,
                               return_attn_probs=False, block_table=None,
                               *args, **kwargs):
        _check_args(dropout_p, softcap, alibi_slopes, return_attn_probs)
        if not (_native_supported_dtype(q) and _native_supported_dtype(k) and _native_supported_dtype(v)):
            target = _float32_cast_target()
            if target and _dtype_name(q) == _dtype_name(k) == _dtype_name(v) == "float32":
                q0 = q
                q = _maybe_cast_float32_tensor(q, target)
                k = _maybe_cast_float32_tensor(k, target)
                v = _maybe_cast_float32_tensor(v, target)
                return flash_attn_varlen_func(
                    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    dropout_p, softmax_scale, causal, window_size, softcap,
                    alibi_slopes, deterministic, return_attn_probs, block_table,
                    *args, **kwargs).to(q0.dtype)
            return None
        wl, wr = _window_size_pair(kwargs.get("window_size", window_size))
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        seqused_k = kwargs.get("seqused_k", None)
        leftpad_k = kwargs.get("leftpad_k", None)
        saved = _mark_readonly_borrow(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            seqused_k, leftpad_k, block_table, alibi_slopes,
        )
        try:
            result = low_varlen_fwd(
                q, k, v, None, cu_seqlens_q, cu_seqlens_k,
                seqused_k, leftpad_k,
                block_table, alibi_slopes, int(max_seqlen_q), int(max_seqlen_k),
                0.0, float(softmax_scale), False, bool(causal),
                wl, wr, 0.0, bool(return_attn_probs), None)
        finally:
            _restore_readonly_borrow(saved)
        return _flashattn_result(result, return_attn_probs)

    def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen,
                                         dropout_p=0.0, softmax_scale=None,
                                         causal=False, window_size=(-1, -1),
                                         softcap=0.0, alibi_slopes=None,
                                         deterministic=False,
                                         return_attn_probs=False, *args, **kwargs):
        return flash_attn_varlen_func(qkv[:, 0], qkv[:, 1], qkv[:, 2],
                                      cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                                      dropout_p, softmax_scale, causal, window_size,
                                      softcap, alibi_slopes, deterministic,
                                      return_attn_probs, *args, **kwargs)

    def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k,
                                        max_seqlen_q, max_seqlen_k,
                                        dropout_p=0.0, softmax_scale=None,
                                        causal=False, window_size=(-1, -1),
                                        softcap=0.0, alibi_slopes=None,
                                        deterministic=False,
                                        return_attn_probs=False, *args, **kwargs):
        return flash_attn_varlen_func(q, kv[:, 0], kv[:, 1], cu_seqlens_q,
                                      cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                      dropout_p, softmax_scale, causal, window_size,
                                      softcap, alibi_slopes, deterministic,
                                      return_attn_probs, *args, **kwargs)

    mod.flash_attn_func = flash_attn_func
    mod.flash_attn_qkvpacked_func = flash_attn_qkvpacked_func
    mod.flash_attn_kvpacked_func = flash_attn_kvpacked_func
    mod.flash_attn_varlen_func = flash_attn_varlen_func
    mod.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
    mod.flash_attn_varlen_kvpacked_func = flash_attn_varlen_kvpacked_func
    return mod


def _load_official_flash_attention(root: pathlib.Path) -> Optional[ModuleType]:
    if not _looks_like_official_flash_attention(root):
        return None
    if not _ensure_official_cutlass(root):
        return None
    build_dir = _official_build_dir(root)
    sources = _official_forward_sources(root)
    sources.append(_official_stub_source(build_dir, root))
    include_dirs = [
        os.fspath((root / "csrc" / "flash_attn").resolve()),
        os.fspath((root / "csrc" / "flash_attn" / "src").resolve()),
        os.fspath((root / "csrc" / "cutlass" / "include").resolve()),
    ]
    cflags, cuda_cflags = _official_flags()
    module_name = "flash_attn_2_cuda_jittor"
    _log("compile official flash-attn forward backend from %s" % root)
    try:
        from jittor.torch_shim.cpp_extension.torch_utils import load

        low = load(
            name=module_name,
            sources=sources,
            extra_include_paths=include_dirs,
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_cflags,
            build_directory=build_dir,
            verbose=_verbose(),
            force=_truthy(os.environ.get("JITTOR_FLASH_ATTN_FORCE_BUILD")),
        )
    except Exception as exc:
        _remember_error("compile official flash-attn backend failed: %s" % exc)
        return None
    return _make_official_backend(low, root)


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

    if _looks_like_official_flash_attention(root):
        return _load_official_flash_attention(root)

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
                _LAST_ERROR = None
                return mod

        mod = _import_from_known_modules()
        if mod is not None:
            _BACKEND = mod
            _BACKEND_NAME = getattr(mod, "__name__", "flashattn_jittor")
            _LAST_ERROR = None
            return mod

        for root in candidate_source_roots():
            if root in explicit_roots:
                continue
            mod = _load_from_source_root(root)
            if mod is not None:
                _BACKEND = mod
                _BACKEND_NAME = "%s:%s" % (getattr(mod, "__name__", "flashattn_jittor"), root)
                _LAST_ERROR = None
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
