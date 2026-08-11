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


def _official_packed_build_dir(root: pathlib.Path) -> str:
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
    digest_key += "|direct_packed_forward=6"
    digest = hashlib.sha256(digest_key.encode("utf-8")).hexdigest()[:16]
    return _default_build_root("flashattn_jittor", "official_flash_attn_packed", digest)


def _official_import_identity(kind: str, build_dir: str, module_name: str,
                              generation: Optional[int] = None) -> str:
    """Identify one official build without changing its extension name."""
    if generation is None:
        generation = _BACKEND_LOAD_GENERATION
    build_digest = pathlib.Path(build_dir).name
    safe_kind = "".join(ch if ch.isalnum() else "_" for ch in kind)
    namespace = "_jittor_flash_%s_%s_g%d" % (
        safe_kind, build_digest, int(generation))
    return namespace + "." + module_name


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
        # Keep the common 128-wide kernel as the default. Other official kernels
        # are covered by generated runtime stubs unless explicitly requested.
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


def _official_packed_source(build_dir: str) -> str:
    path = pathlib.Path(build_dir) / "flashattn_jittor_packed_fwd.cu"
    body = r'''
#include <cmath>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "hardware_info.h"
#include "flash.h"
#include "static_switch.h"

namespace jtorch { namespace detail {
void data_ptrs(std::initializer_list<jtorch::Tensor> tensors, void** out);
}} // namespace jtorch::detail

namespace FLASH_NAMESPACE {

static at::Tensor jt_readonly_tensor(py::handle obj, const char *name) {
    TORCH_CHECK(!obj.is_none(), name, " must be a Jittor Var");
    TORCH_CHECK(::jtorch::detail::is_jittor_var(obj.ptr()), name, " must be a Jittor Var");
    return ::jtorch::detail::tensor_from_pyvar_readonly(obj.ptr());
}

static inline int jt_round_multiple(int x, int m) {
    return (x + m - 1) / m * m;
}

static void jt_run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
            });
        });
    });
}

static void jt_check_cuda(const at::Tensor &x, const char *name) {
    TORCH_CHECK(x.is_cuda(), name, " must be on CUDA");
}

static at::TensorOptions jt_cuda_options(at::ScalarType dtype) {
    return torch::dtype(dtype).device(torch::kCUDA);
}

static void jt_fill_params(
        Flash_fwd_params &params,
        at::ScalarType dtype,
        int batch_size,
        int seqlen_q,
        int seqlen_k,
        int num_heads,
        int num_heads_k,
        int head_size,
        void *q_ptr,
        void *k_ptr,
        void *v_ptr,
        int64_t q_batch_stride,
        int64_t k_batch_stride,
        int64_t v_batch_stride,
        int64_t q_row_stride,
        int64_t k_row_stride,
        int64_t v_row_stride,
        int64_t q_head_stride,
        int64_t k_head_stride,
        int64_t v_head_stride,
        at::Tensor &out,
        void *out_ptr,
        void *cu_seqlens_q,
        void *cu_seqlens_k,
        at::Tensor &softmax_lse,
        void *softmax_lse_ptr,
        float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        bool unpadded_lse) {
    params = {};
    params.is_bf16 = dtype == torch::kBFloat16;
    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_ptr = out_ptr;
    params.o_batch_stride = out.stride(0);
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);
    params.p_ptr = nullptr;
    params.softmax_lse_ptr = softmax_lse_ptr;
    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_heads_k;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = jt_round_multiple(seqlen_q, 128);
    params.seqlen_k_rounded = jt_round_multiple(seqlen_k, 128);
    params.d = head_size;
    params.d_rounded = jt_round_multiple(head_size, head_size <= 128 ? 32 : 64);
    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k);
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    params.p_dropout = 1.f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.f;
    params.scale_softmax_rp_dropout = softmax_scale;
    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }
    if (is_causal) { window_size_right = 0; }
    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.is_causal = window_size_left < 0 && window_size_right == 0;
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;
    params.is_seqlens_k_cumulative = true;
    params.unpadded_lse = unpadded_lse;
    params.num_splits = 1;
    params.total_q = unpadded_lse ? out.size(0) : 0;
}

static void jt_finish_and_run(Flash_fwd_params &params, void *rng_state_ptr) {
    params.rng_state = reinterpret_cast<uint64_t *>(rng_state_ptr);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    jt_run_mha_fwd(params, stream);
}

at::Tensor
jt_fwd(py::handle q_obj,
       py::handle k_obj,
       py::handle v_obj,
       float softmax_scale,
       bool is_causal,
       int window_size_left,
       int window_size_right) {
    auto q = jt_readonly_tensor(q_obj, "q");
    auto k = jt_readonly_tensor(k_obj, "k");
    auto v = jt_readonly_tensor(v_obj, "v");
    at::cuda::CUDAGuard device_guard{0};
    TORCH_CHECK(q.dim() == 4, "q must be [batch, seqlen_q, heads, dim]");
    TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "k/v must be [batch, seqlen_k, heads, dim]");
    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16,
                "q must be fp16 or bf16");
    TORCH_CHECK(k.dtype() == q.dtype(), "k dtype mismatch");
    TORCH_CHECK(v.dtype() == q.dtype(), "v dtype mismatch");
    const int batch_size = q.size(0);
    const int seqlen_q = q.size(1);
    const int seqlen_k = k.size(1);
    const int num_heads = q.size(2);
    const int num_heads_k = k.size(2);
    const int head_size = q.size(3);
    TORCH_CHECK(batch_size == k.size(0) && batch_size == v.size(0), "q/k/v batch mismatch");
    TORCH_CHECK(k.size(1) == v.size(1), "k/v seqlen mismatch");
    TORCH_CHECK(k.size(2) == v.size(2), "k/v heads mismatch");
    TORCH_CHECK(head_size == k.size(3) && head_size == v.size(3), "q/k/v head dim mismatch");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide query heads");
    auto opts = jt_cuda_options(q.dtype());
    auto out = torch::empty({batch_size, seqlen_q, num_heads, head_size}, opts);
    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(torch::kFloat));
    auto rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
    void *ptrs[6];
    ::jtorch::detail::data_ptrs({q, k, v, out, softmax_lse, rng_state}, ptrs);
    jt_check_cuda(q, "q");
    jt_check_cuda(k, "k");
    jt_check_cuda(v, "v");
    Flash_fwd_params params;
    jt_fill_params(params, q.dtype(), batch_size, seqlen_q, seqlen_k,
                   num_heads, num_heads_k, head_size,
                   ptrs[0], ptrs[1], ptrs[2],
                   q.stride(0), k.stride(0), v.stride(0),
                   q.stride(1), k.stride(1), v.stride(1),
                   q.stride(2), k.stride(2), v.stride(2),
                   out, ptrs[3], nullptr, nullptr, softmax_lse, ptrs[4], softmax_scale,
                   is_causal, window_size_left, window_size_right, false);
    jt_finish_and_run(params, ptrs[5]);
    return out;
}

at::Tensor
jt_varlen_fwd(py::handle q_obj,
              py::handle k_obj,
              py::handle v_obj,
              py::handle cu_seqlens_q_obj,
              py::handle cu_seqlens_k_obj,
              int max_seqlen_q,
              int max_seqlen_k,
              float softmax_scale,
              bool is_causal,
              int window_size_left,
              int window_size_right) {
    auto q = jt_readonly_tensor(q_obj, "q");
    auto k = jt_readonly_tensor(k_obj, "k");
    auto v = jt_readonly_tensor(v_obj, "v");
    auto cu_seqlens_q = jt_readonly_tensor(cu_seqlens_q_obj, "cu_seqlens_q");
    auto cu_seqlens_k = jt_readonly_tensor(cu_seqlens_k_obj, "cu_seqlens_k");
    at::cuda::CUDAGuard device_guard{0};
    TORCH_CHECK(q.dim() == 3, "q must be [total_q, heads, dim]");
    TORCH_CHECK(k.dim() == 3 && v.dim() == 3, "k/v must be [total_k, heads, dim]");
    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16,
                "q must be fp16 or bf16");
    TORCH_CHECK(k.dtype() == q.dtype(), "k dtype mismatch");
    TORCH_CHECK(v.dtype() == q.dtype(), "v dtype mismatch");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32 && cu_seqlens_k.dtype() == torch::kInt32,
                "cu_seqlens tensors must be int32");
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int total_q = q.size(0);
    const int num_heads = q.size(1);
    const int num_heads_k = k.size(1);
    const int head_size = q.size(2);
    TORCH_CHECK(cu_seqlens_k.numel() == cu_seqlens_q.numel(), "cu_seqlens batch mismatch");
    TORCH_CHECK(k.size(0) == v.size(0), "k/v total length mismatch");
    TORCH_CHECK(k.size(1) == v.size(1), "k/v heads mismatch");
    TORCH_CHECK(head_size == k.size(2) && head_size == v.size(2), "q/k/v head dim mismatch");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide query heads");
    auto opts = jt_cuda_options(q.dtype());
    auto out = torch::empty({total_q, num_heads, head_size}, opts);
    auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(torch::kFloat));
    auto rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
    void *ptrs[8];
    ::jtorch::detail::data_ptrs({q, k, v, cu_seqlens_q, cu_seqlens_k,
                                 out, softmax_lse, rng_state}, ptrs);
    jt_check_cuda(q, "q");
    jt_check_cuda(k, "k");
    jt_check_cuda(v, "v");
    jt_check_cuda(cu_seqlens_q, "cu_seqlens_q");
    jt_check_cuda(cu_seqlens_k, "cu_seqlens_k");
    Flash_fwd_params params;
    jt_fill_params(params, q.dtype(), batch_size, max_seqlen_q, max_seqlen_k,
                   num_heads, num_heads_k, head_size,
                   ptrs[0], ptrs[1], ptrs[2],
                   0, 0, 0,
                   q.stride(0), k.stride(0), v.stride(0),
                   q.stride(1), k.stride(1), v.stride(1),
                   out, ptrs[5],
                   ptrs[3],
                   ptrs[4],
                   softmax_lse, ptrs[6], softmax_scale, is_causal,
                   window_size_left, window_size_right, true);
    jt_finish_and_run(params, ptrs[7]);
    return out;
}

at::Tensor
jt_varlen_qkvpacked_fwd(py::handle qkv_obj,
                        py::handle cu_seqlens_obj,
                        int max_seqlen,
                        float softmax_scale,
                        bool is_causal,
                        int window_size_left,
                        int window_size_right) {
    auto qkv = jt_readonly_tensor(qkv_obj, "qkv");
    auto cu_seqlens = jt_readonly_tensor(cu_seqlens_obj, "cu_seqlens");
    at::cuda::CUDAGuard device_guard{0};
    TORCH_CHECK(qkv.dim() == 4, "qkv must be [total, 3, heads, dim]");
    TORCH_CHECK(qkv.size(1) == 3, "qkv packed dimension must be 3");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must be int32");
    TORCH_CHECK(qkv.dtype() == torch::kFloat16 || qkv.dtype() == torch::kBFloat16,
                "qkv must be fp16 or bf16");
    const int batch_size = cu_seqlens.numel() - 1;
    const int total_q = qkv.size(0);
    const int num_heads = qkv.size(2);
    const int head_size = qkv.size(3);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256 && head_size % 8 == 0, "unsupported head size");
    auto opts = jt_cuda_options(qkv.dtype());
    auto out = torch::empty({total_q, num_heads, head_size}, opts);
    auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(torch::kFloat));
    auto rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
    void *ptrs[5];
    ::jtorch::detail::data_ptrs({qkv, cu_seqlens, out, softmax_lse, rng_state}, ptrs);
    jt_check_cuda(qkv, "qkv");
    jt_check_cuda(cu_seqlens, "cu_seqlens");
    char *base = reinterpret_cast<char *>(ptrs[0]);
    const int64_t elem = qkv.element_size();
    const int64_t fused_stride = qkv.stride(1);
    Flash_fwd_params params;
    jt_fill_params(params, qkv.dtype(), batch_size, max_seqlen, max_seqlen,
                   num_heads, num_heads, head_size,
                   base,
                   base + fused_stride * elem,
                   base + 2 * fused_stride * elem,
                   0, 0, 0,
                   qkv.stride(0), qkv.stride(0), qkv.stride(0),
                   qkv.stride(2), qkv.stride(2), qkv.stride(2),
                   out, ptrs[2],
                   ptrs[1],
                   ptrs[1],
                   softmax_lse, ptrs[3], softmax_scale, is_causal,
                   window_size_left, window_size_right, true);
    jt_finish_and_run(params, ptrs[4]);
    return out;
}

at::Tensor
jt_varlen_kvpacked_fwd(py::handle q_obj,
                       py::handle kv_obj,
                       py::handle cu_seqlens_q_obj,
                       py::handle cu_seqlens_k_obj,
                       int max_seqlen_q,
                       int max_seqlen_k,
                       float softmax_scale,
                       bool is_causal,
                       int window_size_left,
                       int window_size_right) {
    auto q = jt_readonly_tensor(q_obj, "q");
    auto kv = jt_readonly_tensor(kv_obj, "kv");
    auto cu_seqlens_q = jt_readonly_tensor(cu_seqlens_q_obj, "cu_seqlens_q");
    auto cu_seqlens_k = jt_readonly_tensor(cu_seqlens_k_obj, "cu_seqlens_k");
    at::cuda::CUDAGuard device_guard{0};
    TORCH_CHECK(q.dim() == 3, "q must be [total_q, heads, dim]");
    TORCH_CHECK(kv.dim() == 4 && kv.size(1) == 2, "kv must be [total_k, 2, heads, dim]");
    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16,
                "q must be fp16 or bf16");
    TORCH_CHECK(kv.dtype() == q.dtype(), "kv dtype mismatch");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32 && cu_seqlens_k.dtype() == torch::kInt32,
                "cu_seqlens tensors must be int32");
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int total_q = q.size(0);
    const int num_heads = q.size(1);
    const int num_heads_k = kv.size(2);
    const int head_size = q.size(2);
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide query heads");
    TORCH_CHECK(head_size == kv.size(3), "q/kv head dim mismatch");
    auto opts = jt_cuda_options(q.dtype());
    auto out = torch::empty({total_q, num_heads, head_size}, opts);
    auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(torch::kFloat));
    auto rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
    void *ptrs[7];
    ::jtorch::detail::data_ptrs({q, kv, cu_seqlens_q, cu_seqlens_k,
                                 out, softmax_lse, rng_state}, ptrs);
    jt_check_cuda(q, "q");
    jt_check_cuda(kv, "kv");
    jt_check_cuda(cu_seqlens_q, "cu_seqlens_q");
    jt_check_cuda(cu_seqlens_k, "cu_seqlens_k");
    char *kv_base = reinterpret_cast<char *>(ptrs[1]);
    const int64_t elem = kv.element_size();
    const int64_t fused_stride = kv.stride(1);
    Flash_fwd_params params;
    jt_fill_params(params, q.dtype(), batch_size, max_seqlen_q, max_seqlen_k,
                   num_heads, num_heads_k, head_size,
                   ptrs[0],
                   kv_base,
                   kv_base + fused_stride * elem,
                   0, 0, 0,
                   q.stride(0), kv.stride(0), kv.stride(0),
                   q.stride(1), kv.stride(2), kv.stride(2),
                   out, ptrs[4],
                   ptrs[2],
                   ptrs[3],
                   softmax_lse, ptrs[5], softmax_scale, is_causal,
                   window_size_left, window_size_right, true);
    jt_finish_and_run(params, ptrs[6]);
    return out;
}

at::Tensor
jt_qkvpacked_fwd(py::handle qkv_obj,
                 float softmax_scale,
                 bool is_causal,
                 int window_size_left,
    int window_size_right) {
    auto qkv = jt_readonly_tensor(qkv_obj, "qkv");
    at::cuda::CUDAGuard device_guard{0};
    TORCH_CHECK(qkv.dim() == 5 && qkv.size(2) == 3, "qkv must be [batch, seqlen, 3, heads, dim]");
    TORCH_CHECK(qkv.dtype() == torch::kFloat16 || qkv.dtype() == torch::kBFloat16,
                "qkv must be fp16 or bf16");
    const int batch_size = qkv.size(0);
    const int seqlen = qkv.size(1);
    const int num_heads = qkv.size(3);
    const int head_size = qkv.size(4);
    auto opts = jt_cuda_options(qkv.dtype());
    auto out = torch::empty({batch_size, seqlen, num_heads, head_size}, opts);
    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen}, opts.dtype(torch::kFloat));
    auto rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
    void *ptrs[4];
    ::jtorch::detail::data_ptrs({qkv, out, softmax_lse, rng_state}, ptrs);
    jt_check_cuda(qkv, "qkv");
    char *base = reinterpret_cast<char *>(ptrs[0]);
    const int64_t elem = qkv.element_size();
    const int64_t fused_stride = qkv.stride(2);
    Flash_fwd_params params;
    jt_fill_params(params, qkv.dtype(), batch_size, seqlen, seqlen,
                   num_heads, num_heads, head_size,
                   base,
                   base + fused_stride * elem,
                   base + 2 * fused_stride * elem,
                   qkv.stride(0), qkv.stride(0), qkv.stride(0),
                   qkv.stride(1), qkv.stride(1), qkv.stride(1),
                   qkv.stride(3), qkv.stride(3), qkv.stride(3),
                   out, ptrs[1], nullptr, nullptr, softmax_lse, ptrs[2], softmax_scale,
                   is_causal, window_size_left, window_size_right, false);
    jt_finish_and_run(params, ptrs[3]);
    return out;
}

at::Tensor
jt_kvpacked_fwd(py::handle q_obj,
                py::handle kv_obj,
                float softmax_scale,
                bool is_causal,
                int window_size_left,
    int window_size_right) {
    auto q = jt_readonly_tensor(q_obj, "q");
    auto kv = jt_readonly_tensor(kv_obj, "kv");
    at::cuda::CUDAGuard device_guard{0};
    TORCH_CHECK(q.dim() == 4, "q must be [batch, seqlen_q, heads, dim]");
    TORCH_CHECK(kv.dim() == 5 && kv.size(2) == 2, "kv must be [batch, seqlen_k, 2, heads, dim]");
    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16,
                "q must be fp16 or bf16");
    TORCH_CHECK(kv.dtype() == q.dtype(), "kv dtype mismatch");
    const int batch_size = q.size(0);
    const int seqlen_q = q.size(1);
    const int seqlen_k = kv.size(1);
    const int num_heads = q.size(2);
    const int num_heads_k = kv.size(3);
    const int head_size = q.size(3);
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide query heads");
    TORCH_CHECK(head_size == kv.size(4), "q/kv head dim mismatch");
    auto opts = jt_cuda_options(q.dtype());
    auto out = torch::empty({batch_size, seqlen_q, num_heads, head_size}, opts);
    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(torch::kFloat));
    auto rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
    void *ptrs[5];
    ::jtorch::detail::data_ptrs({q, kv, out, softmax_lse, rng_state}, ptrs);
    jt_check_cuda(q, "q");
    jt_check_cuda(kv, "kv");
    char *kv_base = reinterpret_cast<char *>(ptrs[1]);
    const int64_t elem = kv.element_size();
    const int64_t fused_stride = kv.stride(2);
    Flash_fwd_params params;
    jt_fill_params(params, q.dtype(), batch_size, seqlen_q, seqlen_k,
                   num_heads, num_heads_k, head_size,
                   ptrs[0],
                   kv_base,
                   kv_base + fused_stride * elem,
                   q.stride(0), kv.stride(0), kv.stride(0),
                   q.stride(1), kv.stride(1), kv.stride(1),
                   q.stride(2), kv.stride(3), kv.stride(3),
                   out, ptrs[2], nullptr, nullptr, softmax_lse, ptrs[3], softmax_scale,
                   is_causal, window_size_left, window_size_right, false);
    jt_finish_and_run(params, ptrs[4]);
    return out;
}

} // namespace FLASH_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &FLASH_NAMESPACE::jt_fwd, "Jittor direct forward");
    m.def("varlen_fwd", &FLASH_NAMESPACE::jt_varlen_fwd, "Jittor direct varlen forward");
    m.def("varlen_qkvpacked_fwd", &FLASH_NAMESPACE::jt_varlen_qkvpacked_fwd, "Jittor direct varlen qkvpacked forward");
    m.def("varlen_kvpacked_fwd", &FLASH_NAMESPACE::jt_varlen_kvpacked_fwd, "Jittor direct varlen kvpacked forward");
    m.def("qkvpacked_fwd", &FLASH_NAMESPACE::jt_qkvpacked_fwd, "Jittor direct qkvpacked forward");
    m.def("kvpacked_fwd", &FLASH_NAMESPACE::jt_kvpacked_fwd, "Jittor direct kvpacked forward");
}
'''
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


def _packed_split_enabled() -> bool:
    return _truthy(os.environ.get("JITTOR_FLASH_ATTN_FUSED_PACKED_SPLIT"))


def _is_cuda_jittor_var(x) -> bool:
    if not _packed_split_enabled():
        return False
    try:
        import jittor as jt
    except Exception:
        return False
    try:
        return bool(jt.flags.use_cuda) and isinstance(x, jt.Var)
    except Exception:
        return False


def _split_qkvpacked_cuda(qkv):
    if not _is_cuda_jittor_var(qkv):
        return None
    try:
        import jittor as jt

        shape = list(qkv.shape)
        dtype = qkv.dtype
        if len(shape) == 4 and int(shape[1]) == 3:
            total, _, heads, dim = shape
            n = int(total) * int(heads) * int(dim)
            if n == 0:
                return qkv[:, 0], qkv[:, 1], qkv[:, 2]
            out_shape = [total, heads, dim]
            q, k, v = jt.code(
                [out_shape, out_shape, out_shape],
                [dtype, dtype, dtype],
                [qkv],
                cuda_src="""
__global__ static void split_qkv(@ARGS_DEF) {
    @PRECALC
    int64_t n = (int64_t)out0_shape0 * out0_shape1 * out0_shape2;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int d = i % out0_shape2;
    int h = (i / out0_shape2) % out0_shape1;
    int t = i / ((int64_t)out0_shape2 * out0_shape1);
    @out0(t, h, d) = @in0(t, 0, h, d);
    @out1(t, h, d) = @in0(t, 1, h, d);
    @out2(t, h, d) = @in0(t, 2, h, d);
}
int64_t n = (int64_t)out0_shape0 * out0_shape1 * out0_shape2;
split_qkv<<<(n + 255) / 256, 256>>>(@ARGS);
""",
            )
            _PACKED_SPLIT_STATS["qkv_cuda"] += 1
            return q, k, v
        if len(shape) == 5 and int(shape[2]) == 3:
            batch, seqlen, _, heads, dim = shape
            n = int(batch) * int(seqlen) * int(heads) * int(dim)
            if n == 0:
                return qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            out_shape = [batch, seqlen, heads, dim]
            q, k, v = jt.code(
                [out_shape, out_shape, out_shape],
                [dtype, dtype, dtype],
                [qkv],
                cuda_src="""
__global__ static void split_qkv(@ARGS_DEF) {
    @PRECALC
    int64_t n = (int64_t)out0_shape0 * out0_shape1 * out0_shape2 * out0_shape3;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int d = i % out0_shape3;
    int h = (i / out0_shape3) % out0_shape2;
    int s = (i / ((int64_t)out0_shape3 * out0_shape2)) % out0_shape1;
    int b = i / ((int64_t)out0_shape3 * out0_shape2 * out0_shape1);
    @out0(b, s, h, d) = @in0(b, s, 0, h, d);
    @out1(b, s, h, d) = @in0(b, s, 1, h, d);
    @out2(b, s, h, d) = @in0(b, s, 2, h, d);
}
int64_t n = (int64_t)out0_shape0 * out0_shape1 * out0_shape2 * out0_shape3;
split_qkv<<<(n + 255) / 256, 256>>>(@ARGS);
""",
            )
            _PACKED_SPLIT_STATS["qkv_cuda"] += 1
            return q, k, v
    except Exception as exc:
        _PACKED_SPLIT_STATS["error"] += 1
        _remember_error("fused qkvpacked split failed: %s" % exc)
        return None
    _PACKED_SPLIT_STATS["fallback"] += 1
    return None


def _split_kvpacked_cuda(kv):
    if not _is_cuda_jittor_var(kv):
        return None
    try:
        import jittor as jt

        shape = list(kv.shape)
        dtype = kv.dtype
        if len(shape) == 4 and int(shape[1]) == 2:
            total, _, heads, dim = shape
            n = int(total) * int(heads) * int(dim)
            if n == 0:
                return kv[:, 0], kv[:, 1]
            out_shape = [total, heads, dim]
            k, v = jt.code(
                [out_shape, out_shape],
                [dtype, dtype],
                [kv],
                cuda_src="""
__global__ static void split_kv(@ARGS_DEF) {
    @PRECALC
    int64_t n = (int64_t)out0_shape0 * out0_shape1 * out0_shape2;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int d = i % out0_shape2;
    int h = (i / out0_shape2) % out0_shape1;
    int t = i / ((int64_t)out0_shape2 * out0_shape1);
    @out0(t, h, d) = @in0(t, 0, h, d);
    @out1(t, h, d) = @in0(t, 1, h, d);
}
int64_t n = (int64_t)out0_shape0 * out0_shape1 * out0_shape2;
split_kv<<<(n + 255) / 256, 256>>>(@ARGS);
""",
            )
            _PACKED_SPLIT_STATS["kv_cuda"] += 1
            return k, v
        if len(shape) == 5 and int(shape[2]) == 2:
            batch, seqlen, _, heads, dim = shape
            n = int(batch) * int(seqlen) * int(heads) * int(dim)
            if n == 0:
                return kv[:, :, 0], kv[:, :, 1]
            out_shape = [batch, seqlen, heads, dim]
            k, v = jt.code(
                [out_shape, out_shape],
                [dtype, dtype],
                [kv],
                cuda_src="""
__global__ static void split_kv(@ARGS_DEF) {
    @PRECALC
    int64_t n = (int64_t)out0_shape0 * out0_shape1 * out0_shape2 * out0_shape3;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int d = i % out0_shape3;
    int h = (i / out0_shape3) % out0_shape2;
    int s = (i / ((int64_t)out0_shape3 * out0_shape2)) % out0_shape1;
    int b = i / ((int64_t)out0_shape3 * out0_shape2 * out0_shape1);
    @out0(b, s, h, d) = @in0(b, s, 0, h, d);
    @out1(b, s, h, d) = @in0(b, s, 1, h, d);
}
int64_t n = (int64_t)out0_shape0 * out0_shape1 * out0_shape2 * out0_shape3;
split_kv<<<(n + 255) / 256, 256>>>(@ARGS);
""",
            )
            _PACKED_SPLIT_STATS["kv_cuda"] += 1
            return k, v
    except Exception as exc:
        _PACKED_SPLIT_STATS["error"] += 1
        _remember_error("fused kvpacked split failed: %s" % exc)
        return None
    _PACKED_SPLIT_STATS["fallback"] += 1
    return None


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


def _direct_packed_enabled() -> bool:
    value = os.environ.get("JITTOR_FLASH_ATTN_DIRECT_ADAPTER")
    if value is None:
        value = os.environ.get("JITTOR_FLASH_ATTN_DIRECT_PACKED")
    return not _falsey(value)


def _make_official_backend(low_level: ModuleType, root: pathlib.Path,
                           packed_low_level: Optional[ModuleType] = None) -> ModuleType:
    mod = ModuleType("flashattn_jittor_official")
    mod.__file__ = os.fspath(root)
    mod._flashattn_jittor_official = True
    mod._flashattn_jittor_low_level = low_level
    mod._flashattn_jittor_packed_low_level = packed_low_level
    mod._flashattn_jittor_head_dims = tuple(_official_head_dims(root))
    mod._flashattn_jittor_dtypes = tuple(_official_dtypes())
    mod._flashattn_jittor_packed_split_stats = _PACKED_SPLIT_STATS
    low_fwd = low_level.fwd
    low_varlen_fwd = low_level.varlen_fwd
    packed_split_enabled = _packed_split_enabled()

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
        if packed_low_level is not None:
            return packed_low_level.fwd(q, k, v, float(softmax_scale), bool(causal), wl, wr)
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
        if packed_low_level is not None and _native_supported_dtype(qkv):
            _check_args(dropout_p, softcap, alibi_slopes, return_attn_probs)
            wl, wr = _window_size_pair(kwargs.get("window_size", window_size))
            scale = qkv.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
            return packed_low_level.qkvpacked_fwd(qkv, scale, bool(causal), wl, wr)
        if packed_split_enabled:
            split = _split_qkvpacked_cuda(qkv)
            if split is not None:
                return flash_attn_func(split[0], split[1], split[2], dropout_p,
                                       softmax_scale, causal, window_size,
                                       softcap, alibi_slopes, deterministic,
                                       return_attn_probs, *args, **kwargs)
        return flash_attn_func(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2],
                               dropout_p, softmax_scale, causal, window_size,
                               softcap, alibi_slopes, deterministic,
                               return_attn_probs, *args, **kwargs)

    def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None,
                                 causal=False, window_size=(-1, -1), softcap=0.0,
                                 alibi_slopes=None, deterministic=False,
                                 return_attn_probs=False, *args, **kwargs):
        if packed_low_level is not None and _native_supported_dtype(q) and _native_supported_dtype(kv):
            _check_args(dropout_p, softcap, alibi_slopes, return_attn_probs)
            wl, wr = _window_size_pair(kwargs.get("window_size", window_size))
            scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
            return packed_low_level.kvpacked_fwd(q, kv, scale, bool(causal), wl, wr)
        if packed_split_enabled:
            split = _split_kvpacked_cuda(kv)
            if split is not None:
                return flash_attn_func(q, split[0], split[1],
                                       dropout_p, softmax_scale, causal,
                                       window_size, softcap, alibi_slopes,
                                       deterministic, return_attn_probs,
                                       *args, **kwargs)
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
        if (packed_low_level is not None and seqused_k is None and leftpad_k is None
                and block_table is None and alibi_slopes is None):
            return packed_low_level.varlen_fwd(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                int(max_seqlen_q), int(max_seqlen_k),
                float(softmax_scale), bool(causal), wl, wr)
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
        if packed_low_level is not None and _native_supported_dtype(qkv):
            _check_args(dropout_p, softcap, alibi_slopes, return_attn_probs)
            wl, wr = _window_size_pair(kwargs.get("window_size", window_size))
            scale = qkv.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
            return packed_low_level.varlen_qkvpacked_fwd(
                qkv, cu_seqlens, int(max_seqlen), scale, bool(causal), wl, wr)
        if packed_split_enabled:
            split = _split_qkvpacked_cuda(qkv)
            if split is not None:
                return flash_attn_varlen_func(
                    split[0], split[1], split[2], cu_seqlens, cu_seqlens,
                    max_seqlen, max_seqlen, dropout_p, softmax_scale, causal,
                    window_size, softcap, alibi_slopes, deterministic,
                    return_attn_probs, *args, **kwargs)
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
        if packed_low_level is not None and _native_supported_dtype(q) and _native_supported_dtype(kv):
            _check_args(dropout_p, softcap, alibi_slopes, return_attn_probs)
            wl, wr = _window_size_pair(kwargs.get("window_size", window_size))
            scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
            return packed_low_level.varlen_kvpacked_fwd(
                q, kv, cu_seqlens_q, cu_seqlens_k,
                int(max_seqlen_q), int(max_seqlen_k), scale,
                bool(causal), wl, wr)
        if packed_split_enabled:
            split = _split_kvpacked_cuda(kv)
            if split is not None:
                return flash_attn_varlen_func(
                    q, split[0], split[1], cu_seqlens_q, cu_seqlens_k,
                    max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale,
                    causal, window_size, softcap, alibi_slopes, deterministic,
                    return_attn_probs, *args, **kwargs)
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


def _load_official_packed_flash_attention(root: pathlib.Path, low_level: ModuleType) -> Optional[ModuleType]:
    if not _direct_packed_enabled():
        return None
    build_dir = _official_packed_build_dir(root)
    sources = [_official_packed_source(build_dir)]
    include_dirs = [
        os.fspath((root / "csrc" / "flash_attn").resolve()),
        os.fspath((root / "csrc" / "flash_attn" / "src").resolve()),
        os.fspath((root / "csrc" / "cutlass" / "include").resolve()),
    ]
    cflags, cuda_cflags = _official_flags()
    low_path = os.path.abspath(getattr(low_level, "__file__", "") or "")
    if not low_path:
        _remember_error("official flash-attn packed direct backend missing low-level module path")
        return None
    low_dir = os.path.dirname(low_path)
    module_name = "flash_attn_2_cuda_jittor_packed"
    _log("compile official flash-attn packed direct backend from %s" % root)
    try:
        from jittor.torch_shim.cpp_extension.torch_utils import load

        return load(
            name=module_name,
            sources=sources,
            extra_include_paths=include_dirs,
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_cflags,
            extra_ldflags=[low_path, "-Xlinker", "-rpath", "-Xlinker", low_dir],
            build_directory=build_dir,
            import_identity=_official_import_identity(
                "official-packed", build_dir, module_name),
            verbose=_verbose(),
        )
    except Exception as exc:
        _remember_error("compile official flash-attn packed direct backend failed: %s" % exc)
        return None


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
            import_identity=_official_import_identity(
                "official-forward", build_dir, module_name),
            verbose=_verbose(),
            force=_truthy(os.environ.get("JITTOR_FLASH_ATTN_FORCE_BUILD")),
        )
    except Exception as exc:
        _remember_error("compile official flash-attn backend failed: %s" % exc)
        return None
    packed = _load_official_packed_flash_attention(root, low)
    return _make_official_backend(low, root, packed)


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


def _build_setup_backend(root: pathlib.Path) -> bool:
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
        except BaseException:
            # An audit hook exception would abort the environment write. Mark
            # the token unusable and leave the write itself untouched.
            state["reliable"] = False

    # Audit hooks cannot be removed. Keep the state on sys so module reloads
    # reuse this hook rather than installing duplicate watchers.
    state["hook"] = audit_hook
    try:
        sys.addaudithook(audit_hook)
        sys.audit(_BACKEND_ENV_EPOCH_PROBE)
    except Exception:
        pass
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
