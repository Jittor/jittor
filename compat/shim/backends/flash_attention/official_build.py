"""FlashAttention official build implementation."""
from __future__ import annotations
import pathlib
from types import ModuleType
from typing import List, Optional, Tuple

def _looks_like_official_flash_attention(root: pathlib.Path) -> bool:
    return (
        (root / "csrc" / "flash_attn" / "flash_api.cpp").is_file()
        and (root / "csrc" / "flash_attn" / "src" / "flash.h").is_file()
    )


def _official_build_dir(root: pathlib.Path) -> str:
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    digest_key = _facade.os.fspath(root.resolve())
    try:
        head = _facade.subprocess.check_output(
            ["git", "-C", _facade.os.fspath(root), "rev-parse", "HEAD"],
            text=True,
            stderr=_facade.subprocess.DEVNULL,
        ).strip()
        if head:
            digest_key += "|" + head
    except _facade.EXPECTED as exc:
        _facade.swallowed("shim/backends/flash_attention.py _official_build_dir: head = subprocess.check_output(", exc)
    digest_key += "|head_dims=" + ",".join(_facade._official_head_dims(root))
    digest_key += "|dtypes=" + ",".join(_facade._official_dtypes())
    digest_key += "|native_forward_backward_dropout=1"
    digest = _facade.hashlib.sha256(digest_key.encode("utf-8")).hexdigest()[:16]
    return _facade._default_build_root("flashattn_jittor", "official_flash_attn", digest)


def _official_packed_build_dir(root: pathlib.Path) -> str:
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    digest_key = _facade.os.fspath(root.resolve())
    try:
        head = _facade.subprocess.check_output(
            ["git", "-C", _facade.os.fspath(root), "rev-parse", "HEAD"],
            text=True,
            stderr=_facade.subprocess.DEVNULL,
        ).strip()
        if head:
            digest_key += "|" + head
    except _facade.EXPECTED as exc:
        _facade.swallowed("shim/backends/flash_attention.py _official_packed_build_dir: head = subprocess.check_output(", exc)
    digest_key += "|head_dims=" + ",".join(_facade._official_head_dims(root))
    digest_key += "|dtypes=" + ",".join(_facade._official_dtypes())
    digest_key += "|direct_packed_forward=6"
    digest = _facade.hashlib.sha256(digest_key.encode("utf-8")).hexdigest()[:16]
    return _facade._default_build_root("flashattn_jittor", "official_flash_attn_packed", digest)


def _official_import_identity(kind: str, build_dir: str, module_name: str,
                              generation: Optional[int] = None) -> str:
    """Identify one official build without changing its extension name."""
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    if generation is None:
        generation = _facade._BACKEND_LOAD_GENERATION
    build_digest = _facade.pathlib.Path(build_dir).name
    safe_kind = "".join(ch if ch.isalnum() else "_" for ch in kind)
    namespace = "_jittor_flash_%s_%s_g%d" % (
        safe_kind, build_digest, int(generation))
    return namespace + "." + module_name


def _ensure_official_cutlass(root: pathlib.Path) -> bool:
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    cutlass_h = root / "csrc" / "cutlass" / "include" / "cutlass" / "cutlass.h"
    if cutlass_h.is_file():
        return True
    gitmodules = root / ".gitmodules"
    if (root / ".git").exists() and gitmodules.is_file():
        try:
            _facade._log("initializing official flash-attn CUTLASS submodule")
            _facade.subprocess.run(
                ["git", "-C", _facade.os.fspath(root), "submodule", "update", "--init", "csrc/cutlass"],
                check=True,
                stdout=_facade.subprocess.PIPE,
                stderr=_facade.subprocess.PIPE,
                text=True,
            )
        except _facade.EXPECTED as exc:
            _facade.swallowed("shim/backends/flash_attention.py _ensure_official_cutlass: _log('initializing official flash-attn CUTLASS submodule')", exc)
            _facade._remember_error("initialize official flash-attn CUTLASS failed: %s" % exc)
            return False
    if not cutlass_h.is_file():
        _facade._remember_error("official flash-attn CUTLASS headers missing: %s" % cutlass_h)
        return False
    return True


def _official_dropout_backward_supported(head_dim: int, cuda_archs) -> bool:
    if int(head_dim) <= 192:
        return True
    try:
        archs = {int(arch) for arch in cuda_archs}
    except (TypeError, ValueError):
        return False
    # Upstream only supports >192 dropout backward on A100/A800 and H100/H800.
    return bool(archs) and archs.issubset({80, 90})


def _official_head_dims(root: pathlib.Path) -> List[str]:
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    raw = _facade.os.environ.get("JITTOR_FLASH_ATTN_HEAD_DIMS") or _facade.os.environ.get("FLASH_ATTN_HEAD_DIMS")
    if raw:
        if raw.strip().lower() in ("all", "full", "*"):
            dims = list(_facade._OFFICIAL_FLASH_ATTN_HEAD_DIMS)
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
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    raw = _facade.os.environ.get("JITTOR_FLASH_ATTN_DTYPES") or _facade.os.environ.get("FLASH_ATTN_DTYPES")
    if raw:
        if raw.strip().lower() in ("all", "full", "*"):
            dtypes = list(_facade._OFFICIAL_FLASH_ATTN_DTYPES)
        else:
            dtypes = [item.strip().lower() for item in raw.replace(";", ",").split(",") if item.strip()]
    else:
        dtypes = list(_facade._OFFICIAL_FLASH_ATTN_DTYPES)
    out = [dt for dt in dtypes if dt in ("fp16", "bf16")]
    return out or ["fp16", "bf16"]


def _official_sources(root: pathlib.Path) -> List[str]:
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    src_dir = root / "csrc" / "flash_attn" / "src"
    sources: List[pathlib.Path] = [root / "csrc" / "flash_attn" / "flash_api.cpp"]
    for prefix in ("flash_fwd", "flash_fwd_split"):
        for dim in _facade._official_head_dims(root):
            for dtype in _facade._official_dtypes():
                for causal in ("", "_causal"):
                    path = src_dir / ("%s_hdim%s_%s%s_sm80.cu" % (prefix, dim, dtype, causal))
                    if path.is_file():
                        sources.append(path)
                    else:
                        _facade._remember_error("official flash-attn source missing: %s" % path)
    for dim in _facade._official_head_dims(root):
        for dtype in _facade._official_dtypes():
            for causal in ("", "_causal"):
                path = src_dir / ("flash_bwd_hdim%s_%s%s_sm80.cu" % (
                    dim, dtype, causal))
                if path.is_file():
                    sources.append(path)
                else:
                    _facade._remember_error("official flash-attn source missing: %s" % path)
    return [_facade.os.fspath(p.resolve()) for p in sources]


def _official_compiled_specs(root: pathlib.Path) -> set:
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    src_dir = root / "csrc" / "flash_attn" / "src"
    specs = set()
    for dim in _facade._official_head_dims(root):
        for dtype in _facade._official_dtypes():
            for causal_suffix, causal in (("", False), ("_causal", True)):
                fwd = src_dir / ("flash_fwd_hdim%s_%s%s_sm80.cu" % (dim, dtype, causal_suffix))
                split = src_dir / ("flash_fwd_split_hdim%s_%s%s_sm80.cu" % (dim, dtype, causal_suffix))
                if fwd.is_file():
                    specs.add(("fwd", dtype, dim, causal))
                if split.is_file():
                    specs.add(("split", dtype, dim, causal))
                bwd = src_dir / ("flash_bwd_hdim%s_%s%s_sm80.cu" % (
                    dim, dtype, causal_suffix))
                if bwd.is_file():
                    specs.add(("bwd", dtype, dim, causal))
    return specs


def _official_flags() -> Tuple[List[str], List[str]]:
    common = [
        "-O3",
        "-std=c++17",
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
        "-DFLASHATTENTION_DISABLE_ALIBI",
        "-DFLASHATTENTION_DISABLE_SOFTCAP",
    ]
    return common, cuda


def _load_official_packed_flash_attention(root: pathlib.Path, low_level: ModuleType) -> Optional[ModuleType]:
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    if not _facade._direct_packed_enabled():
        return None
    build_dir = _facade._official_packed_build_dir(root)
    sources = [_facade._official_packed_source(build_dir)]
    include_dirs = [
        _facade.os.fspath((root / "csrc" / "flash_attn").resolve()),
        _facade.os.fspath((root / "csrc" / "flash_attn" / "src").resolve()),
        _facade.os.fspath((root / "csrc" / "cutlass" / "include").resolve()),
    ]
    cflags, cuda_cflags = _facade._official_flags()
    low_path = _facade.os.path.abspath(getattr(low_level, "__file__", "") or "")
    if not low_path:
        _facade._remember_error("official flash-attn packed direct backend missing low-level module path")
        return None
    low_dir = _facade.os.path.dirname(low_path)
    module_name = "flash_attn_2_cuda_jittor_packed"
    _facade._log("compile official flash-attn packed direct backend from %s" % root)
    try:
        from jittor.compat.shim.cpp_extension.torch_utils import load

        return load(
            name=module_name,
            sources=sources,
            extra_include_paths=include_dirs,
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_cflags,
            extra_ldflags=[low_path, "-Xlinker", "-rpath", "-Xlinker", low_dir],
            build_directory=build_dir,
            import_identity=_facade._official_import_identity(
                "official-packed", build_dir, module_name),
            verbose=_facade._verbose(),
        )
    except _facade.EXPECTED as exc:
        _facade.swallowed("shim/backends/flash_attention.py _load_official_packed_flash_attention: from jittor.compat.shim.cpp_extension.torch_utils impor...", exc)
        _facade._remember_error("compile official flash-attn packed direct backend failed: %s" % exc)
        return None


def _load_official_flash_attention(root: pathlib.Path) -> Optional[ModuleType]:
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    if not _facade._looks_like_official_flash_attention(root):
        return None
    if not _facade._ensure_official_cutlass(root):
        return None
    build_dir = _facade._official_build_dir(root)
    sources = _facade._official_sources(root)
    sources.append(_facade._official_stub_source(build_dir, root))
    include_dirs = [
        _facade.os.fspath((root / "csrc" / "flash_attn").resolve()),
        _facade.os.fspath((root / "csrc" / "flash_attn" / "src").resolve()),
        _facade.os.fspath((root / "csrc" / "cutlass" / "include").resolve()),
    ]
    cflags, cuda_cflags = _facade._official_flags()
    module_name = "flash_attn_2_cuda_jittor"
    _facade._log("compile official flash-attn training backend from %s" % root)
    try:
        from jittor.compat.shim.cpp_extension.torch_utils import load

        low = load(
            name=module_name,
            sources=sources,
            extra_include_paths=include_dirs,
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_cflags,
            build_directory=build_dir,
            import_identity=_facade._official_import_identity(
                "official-training", build_dir, module_name),
            verbose=_facade._verbose(),
            force=_facade._truthy(_facade.os.environ.get("JITTOR_FLASH_ATTN_FORCE_BUILD")),
        )
    except _facade.EXPECTED as exc:
        _facade.swallowed("shim/backends/flash_attention.py _load_official_flash_attention: from jittor.compat.shim.cpp_extension.torch_utils impor...", exc)
        _facade._remember_error("compile official flash-attn backend failed: %s" % exc)
        return None
    packed = _facade._load_official_packed_flash_attention(root, low)
    return _facade._make_official_backend(low, root, packed)
