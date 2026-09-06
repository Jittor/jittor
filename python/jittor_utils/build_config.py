"""Immutable build inputs shared by the bootstrap and backend providers."""

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Tuple


@dataclass(frozen=True)
class BuildSource:
    path: str
    language: str = "cxx"
    flags: str = ""
    compiler: str = ""

    def __post_init__(self):
        if self.language not in ("cxx", "cuda", "hip"):
            raise ValueError("unsupported backend source language: " + self.language)


@dataclass(frozen=True)
class BuildConfig:
    backend: str = "cpu"
    cc_path: str = ""
    cc_type: str = ""
    cc_flags: str = ""
    nvcc_path: str = ""
    nvcc_flags: str = ""
    kernel_flags: str = ""
    cache_path: str = ""
    jittor_path: str = ""
    has_accelerator: bool = False
    has_cuda: bool = False
    is_cuda: bool = False
    has_acl: bool = False
    has_rocm: bool = False
    has_corex: bool = False
    hipcc_path: str = ""
    tikcc_path: str = ""
    setup_fake_cuda_lib: bool = False
    extra_core_files: Tuple[str, ...] = ()
    backend_sources: Tuple[BuildSource, ...] = ()
    backend_link_flags: str = ""
    extension_compile_flags: str = ""
    kernel_compiler: str = ""
    kernel_language: str = "cxx"
    kernel_compile_flags: str = ""
    kernel_flag_filter: Tuple[str, ...] = ()
    kernel_source_roots: Tuple[str, ...] = ()
    kernel_source_suffix: str = ".cc"
    kernel_device_link: bool = False
    environment: Mapping[str, str] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict, compare=False)
    convert_nvcc_flags: Optional[Callable[[str], str]] = field(default=None, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "extra_core_files", tuple(self.extra_core_files))
        object.__setattr__(self, "backend_sources", tuple(self.backend_sources))
        object.__setattr__(self, "kernel_flag_filter", tuple(self.kernel_flag_filter))
        object.__setattr__(self, "kernel_source_roots", tuple(self.kernel_source_roots))
        if self.kernel_language not in ("cxx", "cuda", "hip"):
            raise ValueError("unsupported JIT source language: " + self.kernel_language)
        if any(not isinstance(source, BuildSource) for source in self.backend_sources):
            raise TypeError("backend_sources must contain BuildSource values")
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))
        object.__setattr__(self, "resources", MappingProxyType(dict(self.resources)))

    def evolve(self, **changes):
        return replace(self, **changes)


@dataclass(frozen=True)
class BuildContext:
    """Explicit services a backend can use without importing its consumer."""

    config: BuildConfig
    compile_module: Callable
    transform_sources: Callable
    compile: Optional[Callable] = None
    compile_custom_ops: Optional[Callable] = None
    publish_library: Optional[Callable] = None
    make_cache_dir: Optional[Callable] = None
    load_library: Optional[Callable] = None
    mpi_compile_flags: str = ""
    so: str = ".so"
    native_core: Any = None

    def with_config(self, config):
        return replace(self, config=config)


@dataclass(frozen=True)
class ModuleBuildServices:
    """The binding generator and command formatter are injected by their owner."""

    compile_single: Callable
    fix_flags: Callable
    cc_path: str
    cache_path: str
    jittor_path: str
