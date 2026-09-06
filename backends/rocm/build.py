"""Source-owned ROCm library builds; no compiler or driver side effects."""

from dataclasses import dataclass
from pathlib import Path
import shlex


@dataclass(frozen=True)
class LibraryBuild:
    name: str
    sources: tuple
    include_dirs: tuple
    library_dirs: tuple
    libraries: tuple

    @property
    def extra_flags(self):
        args = ["-I" + path for path in self.include_dirs]
        args += ["-L" + path for path in self.library_dirs]
        args += ["-Wl,-rpath," + path for path in self.library_dirs]
        args += ["-l" + name for name in self.libraries]
        return " " + " ".join(shlex.quote(arg) for arg in args) + " "


_LIBRARIES = {
    "hipblas": (
        "hipblas/hipblas.h", ("hipblas",),
        ("hipblas/hipblas_matmul_op.h", "hipblas/hipblas_matmul_op.cc", "hipblas/gemm_layout.h",
         "hipblas/hipblas_wrapper.h", "hipblas/hipblas_wrapper.cc",
         "hipblas/hipblas_capabilities.cc"),
    ),
    "rocprim": (
        "rocprim/device/device_scan.hpp", (),
        ("rocprim/rocprim_cumsum_op.h", "rocprim/rocprim_cumsum_op.cc",
         "rocprim/scan.h", "rocprim/scan.cu"),
    ),
}


def library_build(rocm_home, backend_root, name):
    """Resolve a complete provider or fail before attempting compilation.

    Only providers with native implementations are selectable. MIOpen and RCCL
    must not inherit an apparent implementation by compiling CUDA sources.
    """
    if name not in _LIBRARIES:
        raise NotImplementedError("no native ROCm library provider: " + name)
    header, libraries, relative_sources = _LIBRARIES[name]
    sdk = Path(rocm_home).resolve()
    root = Path(backend_root).resolve() / "libraries"
    include_candidates = (sdk / "include", sdk / name / "include")
    includes = tuple(str(path) for path in include_candidates if (path / header).is_file())
    if not includes:
        raise RuntimeError("ROCm %s development header is missing under %s: %s"
                           % (name, sdk, header))
    library_dirs = []
    candidates = (sdk / "lib", sdk / "lib64", sdk / name / "lib", sdk / name / "lib64")
    for library in libraries:
        directory = next((path for path in candidates
                          if (path / ("lib" + library + ".so")).is_file()), None)
        if directory is None:
            raise RuntimeError("ROCm development library lib%s.so is missing under %s"
                               % (library, sdk))
        if str(directory) not in library_dirs:
            library_dirs.append(str(directory))
    sources = tuple(str(root / source) for source in relative_sources)
    for source in sources:
        if not Path(source).is_file():
            raise RuntimeError("ROCm provider source is missing: " + source)
    # HIP headers are shared by all providers, including the header-only rocPRIM.
    includes = tuple(dict.fromkeys((str(sdk / "include"),) + includes + (str(root),)))
    return LibraryBuild(name, sources, includes, tuple(library_dirs), libraries)
