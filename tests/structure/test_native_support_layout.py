"""Native helpers have domain owners instead of a shared misc directory."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OWNERS = {
    "debug": ("nan_checker.h", "nan_checker.cc"),
    "runtime": ("cuda_streams.h", "float32_precision.h",
                "node_index.h", "ring_buffer.h", "ring_buffer.cc",
                "collective_dtype.h", "file_rendezvous.h"),
    "type": ("cpu_atomic.h", "cpu_atomic.cc", "cpu_math.h", "cpu_math.cc",
             "intrin.h", "cuda_atomic.h", "cuda_limits.h", "nano_string.h",
             "nano_string.cc", "nano_vector.h"),
    "utils": ("cstr.h", "deleter.h", "fast_shared_ptr.h", "hash.h",
              "stack_vector.h", "jit_cache_map.h"),
    "third_party": ("miniz.h", "miniz.cc"),
}


def test_native_support_files_have_physical_domain_owners():
    assert not (SRC / "misc").exists()
    for owner, files in OWNERS.items():
        for name in files:
            assert (SRC / owner / name).is_file(), (owner, name)
    assert (ROOT / "backends/cuda/kernels/debug/nan_checker.cu").is_file()
    assert (ROOT / "backends/cuda/runtime/driver.cc").is_file()
    assert not (SRC / "runtime/backends/cuda_streams.cc").exists()
    assert not (SRC / "debug/nan_checker.cu").exists()


def test_native_sources_do_not_include_the_removed_misc_directory():
    obsolete = []
    for root in (SRC, ROOT / "backends"):
        for path in root.rglob("*"):
            if path.suffix not in (".cc", ".cu", ".h", ".py"):
                continue
            if re.search(r'#\s*include\s*[<"]misc/', path.read_text(encoding="utf-8")):
                obsolete.append(str(path.relative_to(ROOT)))
    assert obsolete == []
