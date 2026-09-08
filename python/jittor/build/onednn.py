"""oneDNN v3 discovery and reproducible source installation (no eager runtime)."""
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
import time


LIBRARY_NAMES = (
    ("lib", "libdnnl.so", "dnnl"), ("lib64", "libdnnl.so", "dnnl"),
    ("bin", "dnnl.dll", "dnnl"), ("lib", "libdnnl.dylib", "dnnl"),
    ("lib", "libmkldnn.so", "mkldnn"), ("lib", "libmkldnn.dylib", "mkldnn"),
)


def library_layout(prefix):
    for directory, filename, linker in LIBRARY_NAMES:
        path = os.path.join(str(prefix), directory, filename)
        if os.path.isfile(path):
            return path, linker
    return None


def explicit_library(directory):
    if os.path.isfile(directory):
        return os.path.abspath(directory), "dnnl"
    for _, filename, linker in LIBRARY_NAMES:
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            return os.path.abspath(path), linker
    if os.name == "nt" and os.path.isdir(directory):
        sibling = library_layout(os.path.dirname(os.path.abspath(directory)))
        if sibling is not None:
            return sibling
    raise RuntimeError("oneDNN shared library not found in explicit path: " + str(directory))


def check_usable(library, include=None):
    try:
        loaded = ctypes.CDLL(str(library), getattr(os, "RTLD_NOW", 0) | ctypes.RTLD_GLOBAL)
    except OSError as error:
        raise RuntimeError("could not load oneDNN library %s: %s" % (library, error)) from error
    for symbol in ("dnnl_version", "dnnl_primitive_create", "dnnl_memory_set_data_handle"):
        if not hasattr(loaded, symbol):
            raise RuntimeError("%s is not a usable oneDNN library: missing %s" % (library, symbol))

    class Version(ctypes.Structure):
        _fields_ = [("major", ctypes.c_int), ("minor", ctypes.c_int), ("patch", ctypes.c_int)]

    loaded.dnnl_version.restype = ctypes.POINTER(Version)
    version = loaded.dnnl_version().contents
    actual = (version.major, version.minor, version.patch)
    if actual[0] != 3:
        raise RuntimeError("oneDNN v3 is required, but %s reports %s" % (library, actual))
    if include is not None:
        header_version = None
        for relative in ("oneapi/dnnl/dnnl_version.h", "dnnl_version.h"):
            path = Path(include) / relative
            if not path.is_file():
                continue
            content = path.read_text(encoding="utf-8")
            fields = [re.search(r"#\s*define\s+DNNL_VERSION_%s\s+(\d+)" % field, content)
                      for field in ("MAJOR", "MINOR", "PATCH")]
            if all(fields):
                header_version = tuple(int(field.group(1)) for field in fields)
                break
        if header_version != actual:
            raise RuntimeError("oneDNN header/library version mismatch: %s reports %s; %s reports %s" %
                               (include, header_version, library, actual))
    return actual


@contextmanager
def _install_lock(path):
    descriptor = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == "nt":
            import msvcrt
            os.write(descriptor, b"0")
            deadline = time.monotonic() + 1800
            while True:
                os.lseek(descriptor, 0, os.SEEK_SET)
                try:
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("timed out waiting for oneDNN build lock " + str(path))
                    time.sleep(1)
        else:
            import fcntl
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def install_source(root, asset, version, compiler, download, safe_extract, allow_build=True):
    """Return the exact installed prefix; never choose a sibling by directory order."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which(compiler) or os.path.abspath(compiler)
    stat = os.stat(compiler)
    cpu_runtime = "SEQ" if platform.system() == "Darwin" else "OMP"
    options = ["-DCMAKE_BUILD_TYPE=Release", "-DDNNL_BUILD_TESTS=OFF", "-DDNNL_BUILD_EXAMPLES=OFF",
               "-DONEDNN_BUILD_GRAPH=OFF", "-DDNNL_GPU_RUNTIME=NONE", "-DDNNL_CPU_RUNTIME=" + cpu_runtime,
               "-DDNNL_ENABLE_PRIMITIVE=CONVOLUTION;MATMUL;REORDER", "-DCMAKE_INSTALL_LIBDIR=lib",
               "-DCMAKE_CXX_COMPILER=" + compiler]
    metadata = dict(source_sha256=asset.sha256, version=version, compiler=compiler,
                    compiler_size=stat.st_size, compiler_mtime_ns=stat.st_mtime_ns,
                    platform=platform.system(), machine=platform.machine(), options=options)
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:20]
    prefix = root / ("onednn-" + version + "-" + fingerprint)
    with _install_lock(root / "onednn-source.lock"):
        marker = prefix / "jittor-build.json"
        layout = library_layout(prefix)
        if marker.is_file() and layout is not None:
            if json.loads(marker.read_text(encoding="utf-8")) == metadata:
                check_usable(layout[0], prefix / "include")
                return str(prefix)
        if not allow_build:
            raise RuntimeError("JITTOR_NO_BUILD=1: oneDNN v3 is not built; provide matching "
                               "JT_BUILD_MKL_INCLUDE_PATH/JT_BUILD_MKL_LIB_PATH or bootstrap it first")
        cmake = shutil.which("cmake")
        if cmake is None:
            raise RuntimeError("building oneDNN v3 requires cmake; install it or provide "
                               "JT_BUILD_MKL_INCLUDE_PATH and JT_BUILD_MKL_LIB_PATH")
        download(asset.url, asset.filename, str(root), asset.sha256)
        source_parent = root / ("source-" + asset.sha256[:16])
        source = source_parent / ("oneDNN-" + version)
        if not (source / "CMakeLists.txt").is_file():
            source_parent.mkdir(exist_ok=True)
            with tarfile.open(str(root / asset.filename), "r:gz") as archive:
                safe_extract(archive, str(source_parent))
        build = root / ("build-" + fingerprint)
        temporary = root / ("install-" + fingerprint + "-" + str(os.getpid()))
        log = root / ("build-" + fingerprint + ".log")
        commands = [
            [cmake, "-S", str(source), "-B", str(build), *options],
            [cmake, "--build", str(build), "--config", "Release", "--parallel",
             str(min(4, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1))],
            [cmake, "--install", str(build), "--config", "Release", "--prefix", str(temporary)],
        ]
        with log.open("w", encoding="utf-8") as output:
            for command in commands:
                output.write(repr(command) + "\n")
                output.flush()
                result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError("oneDNN v3 build failed (%s); see %s" % (result.returncode, log))
        layout = library_layout(temporary)
        if layout is None:
            raise RuntimeError("oneDNN build produced no shared library; see " + str(log))
        check_usable(layout[0], temporary / "include")
        (temporary / "jittor-build.json").write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
        if prefix.exists():
            prefix.rename(root / (prefix.name + ".replaced-" + str(os.getpid())))
        temporary.rename(prefix)
        return str(prefix)
