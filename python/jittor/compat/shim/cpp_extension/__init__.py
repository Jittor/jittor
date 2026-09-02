"""
jittor.compat.shim.cpp_extension — compile a PyTorch C++/CUDA extension against the
Jittor-backed libtorch ABI shim, with ZERO dependency on real libtorch.

This is the in-core relocation of the standalone ``jtorch/build.py`` driver: the
shim headers (``include/torch/extension.h`` etc.) and the single jittor/CUDA impl
TU (``src/jtorch_aten.cu``) now live INSIDE jittor (next to this file), so a torch
extension's own ``.cu``/``.cpp`` kernels compile unchanged and operate on Jittor
Vars at the Python boundary — no external adapter needed.

A torch extension is normally built by ``torch.utils.cpp_extension`` which pulls
libtorch's include dirs + links ``libtorch.so``.  Here we instead point the build
at our shim headers (providing ``torch/extension.h`` etc.) and link Jittor's core
shared libraries.

The toolchain config (compiler, nvcc, CUDA arch, include/lib dirs, the jittor core
.so names) is harvested at runtime from ``jittor.compiler`` so this is portable
across machines.  Discovered the hard way: jittor's core libs are named
``jittor_core.cpython-*.so`` with NO ``lib`` prefix, so GNU ld needs the exact-name
``-l:jittor_core.cpython-*.so`` (colon) syntax, not ``-ljittor_core.cpython-*``.

The public entry point ``build(name, sources, build_dir, ...)`` is what the real
``torch.utils.cpp_extension`` shim (in ``torch__init__.py``) delegates to.
"""
import os
import sys
import glob
import importlib.machinery
import subprocess
import sysconfig
import hashlib
import json
import re
import shlex
from ...diagnostics import EXPECTED, swallowed

# --- in-jittor shim locations (was jtorch/include + jtorch/src in the adapter) ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SHIM_INCLUDE = os.path.join(_THIS_DIR, "include")
SHIM_SOURCES = [os.path.join(_THIS_DIR, "src", "jtorch_aten.cu")]
CXX11_ABI = True

_TORCH_NAMED_CUDA_ARCHES = {
    "kepler+tesla": "3.7",
    "kepler": "3.5+PTX",
    "maxwell+tegra": "5.3",
    "maxwell": "5.0;5.2+PTX",
    "pascal": "6.0;6.1+PTX",
    "volta+tegra": "7.2",
    "volta": "7.0+PTX",
    "turing": "7.5+PTX",
    "ampere+tegra": "8.7",
    "ampere": "8.0;8.6+PTX",
    "ada": "8.9+PTX",
    "hopper": "9.0+PTX",
}


def _cuda_arch_number(value):
    text = str(value).strip().lower()
    for prefix in ("sm_", "compute_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    if text.endswith("+ptx"):
        text = text[:-4]
    text = text.replace(".", "")
    return text if text.isdigit() else None


def _torch_cuda_arch_flags(value):
    flags = []
    items = []
    for item in re.split(r"[;,\s]+", str(value or "").strip()):
        expanded = _TORCH_NAMED_CUDA_ARCHES.get(item.lower(), item)
        items.extend(re.split(r"[;,\s]+", expanded))
    for item in items:
        if not item:
            continue
        ptx = item.lower().endswith("+ptx")
        arch = _cuda_arch_number(item)
        if arch is None:
            raise RuntimeError(
                f"unsupported TORCH_CUDA_ARCH_LIST entry: {item!r}"
            )
        flag = f"-gencode=arch=compute_{arch},code=sm_{arch}"
        if flag not in flags:
            flags.append(flag)
        if ptx:
            ptx_flag = f"-gencode=arch=compute_{arch},code=compute_{arch}"
            if ptx_flag not in flags:
                flags.append(ptx_flag)
    return flags


def _jittor_cuda_arch_flags(jittor):
    archs = []
    for value in getattr(jittor.flags, "cuda_archs", ()):
        arch = _cuda_arch_number(value)
        if arch is not None and arch not in archs:
            archs.append(arch)
    archs.sort(key=int)
    if not archs:
        return []
    return [f"-arch=compute_{archs[0]}"] + [
        f"-code=sm_{arch}" for arch in archs
    ]


def _compiler_cuda_arch_flags(value):
    tokens = shlex.split(str(value or ""))
    flags = []
    index = 0
    options = {
        "-arch", "--gpu-architecture", "-code", "--gpu-code",
        "-gencode", "--generate-code",
    }
    while index < len(tokens):
        token = tokens[index]
        if token in options and index + 1 < len(tokens):
            flags.extend((token, tokens[index + 1]))
            index += 2
            continue
        if token.startswith((
                "-arch=", "--gpu-architecture=", "-code=", "--gpu-code=",
                "-gencode=", "--generate-code=",
        )):
            flags.append(token)
        index += 1
    return flags


def _cuda_arch_flags(jittor, compiler):
    explicit = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if explicit:
        flags = _torch_cuda_arch_flags(explicit)
        if flags:
            return flags
    flags = _jittor_cuda_arch_flags(jittor)
    if flags:
        return flags
    flags = _compiler_cuda_arch_flags(getattr(compiler, "nvcc_flags", ""))
    if flags:
        return flags
    raise RuntimeError(
        "cannot determine CUDA architecture; make a GPU visible or set "
        "TORCH_CUDA_ARCH_LIST (for example, 8.9)"
    )


def _find_pybind_include():
    """Return a pybind11 include dir without requiring pybind11 to be importable.

    Some environments have pybind11 headers in the base interpreter/conda env or
    vendored under a real torch install, while the active venv cannot import the
    ``pybind11`` Python package.  A C++ extension only needs the headers.
    """
    try:
        import pybind11
        inc = pybind11.get_include()
        if os.path.isfile(os.path.join(inc, "pybind11", "pybind11.h")):
            return inc
    except EXPECTED as exc:
        swallowed("shim/cpp_extension/__init__.py _find_pybind_include: import pybind11", exc)

    candidates = []
    for prefix in dict.fromkeys([
        sys.prefix,
        getattr(sys, "base_prefix", None),
        os.environ.get("CONDA_PREFIX"),
    ]):
        if prefix:
            candidates += glob.glob(os.path.join(
                prefix, "lib", "python*", "site-packages", "pybind11", "include"))
            candidates.append(os.path.join(prefix, "include"))

    for p in list(sys.path):
        candidates.append(os.path.join(p, "pybind11", "include"))
        candidates.append(os.path.join(p, "torch", "include"))

    for inc in candidates:
        if inc and os.path.isfile(os.path.join(inc, "pybind11", "pybind11.h")):
            return inc
    return None


def _jittor_config():
    """Harvest the exact toolchain config from jittor.compiler (cached)."""
    # Importing the root package ensures its native core is compiled and loaded.
    import jittor  # noqa: F401
    from jittor import compiler as c

    cache_path = c.cache_path
    jittor_path = c.jittor_path  # .../python/jittor

    # locate the two core shared libs (no 'lib' prefix). jittor_core lives in the
    # cu<ver> subdir, jit_utils_core in its parent -> search from the parent.
    search_root = os.path.dirname(cache_path)
    cores = {}
    for base in ("jittor_core", "jit_utils_core"):
        hits = glob.glob(os.path.join(search_root, "**", base + ".*.so"), recursive=True)
        # Only this interpreter's ABI. A core built for another Python links
        # and loads, then runs a second copy of the runtime's static state.
        suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
        hits = [h for h in hits if os.path.basename(h)[len(base):] in suffixes]
        if not hits:
            raise RuntimeError(f"cannot find {base}.*.so under {search_root}")
        # prefer the shallowest path
        hits.sort(key=lambda p: len(p))
        cores[base] = hits[0]

    py_inc = sysconfig.get_path("include")
    pybind_inc = _find_pybind_include()

    # CUDA toolkit dir that jittor uses (parent of nvcc)
    nvcc = c.nvcc_path
    cuda_home = os.path.dirname(os.path.dirname(nvcc))
    cuda_includes = list(getattr(c, "cuda_include_dirs", ()))
    cuda_libs = list(getattr(c, "cuda_lib_dirs", ()))
    if not cuda_includes:
        cuda_includes = [os.path.join(cuda_home, "include")]
    if not cuda_libs:
        cuda_libs = [os.path.join(cuda_home, "lib64")]
    cudart_lib = c.find_cuda_library("cudart")
    cuda_wheel_stack = getattr(c, "cuda_wheel_stack", None)

    # Match torch cpp_extension's explicit arch override, otherwise compile for
    # the GPUs Jittor detected in this process. Do not inherit Jittor JIT math
    # flags: project extensions keep nvcc's default math policy unless their
    # setup.py supplies extra_cuda_cflags.
    arch_flags = _cuda_arch_flags(jittor, c)

    return {
        "cc_path": c.cc_path,
        "nvcc_path": nvcc,
        "cache_path": cache_path,
        "jittor_path": jittor_path,
        "src_inc": os.path.join(jittor_path, "src"),
        "extern_inc": os.path.join(jittor_path, "extern"),
        "extern_cuda_inc": os.path.join(jittor_path, "extern", "cuda", "inc"),
        "cuda_inc": cuda_includes[0],
        "cuda_lib": cuda_libs[0],
        "cuda_includes": cuda_includes,
        "cuda_libs": cuda_libs,
        "cudart_lib": cudart_lib,
        "cuda_wheel_fingerprint": (
            cuda_wheel_stack.fingerprint if cuda_wheel_stack else None
        ),
        "cuda_home": cuda_home,
        "py_inc": py_inc,
        "pybind_inc": pybind_inc,
        "ext_suffix": c.extension_suffix,
        "cores": cores,
        "core_dirs": sorted({os.path.dirname(p) for p in cores.values()}),
        "arch_flags": arch_flags,
        # jittor's own generated-header dir (cu<ver>) — include all dirs that hold a core lib
    }


_CFG = None


def cfg():
    global _CFG
    if _CFG is None:
        _CFG = _jittor_config()
    return _CFG


def _common_includes(c, extra):
    incs = [
        SHIM_INCLUDE,          # our torch/extension.h shim FIRST
        c["src_inc"],          # jittor core headers
        c["extern_inc"],
        c["extern_cuda_inc"],
        c["py_inc"],
    ]
    incs += list(c["cuda_includes"])
    if c["pybind_inc"]:
        incs.append(c["pybind_inc"])
    incs += list(c["core_dirs"])   # jittor generated headers (cu12.2.140 etc.)
    incs += list(extra or [])
    return [f'-I{p}' for p in incs if p]


def _link_flags(c):
    flags = []
    for d in c["core_dirs"]:
        flags.append(f'-L{d}')
        flags += ["-Xlinker", f"-rpath={d}"]
    for d in c["cuda_libs"]:
        flags.append(f'-L{d}')
        flags += ["-Xlinker", f"-rpath={d}"]
    # exact-name link (no 'lib' prefix on jittor cores) -> -l:name.so
    for base, path in c["cores"].items():
        flags.append("-l:" + os.path.basename(path))
    if c["cudart_lib"] and os.name != "nt":
        flags.append("-l:" + os.path.basename(c["cudart_lib"]))
    else:
        flags.append("-lcudart")
    flags += ["-lstdc++", "-ldl"]
    return flags


def _shim_files():
    files = list(SHIM_SOURCES)
    for base in (SHIM_INCLUDE,):
        for root, _dirs, names in os.walk(base):
            for name in names:
                if name.endswith((".h", ".hpp", ".cuh")):
                    files.append(os.path.join(root, name))
    files = sorted(dict.fromkeys(os.path.abspath(p) for p in files))
    return files


def _is_shim_source(path):
    path = os.path.abspath(path)
    return path in {os.path.abspath(p) for p in SHIM_SOURCES}


def toolchain_signature():
    """Small stable signature used to detect stale in-place extension builds."""
    c = cfg()
    shim = {}
    for path in _shim_files():
        try:
            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            shim[os.path.relpath(path, _THIS_DIR)] = digest
        except OSError:
            shim[os.path.relpath(path, _THIS_DIR)] = None
    return {
        "version": 4,
        "cc_path": c["cc_path"],
        "nvcc_path": c["nvcc_path"],
        "ext_suffix": c["ext_suffix"],
        "arch_flags": list(c["arch_flags"]),
        "cuda_libs": [os.path.realpath(path) for path in c["cuda_libs"]],
        "cudart_lib": os.path.realpath(c["cudart_lib"]) if c["cudart_lib"] else None,
        "cuda_wheel_fingerprint": c["cuda_wheel_fingerprint"],
        "extension_math_policy": "torch_cpp_extension_default",
        "shim_files": shim,
    }


def _metadata_root():
    root = os.environ.get("JITTOR_TORCH_EXTENSIONS_DIR")
    if root is None:
        try:
            import jittor as _jt
            root = os.path.join(_jt.flags.cache_path, "torch_extensions")
        except EXPECTED as exc:
            swallowed("shim/cpp_extension/__init__.py _metadata_root: import jittor as _jt", exc)
            root = os.path.join(os.path.expanduser("~"), ".cache", "jittor_torch_extensions")
    root = os.path.join(os.path.abspath(root), "metadata")
    os.makedirs(root, exist_ok=True)
    return root


def _shared_object_root():
    root = os.environ.get("JITTOR_TORCH_EXTENSIONS_DIR")
    if root is None:
        try:
            import jittor as _jt
            root = os.path.join(_jt.flags.cache_path, "torch_extensions")
        except EXPECTED as exc:
            swallowed("shim/cpp_extension/__init__.py _shared_object_root: import jittor as _jt", exc)
            root = os.path.join(os.path.expanduser("~"), ".cache", "jittor_torch_extensions")
    root = os.path.join(os.path.abspath(root), "shared_objects")
    os.makedirs(root, exist_ok=True)
    return root


def _metadata_stamp_path(kind, path):
    abspath = os.path.abspath(path)
    digest = hashlib.sha256(abspath.encode("utf-8")).hexdigest()[:24]
    name = os.path.basename(path).replace(os.sep, "_")
    return os.path.join(_metadata_root(), f"{name}.{kind}.{digest}.jittor_torch_build.json")


def stamp_path(output_path):
    return _metadata_stamp_path("so", output_path)


def output_matches_toolchain(output_path):
    try:
        with open(stamp_path(output_path), "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError:
        return False
    return data.get("toolchain") == toolchain_signature()


def write_toolchain_stamp(output_path, payload=None):
    _write_stamp(output_path, payload or {})


def _write_stamp(output_path, payload):
    data = dict(payload)
    data["toolchain"] = toolchain_signature()
    with open(stamp_path(output_path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _file_state(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return {
        "path": os.path.abspath(path),
        "mtime_ns": int(st.st_mtime_ns),
        "size": int(st.st_size),
    }


def _output_matches_build(output_path, payload):
    try:
        if not os.path.exists(output_path):
            return False
        with open(stamp_path(output_path), "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError:
        return False
    expected = dict(payload)
    expected["toolchain"] = toolchain_signature()
    return data == expected


def _object_stamp_path(obj):
    return _metadata_stamp_path("obj", obj)


def _object_matches_command(src, obj, cmd):
    try:
        if (not os.path.exists(obj)) or os.path.getmtime(obj) < os.path.getmtime(src):
            return False
        with open(_object_stamp_path(obj), "r", encoding="utf-8") as f:
            data = json.load(f)
        st = os.stat(src)
    except OSError:
        return False
    return data == {
        "source": os.path.abspath(src),
        "source_mtime_ns": int(st.st_mtime_ns),
        "cmd": list(cmd),
    }


def _write_object_stamp(src, obj, cmd):
    try:
        st = os.stat(src)
        data = {
            "source": os.path.abspath(src),
            "source_mtime_ns": int(st.st_mtime_ns),
            "cmd": list(cmd),
        }
        with open(_object_stamp_path(obj), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
    except OSError as exc:
        swallowed("shim/cpp_extension/__init__.py _write_object_stamp: st = os.stat(src)", exc)


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _shared_object_path(src, cmd_without_output):
    key = {
        "source": os.path.abspath(src),
        "source_sha256": _file_sha256(src),
        "cmd": list(cmd_without_output),
    }
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    base = os.path.splitext(os.path.basename(src))[0]
    return os.path.join(_shared_object_root(), f"{base}_{digest}.o")


def build(name, sources, build_dir, output_path=None,
          include_dirs=None, define_macros=None,
          extra_cflags=None, extra_cuda_cflags=None,
          extra_ldflags=None,
          std="c++17", abi=None, verbose=True, force=False):
    """Compile `sources` into a python extension `.so`.

    name          : final module name passed as -DTORCH_EXTENSION_NAME (e.g. "_C").
    sources       : list of .cu/.cpp/.cc absolute paths.
    build_dir     : where .o objects go.
    output_path   : final .so path (default: build_dir/<name><ext_suffix>).
    """
    c = cfg()
    if abi is None:
        abi = "1" if CXX11_ABI else "0"
    os.makedirs(build_dir, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(build_dir, name + c["ext_suffix"])
    sources = list(sources) + [s for s in SHIM_SOURCES if s not in sources]

    project_macros = [f'-D{m}' if "=" not in str(m) and not isinstance(m, tuple) else
                      (f'-D{m[0]}={m[1]}' if isinstance(m, tuple) else f'-D{m}')
                      for m in (define_macros or [])]
    shim_macros = [
        '-DHAS_CUDA', '-DIS_CUDA',
        f'-D_GLIBCXX_USE_CXX11_ABI={abi}',
        '-DJTORCH_SHIM=1',
        f'-DJTORCH_EXTENSION_MODULE_NAME={name}',
    ]
    extension_macros = project_macros + [f'-DTORCH_EXTENSION_NAME={name}'] + shim_macros

    objs = []
    cmds = []
    for src in sources:
        src = os.path.abspath(src)
        is_cu = src.endswith(".cu")
        is_shim = _is_shim_source(src)
        incs = _common_includes(c, None if is_shim else include_dirs)
        macros = shim_macros if is_shim else extension_macros
        if is_cu:
            cmd_without_output = [c["nvcc_path"], "-c", src,
                                  f"-std={std}", "-Xcompiler", "-fPIC", "-Xcompiler", "-fopenmp",
                                  "--expt-relaxed-constexpr", "--extended-lambda",
                                  "-O3", "-w", "--compiler-bindir", c["cc_path"]]
            cmd_without_output += c["arch_flags"]
            if not is_shim:
                cmd_without_output += (extra_cuda_cflags or [])
            cmd_without_output += incs + macros
            if is_shim:
                obj = _shared_object_path(src, cmd_without_output)
            else:
                oh = hashlib.md5(src.encode()).hexdigest()[:8]
                obj = os.path.join(build_dir, os.path.splitext(os.path.basename(src))[0] + "_" + oh + ".o")
            cmd = [c["nvcc_path"], "-c", src, "-o", obj] + cmd_without_output[3:]
        else:
            cmd_without_output = [c["cc_path"], "-c", src,
                                  f"-std={std}", "-fPIC", "-fopenmp", "-O3", "-w"]
            if not is_shim:
                cmd_without_output += (extra_cflags or [])
            cmd_without_output += incs + macros
            if is_shim:
                obj = _shared_object_path(src, cmd_without_output)
            else:
                oh = hashlib.md5(src.encode()).hexdigest()[:8]
                obj = os.path.join(build_dir, os.path.splitext(os.path.basename(src))[0] + "_" + oh + ".o")
            cmd = [c["cc_path"], "-c", src, "-o", obj] + cmd_without_output[3:]
        objs.append(obj)
        cmds.append((src, obj, cmd))

    # compile (skip if up-to-date unless force)
    for src, obj, cmd in cmds:
        if (not force) and _object_matches_command(src, obj, cmd):
            if verbose:
                print(f"[cpp_extension.build] up-to-date {os.path.basename(obj)}")
            continue
        if verbose:
            print(f"[cpp_extension.build] CC {os.path.basename(src)}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("CMD:", " ".join(cmd))
            print(r.stdout[-4000:])
            print(r.stderr[-8000:])
            raise RuntimeError(f"compile failed: {src}")
        elif verbose and r.stderr.strip():
            print(r.stderr[-2000:])
        _write_object_stamp(src, obj, cmd)

    # link with nvcc (handles cuda device link + -l: cores)
    link = [c["nvcc_path"], "-shared", "-o", output_path] + objs
    link += ["-Xcompiler", "-fPIC", "--compiler-bindir", c["cc_path"]]
    link += c["arch_flags"]
    link += _link_flags(c)
    link += list(extra_ldflags or [])
    stamp_payload = {
        "name": name,
        "sources": [os.path.abspath(s) for s in sources],
        "objects": [_file_state(o) for o in objs],
        "include_dirs": list(include_dirs or []),
        "define_macros": [str(m) for m in (define_macros or [])],
        "extra_cflags": list(extra_cflags or []),
        "extra_cuda_cflags": list(extra_cuda_cflags or []),
        "extra_ldflags": list(extra_ldflags or []),
        "link": list(link),
    }
    if not force and _output_matches_build(output_path, stamp_payload):
        if verbose:
            print(f"[cpp_extension.build] up-to-date {os.path.basename(output_path)}")
        return output_path
    if verbose:
        print(f"[cpp_extension.build] LINK {os.path.basename(output_path)}")
    r = subprocess.run(link, capture_output=True, text=True)
    if r.returncode != 0:
        print("LINK CMD:", " ".join(link))
        print(r.stdout[-4000:])
        print(r.stderr[-8000:])
        raise RuntimeError("link failed")
    if verbose:
        print(f"[cpp_extension.build] OK -> {output_path}")
    _write_stamp(output_path, stamp_payload)
    return output_path


if __name__ == "__main__":
    import json
    print(json.dumps({k: v for k, v in cfg().items() if k != "cores"}, indent=2, default=str))
    print("cores:", cfg()["cores"])
