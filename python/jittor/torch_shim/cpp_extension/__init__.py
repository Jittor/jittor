"""
jittor.torch_shim.cpp_extension — compile a PyTorch C++/CUDA extension against the
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
import subprocess
import sysconfig
import hashlib

# --- in-jittor shim locations (was jtorch/include + jtorch/src in the adapter) ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SHIM_INCLUDE = os.path.join(_THIS_DIR, "include")
SHIM_SOURCES = [os.path.join(_THIS_DIR, "src", "jtorch_aten.cu")]


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
    except Exception:
        pass

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
    import jittor  # noqa: ensures core is compiled/loaded
    from jittor import compiler as c

    cache_path = c.cache_path
    jittor_path = c.jittor_path  # .../python/jittor

    # locate the two core shared libs (no 'lib' prefix). jittor_core lives in the
    # cu<ver> subdir, jit_utils_core in its parent -> search from the parent.
    search_root = os.path.dirname(cache_path)
    cores = {}
    for base in ("jittor_core", "jit_utils_core"):
        hits = glob.glob(os.path.join(search_root, "**", base + ".*.so"), recursive=True)
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

    # GPU arch: parse jittor's nvcc_flags (e.g. '-arch=compute_89 -code=sm_89'),
    # else fall back to a reasonable default.
    arch_flags = []
    nvf = getattr(c, "nvcc_flags", "") or ""
    for tok in nvf.split():
        if tok.startswith("-arch=") or tok.startswith("-code=") or tok.startswith("--gpu-architecture") or tok.startswith("-gencode"):
            arch_flags.append(tok)
    if not arch_flags:
        arch_flags = ["-arch=compute_89", "-code=sm_89"]

    return {
        "cc_path": c.cc_path,
        "nvcc_path": nvcc,
        "cache_path": cache_path,
        "jittor_path": jittor_path,
        "src_inc": os.path.join(jittor_path, "src"),
        "extern_inc": os.path.join(jittor_path, "extern"),
        "extern_cuda_inc": os.path.join(jittor_path, "extern", "cuda", "inc"),
        "cuda_inc": os.path.join(cuda_home, "include"),
        "cuda_lib": os.path.join(cuda_home, "lib64"),
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
        c["cuda_inc"],
        c["py_inc"],
    ]
    if c["pybind_inc"]:
        incs.append(c["pybind_inc"])
    incs += list(c["core_dirs"])   # jittor generated headers (cu12.2.140 etc.)
    incs += list(extra or [])
    return [f'-I{p}' for p in incs if p]


def _link_flags(c):
    flags = []
    for d in c["core_dirs"]:
        flags.append(f'-L{d}')
    flags.append(f'-L{c["cuda_lib"]}')
    # exact-name link (no 'lib' prefix on jittor cores) -> -l:name.so
    for base, path in c["cores"].items():
        flags.append("-l:" + os.path.basename(path))
    flags += ["-lcudart", "-lstdc++", "-ldl"]
    return flags


def build(name, sources, build_dir, output_path=None,
          include_dirs=None, define_macros=None,
          extra_cflags=None, extra_cuda_cflags=None,
          extra_ldflags=None,
          std="c++17", abi="1", verbose=True, force=False):
    """Compile `sources` into a python extension `.so`.

    name          : final module name passed as -DTORCH_EXTENSION_NAME (e.g. "_C").
    sources       : list of .cu/.cpp/.cc absolute paths.
    build_dir     : where .o objects go.
    output_path   : final .so path (default: build_dir/<name><ext_suffix>).
    """
    c = cfg()
    os.makedirs(build_dir, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(build_dir, name + c["ext_suffix"])
    sources = list(sources) + [s for s in SHIM_SOURCES if s not in sources]

    incs = _common_includes(c, include_dirs)
    macros = [f'-D{m}' if "=" not in str(m) and not isinstance(m, tuple) else
              (f'-D{m[0]}={m[1]}' if isinstance(m, tuple) else f'-D{m}')
              for m in (define_macros or [])]
    macros += [
        f'-DTORCH_EXTENSION_NAME={name}',
        '-DHAS_CUDA', '-DIS_CUDA',
        f'-D_GLIBCXX_USE_CXX11_ABI={abi}',
        '-DJTORCH_SHIM=1',
    ]

    objs = []
    cmds = []
    for src in sources:
        src = os.path.abspath(src)
        oh = hashlib.md5(src.encode()).hexdigest()[:8]
        obj = os.path.join(build_dir, os.path.splitext(os.path.basename(src))[0] + "_" + oh + ".o")
        objs.append(obj)
        is_cu = src.endswith(".cu")
        if is_cu:
            cmd = [c["nvcc_path"], "-c", src, "-o", obj,
                   f"-std={std}", "-Xcompiler", "-fPIC", "-Xcompiler", "-fopenmp",
                   "--expt-relaxed-constexpr", "--extended-lambda",
                   "-O3", "-w", "--compiler-bindir", c["cc_path"]]
            cmd += c["arch_flags"]
            cmd += (extra_cuda_cflags or [])
        else:
            cmd = [c["cc_path"], "-c", src, "-o", obj,
                   f"-std={std}", "-fPIC", "-fopenmp", "-O3", "-w"]
            cmd += (extra_cflags or [])
        cmd += incs + macros
        cmds.append((src, obj, cmd))

    # compile (skip if up-to-date unless force)
    for src, obj, cmd in cmds:
        if (not force) and os.path.exists(obj) and os.path.getmtime(obj) >= os.path.getmtime(src):
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

    # link with nvcc (handles cuda device link + -l: cores)
    link = [c["nvcc_path"], "-shared", "-o", output_path] + objs
    link += ["-Xcompiler", "-fPIC", "--compiler-bindir", c["cc_path"]]
    link += c["arch_flags"]
    link += _link_flags(c)
    link += list(extra_ldflags or [])
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
    return output_path


if __name__ == "__main__":
    import json
    print(json.dumps({k: v for k, v in cfg().items() if k != "cores"}, indent=2, default=str))
    print("cores:", cfg()["cores"])
