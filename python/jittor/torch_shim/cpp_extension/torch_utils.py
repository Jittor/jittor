"""torch.utils.cpp_extension facade backed by Jittor's extension builder.

This module is shared by both entry points:

* deployed ``import torch`` via ``jittor/torch_shim/torch__init__.py``
* bare ``import jittor as torch`` via ``jittor/torch_compat.py``

Keeping the facade here avoids two subtly different BuildExtension/load
implementations for PyTorch-style CUDA extensions such as 3DGS' rasterizer.
"""
import importlib.util
import os
import sys
import tempfile
import types
import hashlib


def _jt_cpp_build_cfg():
    from jittor.torch_shim import cpp_extension as _b
    return _b.cfg()


def _default_build_root(*parts):
    root = os.environ.get("JITTOR_TORCH_EXTENSIONS_DIR")
    if root is None:
        try:
            import jittor as _jt
            root = os.path.join(_jt.flags.cache_path, "torch_extensions")
        except Exception:
            root = os.path.join(os.path.expanduser("~"), ".cache", "jittor_torch_extensions")
    return os.path.join(root, *parts)


def _external_build_dir(kind, ext_name, output_path=None):
    key = os.path.abspath(output_path or ext_name)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return _default_build_root(kind, ext_name.replace(".", "_"), digest)


class _JtTorchExtension:
    """setuptools.Extension subclass carrying metadata BuildExtension needs."""

    def __new__(cls, name, sources, include_dirs=None, define_macros=None,
                extra_compile_args=None, extra_link_args=None, **kw):
        from setuptools import Extension
        ext = Extension(
            name, list(sources),
            include_dirs=list(include_dirs or []),
            define_macros=list(define_macros or []),
            language="c++",
        )
        ext.extra_compile_args = extra_compile_args if extra_compile_args is not None else {}
        ext.extra_link_args = list(extra_link_args or [])
        ext._jt_is_cuda = (cls is not _CppExtensionCPU)
        return ext


class _CppExtensionCPU(_JtTorchExtension):
    pass


def CUDAExtension(name, sources, include_dirs=None, define_macros=None,
                  extra_compile_args=None, extra_link_args=None, **kw):
    return _JtTorchExtension(name, sources, include_dirs=include_dirs,
                             define_macros=define_macros,
                             extra_compile_args=extra_compile_args,
                             extra_link_args=extra_link_args, **kw)


def CppExtension(name, sources, include_dirs=None, define_macros=None,
                 extra_compile_args=None, extra_link_args=None, **kw):
    return _CppExtensionCPU(name, sources, include_dirs=include_dirs,
                            define_macros=define_macros,
                            extra_compile_args=extra_compile_args,
                            extra_link_args=extra_link_args, **kw)


def _make_build_extension():
    from setuptools.command.build_ext import build_ext as _setuptools_build_ext

    class BuildExtension(_setuptools_build_ext):
        """Route torch-style extensions through Jittor's libtorch ABI shim."""

        @classmethod
        def with_options(cls, **options):
            return cls

        def build_extension(self, ext):
            from jittor.torch_shim import cpp_extension as _b
            eca = getattr(ext, "extra_compile_args", {}) or {}
            if isinstance(eca, dict):
                extra_cflags = eca.get("cxx")
                extra_cuda_cflags = eca.get("nvcc")
            else:
                extra_cflags = list(eca)
                extra_cuda_cflags = list(eca)

            define_macros = []
            for m in (ext.define_macros or []):
                if isinstance(m, (tuple, list)):
                    nm = m[0]
                    val = m[1] if len(m) > 1 else None
                    define_macros.append((nm, val) if val is not None else nm)
                else:
                    define_macros.append(m)

            out_path = self.get_ext_fullpath(ext.name)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            build_dir = _external_build_dir("build_ext", ext.name, out_path)
            _b.build(
                name=ext.name.split(".")[-1],
                sources=[os.path.abspath(s) for s in ext.sources],
                build_dir=build_dir,
                output_path=out_path,
                include_dirs=list(ext.include_dirs or []),
                define_macros=define_macros,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_ldflags=list(getattr(ext, "extra_link_args", []) or []),
                verbose=True,
            )

        def get_ext_filename(self, ext_name):
            suffix = _jt_cpp_build_cfg()["ext_suffix"]
            return os.path.join(*ext_name.split(".")) + suffix

    return BuildExtension


try:
    BuildExtension = _make_build_extension()
except Exception:  # pragma: no cover - setuptools missing
    class BuildExtension:
        def __init__(self, *a, **k):
            raise RuntimeError(
                "torch.utils.cpp_extension.BuildExtension requires setuptools, "
                "which could not be imported")

        @classmethod
        def with_options(cls, **options):
            return cls


def load(name, sources, extra_include_paths=None, extra_cflags=None,
         extra_cuda_cflags=None, extra_ldflags=None, build_directory=None,
         verbose=False, **kw):
    """Compile sources against Jittor's libtorch ABI shim and import the module."""
    from jittor.torch_shim import cpp_extension as _b
    if isinstance(sources, str):
        sources = [sources]
    build_directory = build_directory or _default_build_root("load", name)
    os.makedirs(build_directory, exist_ok=True)
    suffix = _b.cfg()["ext_suffix"]
    out_path = os.path.join(build_directory, name + suffix)
    _b.build(
        name=name,
        sources=[os.path.abspath(s) for s in sources],
        build_dir=build_directory,
        output_path=out_path,
        include_dirs=list(extra_include_paths or []),
        define_macros=None,
        extra_cflags=list(extra_cflags or []),
        extra_cuda_cflags=list(extra_cuda_cflags or []),
        extra_ldflags=list(extra_ldflags or []),
        verbose=verbose,
    )
    spec = importlib.util.spec_from_file_location(name, out_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


def load_inline(name, cpp_sources=None, cuda_sources=None, functions=None,
                extra_include_paths=None, extra_cflags=None,
                extra_cuda_cflags=None, extra_ldflags=None, build_directory=None,
                verbose=False, with_cuda=None, **kw):
    """Write inline sources to files and defer to :func:`load`."""
    if isinstance(cpp_sources, str):
        cpp_sources = [cpp_sources]
    if isinstance(cuda_sources, str):
        cuda_sources = [cuda_sources]
    build_directory = build_directory or _default_build_root("inline", name)
    os.makedirs(build_directory, exist_ok=True)
    srcs = []
    cpp_body = "\n".join(cpp_sources or [])
    if functions:
        if isinstance(functions, dict):
            names = list(functions.keys())
        elif isinstance(functions, str):
            names = [functions]
        else:
            names = list(functions)
        binds = "\n".join(f'  m.def("{fn}", &{fn});' for fn in names)
        cpp_body += (
            '\n#include <torch/extension.h>\n'
            'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n' + binds + '\n}\n'
        )
    if cpp_body.strip():
        p = os.path.join(build_directory, name + "_inline.cpp")
        with open(p, "w") as f:
            f.write(cpp_body)
        srcs.append(p)
    for i, cu in enumerate(cuda_sources or []):
        p = os.path.join(build_directory, f"{name}_inline_{i}.cu")
        with open(p, "w") as f:
            f.write(cu)
        srcs.append(p)
    return load(name, srcs, extra_include_paths=extra_include_paths,
                extra_cflags=extra_cflags, extra_cuda_cflags=extra_cuda_cflags,
                extra_ldflags=extra_ldflags, build_directory=build_directory,
                verbose=verbose)


def include_paths(cuda=False):
    from jittor.torch_shim import cpp_extension as _b
    c = _b.cfg()
    inc = [_b.SHIM_INCLUDE, c["src_inc"], c["extern_inc"], c["extern_cuda_inc"],
           c["py_inc"]]
    if c["pybind_inc"]:
        inc.append(c["pybind_inc"])
    inc += list(c["core_dirs"])
    if cuda:
        inc.append(c["cuda_inc"])
    return [p for p in inc if p]


def library_paths(cuda=False):
    from jittor.torch_shim import cpp_extension as _b
    c = _b.cfg()
    libs = list(c["core_dirs"])
    if cuda:
        libs.append(c["cuda_lib"])
    return libs


def _jt_cuda_home():
    try:
        return _jt_cpp_build_cfg()["cuda_home"]
    except Exception:
        return os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")


def make_cpp_extension_module():
    mod = types.ModuleType("torch.utils.cpp_extension")
    mod.BuildExtension = BuildExtension
    mod.CUDAExtension = CUDAExtension
    mod.CppExtension = CppExtension
    mod.load = load
    mod.load_inline = load_inline
    mod.include_paths = include_paths
    mod.library_paths = library_paths
    mod.IS_HIP_EXTENSION = False
    mod.ROCM_HOME = None
    try:
        mod.CUDA_HOME = _jt_cuda_home()
    except Exception:
        mod.CUDA_HOME = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    mod._jittor_cpp_extension = True
    return mod


def install_cpp_extension(utils_module=None):
    mod = sys.modules.get("torch.utils.cpp_extension")
    if mod is None or not getattr(mod, "_jittor_cpp_extension", False):
        mod = make_cpp_extension_module()
        sys.modules["torch.utils.cpp_extension"] = mod
    if utils_module is not None:
        utils_module.cpp_extension = mod
    return mod
