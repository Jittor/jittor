"""Corex SDK provider and explicit backend-specific runtime policy.

Moved here from ``python/jittor/extern/corex/corex_compiler.py`` by `4.15`, so
the shape matches the other migrated backends: the provider lives in
``__init__``, library build resolution in ``build.py``.
"""
import os
from collections import namedtuple
import shlex

from jittor_utils.compiler_flags import remove_flags
from jittor_utils.backend_resources import backend_root
from jittor_utils.build_config import BuildSource


CorexDiscovery = namedtuple(
    "CorexDiscovery", "home compiler_path available reason")


def discover(corex_home=None):
    """Inspect a Corex installation without importing or changing Jittor state."""
    home = corex_home or os.environ.get("COREX_HOME") or "/usr/local/corex"
    home = os.path.abspath(os.path.expanduser(home))
    compiler_path = os.path.join(home, "bin", "clang++")
    if not os.path.isdir(home):
        return CorexDiscovery(home, compiler_path, False, "COREX_HOME is absent")
    if not os.path.isfile(compiler_path):
        return CorexDiscovery(
            home, compiler_path, False, "Corex compiler is missing: %s" % compiler_path)
    return CorexDiscovery(home, compiler_path, True, "ready")


def configure(context, corex_home=None):
    """Declare Corex SDK units and kernel compilation without source rewriting."""
    discovery = discover(corex_home)
    if not discovery.available:
        raise RuntimeError(discovery.reason)
    config = context.config
    corex_root = backend_root(config.jittor_path, "corex")
    cuda_root = backend_root(config.jittor_path, "cuda")
    sdk_include = os.path.join(discovery.home, "include")
    sdk_lib = os.path.join(discovery.home, "lib64")
    if not os.path.isdir(sdk_lib) and os.path.isdir(os.path.join(discovery.home, "lib")):
        sdk_lib = os.path.join(discovery.home, "lib")
    sdk_bin = os.path.join(discovery.home, "bin")
    include_paths = (os.path.join(corex_root, "include"), sdk_include,
                     os.path.join(cuda_root, "include"), cuda_root)
    sdk_flags = " " + " ".join("-I" + shlex.quote(path) for path in include_paths)
    common_flags = remove_flags(
        config.cc_flags, ("-fopenmp", "-DIS_CUDA", "-DHAS_CUDA"))
    common_flags += " -DHAS_ACCELERATOR -DIS_COREX -DJT_DEFAULT_PARA_OPT_LEVEL=4 "
    link_flags = (" -L" + shlex.quote(sdk_lib)
                  + " -Wl,-rpath," + shlex.quote(sdk_lib) + " -lcudart ")
    # Corex uses the CUDA-compatible source dialect for generated kernels.  Keep
    # the CUDA ABI markers on the device compile only; the host translation
    # units intentionally remain Corex-only (the two flag domains are not
    # interchangeable).
    device_flags = (sdk_flags + " -x cu -Ofast -DHAS_CUDA -DIS_CUDA "
                    "-DNO_ATOMIC64 -Wno-c++11-narrowing ")
    driver = os.path.join(cuda_root, "runtime", "driver.cc")
    abi_flags = sdk_flags + " -DHAS_CUDA -DIS_CUDA "
    sources = (
        BuildSource(driver, flags=abi_flags,
                    compiler=discovery.compiler_path),
        BuildSource(os.path.join(cuda_root, "runtime", "nan_checker.cc"),
                    flags=abi_flags, compiler=discovery.compiler_path),
        BuildSource(os.path.join(cuda_root, "kernels", "debug", "nan_checker.cu"),
                    language="cuda", flags=abi_flags + device_flags,
                    compiler=discovery.compiler_path),
        BuildSource(os.path.join(corex_root, "runtime", "corex_backend.cc"),
                    compiler=discovery.compiler_path),
    )
    return config.evolve(
        backend="corex", has_corex=True, has_accelerator=True,
        has_cuda=False, is_cuda=False, has_acl=False, has_rocm=False,
        cc_path=discovery.compiler_path, nvcc_path=discovery.compiler_path,
        cc_type="clang", cc_flags=common_flags,
        kernel_flags=remove_flags(config.kernel_flags, ("-fopenmp",)),
        nvcc_flags=convert_nvcc_flags(common_flags + device_flags),
        backend_sources=config.backend_sources + sources,
        backend_link_flags=config.backend_link_flags + link_flags,
        kernel_compiler=discovery.compiler_path, kernel_language="cuda",
        kernel_compile_flags=device_flags,
        kernel_flag_filter=("--extended-lambda", "--expt-relaxed-constexpr"),
        kernel_source_suffix=".cc", kernel_device_link=False,
        # Reuse the reviewed CUDA kernel providers.  An empty source-root
        # tuple would make Corex silently lose builtin accelerator overrides
        # and compile every op through the generic path.
        kernel_source_roots=(os.path.join(cuda_root, "kernels", "core"),),
        convert_nvcc_flags=convert_nvcc_flags,
        environment=dict(config.environment, use_cutt="0"),
        resources=dict(
            config.resources, corex_home=discovery.home,
            cuda_home=discovery.home, cuda_bin=sdk_bin, cuda_dir=sdk_bin,
            cuda_include=sdk_include, cuda_lib=sdk_lib,
            cuda_include_dirs=(sdk_include,), cuda_lib_dirs=(sdk_lib, sdk_bin)),
    )


def convert_nvcc_flags(flags):
    """Adapt flags only; filenames, source text and linker wrappers are not inputs."""
    unsupported = {"--extended-lambda", "--expt-relaxed-constexpr"}
    return " ".join(shlex.quote(flag) for flag in shlex.split(flags)
                    if flag not in unsupported)


def install_extern(context):
    return False


def post_process(context):
    return context.config
