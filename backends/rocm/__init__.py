"""Native HIP/ROCm backend provider.

This provider only declares native HIP runtime and the independently owned
hipBLAS/rocPRIM libraries. It deliberately does not rewrite Jittor sources or
pretend that MIOpen/RCCL are available.
"""
import os
import shutil
from pathlib import Path
from jittor_utils.build_config import BuildConfig, BuildContext, BuildSource
from jittor_utils.backend_resources import backend_root


def _rocm_home():
    return os.path.abspath(os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH") or "/opt/rocm")


def configure(context: BuildContext) -> BuildConfig:
    config = context.config
    home = _rocm_home()
    hipcc = os.environ.get("hipcc_path") or os.path.join(home, "bin", "hipcc")
    hipcc = shutil.which(hipcc) or (hipcc if os.path.isfile(hipcc) else None)
    if not hipcc:
        raise RuntimeError("ROCm was selected but hipcc is unavailable")
    include = os.path.join(home, "include")
    if not os.path.isdir(include):
        raise RuntimeError("ROCm development headers are missing: " + include)
    root = Path(__file__).resolve().parent
    cuda_root = Path(backend_root(context.config.jittor_path, "cuda"))
    flags = config.cc_flags + " -DHAS_CUDA -DIS_ROCM -DHAS_ACCELERATOR -I" + str(include) + " -I" + str(root / "include") + " -I" + str(cuda_root / "include")
    runtime = root / "runtime" / "driver.cc"
    return config.evolve(backend="rocm", has_rocm=True, has_cuda=True, is_cuda=False,
        has_accelerator=True, hipcc_path=hipcc, nvcc_path=hipcc, cc_flags=flags,
        nvcc_flags=flags + " -x hip", kernel_compiler=hipcc, kernel_language="hip",
        kernel_compile_flags=" -I" + str(include) + " -I" + str(cuda_root / "include"),
        backend_sources=config.backend_sources + (BuildSource(str(runtime), language="hip", flags=flags),),
        resources=dict(config.resources, rocm_home=home, rocm_runtime=runtime),
        environment=dict(config.environment, use_mkl="0"))


def install_extern(context: BuildContext) -> bool:
    from .libraries import install_libraries
    return bool(install_libraries(context))


def post_process(context: BuildContext) -> None:
    from .libraries import install_kernels
    install_kernels()
