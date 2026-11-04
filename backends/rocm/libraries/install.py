"""Publish only SDK providers that have compiled native implementations."""

from pathlib import Path

from ..build import library_build


def _hipblas_matmul(a, b, trans_a=False, trans_b=False):
    from jittor._runtime.backend_libraries import get_library_ops
    return get_library_ops("hipblas").hipblas_matmul(a, b, trans_a, trans_b)


def _hipblas_batched_matmul(a, b, trans_a=False, trans_b=False):
    from jittor._runtime.backend_libraries import get_library_ops
    return get_library_ops("hipblas").hipblas_batched_matmul(a, b, trans_a, trans_b)


def _batched_amp_changes_dtype():
    from jittor.nn.functional._amp import prefers_half
    return prefers_half()


def _supports_batched_matmul(a, b, trans_a=False, trans_b=False):
    # Strided-batch GEMM handles dense, equally sized batches. Other batch
    # broadcasts retain the existing same-device contraction route.
    return (a.ndim >= 3 and a.ndim == b.ndim and a.dtype == b.dtype
            and a.shape[:-2] == b.shape[:-2] and not _batched_amp_changes_dtype())


def _supports_matmul(a, b, trans_a=False, trans_b=False):
    return a.ndim == 2 and b.ndim == 2 and a.dtype == b.dtype


def _rocprim_scan(x, reverse=False):
    from jittor._runtime.backend_libraries import get_library_ops
    return get_library_ops("rocprim").rocprim_cumsum(x, reverse)


def install_libraries(context):
    """Build native sources without CUDA aliases or source transformation."""
    root = Path(__file__).resolve().parents[1]
    specs = tuple(library_build(context.config.resources["rocm_home"], root, name)
                  for name in ("hipblas", "rocprim", "hiprand"))
    modules = []
    for spec in specs:
        module = context.compile_custom_ops(
            spec.sources, return_module=True, extra_flags=spec.extra_flags,
            backend="accelerator")
        modules.append((spec.name, module))
    for name, module in modules:
        context.publish_library(name, module)
    return tuple(modules)


def install_kernels():
    """Run after core loading; native domain imports may follow this step."""
    from jittor._runtime.dispatch import register_kernel
    register_kernel("matmul", "rocm", _hipblas_matmul,
                    dtypes={"float32", "float64"}, supports=_supports_matmul, priority=100)
    # A typed dgemm entry alone does not establish full-precision FP64 math
    # on every SDK; admit only the independently validated FP32 batch path.
    register_kernel("batched_matmul", "rocm", _hipblas_batched_matmul,
                    dtypes={"float32"}, supports=_supports_batched_matmul, priority=100)
    register_kernel("misc.scan_2d", "rocm", _rocprim_scan,
                    dtypes={"float32", "float64", "int32", "int64"}, priority=100)
