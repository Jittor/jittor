"""Publish only SDK providers that have compiled native implementations."""

from pathlib import Path

from ..build import library_build


def _matmul_operands(a, b):
    import jittor as jt
    if jt.flags.amp_reg and a.dtype.is_float() and b.dtype.is_float():
        dtype = jt.binary_dtype_infer("multiply", a.dtype, b.dtype)
        if a.dtype != dtype or b.dtype != dtype:
            a, b = a.cast(dtype), b.cast(dtype)
    return a, b


def _hipblas_matmul(a, b, trans_a=False, trans_b=False):
    from jittor._runtime.backend_libraries import get_library_ops
    a, b = _matmul_operands(a, b)
    return get_library_ops("hipblas").hipblas_matmul(a, b, trans_a, trans_b)


def _hipblas_batched_matmul(a, b, trans_a=False, trans_b=False):
    from jittor._runtime.backend_libraries import get_library_ops
    a, b = _matmul_operands(a, b)
    return get_library_ops("hipblas").hipblas_batched_matmul(a, b, trans_a, trans_b)


def _supports_batched_matmul(a, b, trans_a=False, trans_b=False):
    # Strided-batch GEMM handles dense, equally sized batches. Other batch
    # broadcasts retain the existing same-device contraction route.
    return (a.ndim >= 3 and a.ndim == b.ndim and a.dtype == b.dtype
            and a.shape[:-2] == b.shape[:-2])


def _supports_matmul(a, b, trans_a=False, trans_b=False):
    if b.ndim != 2 or a.dtype != b.dtype:
        return False
    if a.ndim == 2:
        return True
    from jittor._core.dtypes import dtype_name
    return (a.ndim > 2 and not trans_a and dtype_name(a.dtype) in ("float32", "float16", "bfloat16"))


def _hipblas_linear_bias(x, weight, bias):
    from jittor._runtime.core_api import _stop_grad_outputs
    return _stop_grad_outputs(_hipblas_matmul(x, weight, False, True) + bias)


def _supports_linear_bias(x, weight, bias):
    import jittor as jt
    from jittor._core.dtypes import dtype_name
    from jittor._runtime.core_api import _output_requires_grad
    return (all(isinstance(v, jt.Var) and dtype_name(v.dtype) == "float32"
                and v._storage_is_contiguous() for v in (x, weight, bias))
            and not jt.flags.amp_reg and not _output_requires_grad(x, weight, bias)
            and x.ndim >= 2 and weight.ndim == 2 and bias.ndim == 1
            and x.numel() > 0 and weight.numel() > 0
            and x.shape[-1] == weight.shape[1] and bias.shape[0] == weight.shape[0])


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
    from jittor._runtime.dispatch import register_kernel, native_rule
    native_rule("rocm_linear_bias")(_supports_linear_bias)
    register_kernel("matmul", "rocm", _hipblas_matmul,
                    dtypes={"float32", "float64", "float16", "bfloat16"}, supports=_supports_matmul, priority=100)
    register_kernel("int8_matmul", "rocm", _hipblas_matmul,
                    dtypes={"int8"}, supports=_supports_matmul, priority=100)
    register_kernel("linear_bias_accumulator", "rocm", _hipblas_linear_bias,
                    dtypes={"float32"}, supports=_supports_linear_bias, priority=100)
    # A typed dgemm entry alone does not establish full-precision FP64 math
    # on every SDK; admit only the independently validated FP32 batch path.
    register_kernel("batched_matmul", "rocm", _hipblas_batched_matmul,
                    dtypes={"float32", "float16", "bfloat16"}, supports=_supports_batched_matmul, priority=100)
    register_kernel("misc.scan_2d", "rocm", _rocprim_scan,
                    dtypes={"float32", "float64", "int32", "int64"}, priority=100)
