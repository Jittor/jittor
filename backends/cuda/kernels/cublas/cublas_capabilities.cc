#include "core/var.h"
#include "ops/composite/op_capability.h"

#ifndef JIT
namespace jittor {
namespace {
bool supports_matmul(Var* a, Var* b, bool, bool) {
#ifndef IS_ROCM
    if (a->dtype().is_complex()) return a->dtype() == b->dtype();
#endif
    return a->dtype().is_float() && a->dtype() == b->dtype();
}

// Every float width cuBLAS has a gemm for, which is what `supports_matmul`
// above accepts (`is_float()` plus matching operand dtypes), and on CUDA the
// two complex widths (cublasGemmEx with CUDA_C_32F/CUDA_C_64F). Declared so
// the CPU/accelerator gap is queryable rather than implied: oneDNN's CPU
// matmul is f32-only, so a f64/f16/bf16/complex matmul is a real per-backend
// difference.
RegisterOpCapability<VarPtr, Var*, Var*, bool, bool> matmul(
    accelerator_backend_id(), OpCapability::Matmul, "cublas_matmul", supports_matmul,
#ifndef IS_ROCM
    {"float32", "float64", "float16", "bfloat16", "complex64", "complex128"});
#else
    {"float32", "float64", "float16", "bfloat16"});
#endif
}
} // namespace jittor
#endif
