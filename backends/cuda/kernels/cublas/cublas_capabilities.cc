#include "core/var.h"
#include "ops/composite/op_capability.h"

#ifndef JIT
namespace jittor {
namespace {
bool supports_matmul(Var* a, Var* b, bool, bool) {
    return a->dtype().is_float() && a->dtype() == b->dtype();
}

// Every float width cuBLAS has a gemm for, which is what `supports_matmul`
// above accepts (`is_float()` plus matching operand dtypes). Declared so the
// CPU/accelerator gap is queryable rather than implied: oneDNN's CPU matmul is
// f32-only, so a f64/f16/bf16 matmul is a real per-backend difference.
RegisterOpCapability<VarPtr, Var*, Var*, bool, bool> matmul(
    accelerator_backend_id(), OpCapability::Matmul, "cublas_matmul", supports_matmul,
    {"float32", "float64", "float16", "bfloat16"});
}
} // namespace jittor
#endif
