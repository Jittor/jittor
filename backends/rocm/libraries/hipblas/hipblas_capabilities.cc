#include "var.h"
#include "ops/op_capability.h"

namespace jittor {
namespace {
bool supports_matmul(Var* a, Var* b, bool, bool) {
    return a->shape.size() == 2 && b->shape.size() == 2
        && a->dtype() == b->dtype()
        && (a->dtype() == ns_float32 || a->dtype() == ns_float64);
}

// Narrower than cuBLAS: `supports_matmul` above accepts f32 and f64 only, so
// f16/bf16 matmul on ROCm falls back to the generic kernel. Declared rather
// than left inside the predicate -- no ROCm hardware is available here, and a
// declaration is checkable by reading it where a function pointer is not.
RegisterOpCapability<VarPtr, Var*, Var*, bool, bool> matmul(
    BackendId::Rocm, OpCapability::Matmul, "hipblas_matmul", supports_matmul,
    {"float32", "float64"});
} // namespace
} // namespace jittor
