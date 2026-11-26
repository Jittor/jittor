#include "core/var.h"
#include "ops/composite/op_capability.h"

namespace jittor {
namespace {
bool supports_matmul(Var* a, Var* b, bool, bool) {
    return a->shape.size() == 2 && b->shape.size() == 2
        && a->dtype() == b->dtype()
        && dtype_infer(a->dtype(), b->dtype()) == a->dtype()
        && (a->dtype() == ns_float32 || a->dtype() == ns_float64
            || a->dtype() == ns_float16 || a->dtype() == ns_bfloat16);
}

RegisterOpCapability<VarPtr, Var*, Var*, bool, bool> matmul(
    BackendId::Rocm, OpCapability::Matmul, "hipblas_matmul", supports_matmul,
    {"float32", "float64", "float16", "bfloat16"});
} // namespace
} // namespace jittor
