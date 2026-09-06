#include "var.h"
#include "ops/op_capability.h"

namespace jittor {
namespace {
bool supports_matmul(Var* a, Var* b, bool, bool) {
    return a->shape.size() == 2 && b->shape.size() == 2
        && a->dtype() == b->dtype()
        && (a->dtype() == ns_float32 || a->dtype() == ns_float64);
}

RegisterOpCapability<VarPtr, Var*, Var*, bool, bool> matmul(
    BackendId::Rocm, OpCapability::Matmul, "hipblas_matmul", supports_matmul);
} // namespace
} // namespace jittor
