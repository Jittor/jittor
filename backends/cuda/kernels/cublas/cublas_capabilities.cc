#include "var.h"
#include "ops/op_capability.h"

#ifndef JIT
namespace jittor {
namespace {
bool supports_matmul(Var* a, Var* b, bool, bool) {
    return a->dtype().is_float() && a->dtype() == b->dtype();
}

RegisterOpCapability<VarPtr, Var*, Var*, bool, bool> matmul(
    accelerator_backend_id(), OpCapability::Matmul, "cublas_matmul", supports_matmul);
}
} // namespace jittor
#endif
