#include "core/var.h"
#include "ops/composite/op_capability.h"

#ifndef JIT
namespace jittor {
namespace {
RegisterOpCapability<VarPtr, Var*, NanoVector> transpose(
    accelerator_backend_id(), OpCapability::Transpose, "cutt_transpose");
}
} // namespace jittor
#endif
