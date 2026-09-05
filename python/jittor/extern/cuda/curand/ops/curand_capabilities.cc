#include "var.h"
#include "ops/op_capability.h"

#ifndef JIT
namespace jittor {
namespace {
// Unsupported dtype/type errors remain in the constructor: the generic
// accelerator random path is not a numerical fallback for rejected requests.
RegisterOpCapability<VarPtr, NanoVector, NanoString, NanoString> random(
    accelerator_backend_id(), OpCapability::Random, "curand_random");
}
} // namespace jittor
#endif
