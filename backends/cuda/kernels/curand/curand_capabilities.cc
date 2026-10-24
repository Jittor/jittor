#include "core/var.h"
#include "ops/composite/op_capability.h"
#include "runtime/random_generator.h"

#ifndef JIT
namespace jittor {
namespace {
// Unsupported dtype/type errors remain in the constructor: the generic
// accelerator random path is not a numerical fallback for rejected requests.
RegisterOpCapability<VarPtr, NanoVector, NanoString, NanoString, RandomGenerator*> random(
    accelerator_backend_id(), OpCapability::Random, "curand_random");
const bool integer_registered = [] {
    if (accelerator_backend_id() == BackendId::Cuda) {
        RegisterOpCapability<VarPtr, NanoVector, int64, int64, bool, RandomGenerator*> integer_random(
            BackendId::Cuda, OpCapability::RandomInteger, "curand_random_int", nullptr, {"int64"});
    }
    return true;
}();
}
} // namespace jittor
#endif
