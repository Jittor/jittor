#pragma once
#include "core/op.h"
#include "runtime/random_generator.h"

namespace jittor {

struct HiprandRandomOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var* output;
    bool normal;
    uint64 seed, offset;

    HiprandRandomOp(NanoVector shape, NanoString dtype=ns_float32, NanoString type=ns_uniform, RandomGenerator* generator=nullptr);
    const char* name() const override { return "hiprand_random"; }
    void run() override;
};

} // namespace jittor
