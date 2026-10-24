#pragma once
#include "core/op.h"
#include "runtime/random_generator.h"

namespace jittor {

// Integer-only provider: existing floating provider and stream are unchanged.
struct CurandRandomIntOp : Op {
    Var* output;
    uint64 philox_seed = 0;
    uint64 philox_offset = 0;
    uint64 range = 0;
    uint64 base_bits = 0;
    int raw_per_value = 0;
    bool full_range = false;
    CurandRandomIntOp(NanoVector shape, int64 low, int64 high, bool high_none, RandomGenerator* generator);
    const char* name() const override { return "curand_random_int"; }
    DECLARE_jit_run;
};

} // namespace jittor
