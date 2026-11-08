#pragma once
#include "core/op.h"
#include "runtime/random_generator.h"

namespace jittor {

// Bounded integer draws use the canonical native Generator when provided.
// Explicit seed/offset callers retain the stateless Philox entry point.
struct GeneratorRandintOp : Op {
    Var* output;
    shared_ptr<RandomGeneratorState> gen;
    int64 low, high, seed, offset;
    GeneratorRandintOp(NanoVector shape, NanoString dtype, int64 low, int64 high, int64 seed, int64 offset, RandomGenerator* generator=nullptr);
    const char* name() const override { return "generator_randint"; }
    DECLARE_jit_run;
};

} // namespace jittor
