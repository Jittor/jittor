#pragma once
#include "core/op.h"

namespace jittor {

// Stateless, explicitly-owned integer draws.  The seed and element offset
// are supplied by the frontend Generator, so this op never touches Jittor's
// process-wide RNG stream.
struct GeneratorRandintOp : Op {
    Var* output;
    int64 low, high, seed, offset;
    GeneratorRandintOp(NanoVector shape, NanoString dtype, int64 low, int64 high, int64 seed, int64 offset);
    const char* name() const override { return "generator_randint"; }
    DECLARE_jit_run;
};

} // namespace jittor
