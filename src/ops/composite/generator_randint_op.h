#pragma once
#include "core/op.h"

namespace jittor {
struct GeneratorRandintOp : Op {
    Var* output;
    int64 low, high, seed, offset;
    GeneratorRandintOp(NanoVector shape, int64 low, int64 high, int64 seed, int64 offset, NanoString dtype=ns_int64);
    const char* name() const override { return "generator_randint"; }
    DECLARE_jit_run;
};
}
