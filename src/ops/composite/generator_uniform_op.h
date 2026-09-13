#pragma once
#include "core/op.h"
namespace jittor {
struct GeneratorUniformOp : Op {
    Var* output;
    int64 seed, offset, precision_bits;
    double low, high;
    GeneratorUniformOp(NanoVector shape, NanoString dtype, double low, double high, int64 seed, int64 offset, int64 precision_bits);
    const char* name() const override { return "generator_uniform"; }
    DECLARE_jit_run;
};
}
