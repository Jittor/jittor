#pragma once
#include "core/op.h"
namespace jittor {
struct GeneratorRandpermOp : Op {
    Var* output;
    int64 n, seed, offset;
    GeneratorRandpermOp(int64 n, int64 seed, int64 offset, NanoString dtype=ns_int64);
    const char* name() const override { return "generator_randperm"; }
    DECLARE_jit_run;
};
}
