// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"
#include "runtime/random_generator.h"

namespace jittor {

struct RandomOp : Op {
    // Accelerators reach this op either through their Random capability op
    // (CUDA-family curand_random) or, when they run `random` themselves,
    // through a native callback their composer installs on this entry.
    static constexpr uint32 backend_mask = OpBackendAny;
    Var* output;
    NanoString type;
    // CPU integer distribution parameters. This changes RandomOp's ABI;
    // rebuild the core, generated wrappers, and JIT/extension consumers.
    bool integer_random = false;
    int64 integer_low = 0;
    int64 integer_high = 0; // inclusive, unlike the public exclusive bound
    // Non-null: draws consume (and advance) THIS generator's own
    // independent, device-native RNG state instead of the process-global
    // one -- captured as a shared_ptr (not the raw RandomGenerator* passed
    // in) so the state stays alive for this op's lazy execution regardless
    // of the Python-side Generator wrapper's own lifetime.
    shared_ptr<RandomGeneratorState> gen;
    RandomOp(NanoVector shape, NanoString dtype=ns_float32, NanoString type=ns_uniform, RandomGenerator* generator=nullptr);

    // @pybind(random_int64)
    RandomOp(NanoVector shape, int64 low, int64 high, bool high_none, RandomGenerator* generator);

    const char* name() const override { return "random"; }
    DECLARE_jit_run;
};

} // jittor
