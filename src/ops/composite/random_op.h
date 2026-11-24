// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"
#include "runtime/random_generator.h"
#include <random>

namespace jittor {

struct RandomOp : Op {
    // Accelerators reach this op either through their Random capability op
    // (CUDA-family curand_random) or, when they run `random` themselves,
    // through a native callback their composer installs on this entry.
    static constexpr uint32 backend_mask = OpBackendAny;
    Var* output;
    NanoString type;
    // Non-null: draws consume (and advance) THIS generator's own
    // independent, device-native RNG state instead of the process-global
    // one -- captured as a shared_ptr (not the raw RandomGenerator* passed
    // in) so the state stays alive for this op's lazy execution regardless
    // of the Python-side Generator wrapper's own lifetime.
    shared_ptr<RandomGeneratorState> gen;
    // Immutable draw reservation: execution and rematerialization never
    // consult the owner's subsequently reseeded/restored/advanced state.
    bool from_acl_generator = false;
    uint64 acl_seed = 0;
    int64 acl_offset = 0;
    int64 acl_span = 0;
    bool torch_normal;
    bool rematerialization_preserve_rng = false;
    shared_ptr<std::default_random_engine> rematerialization_rng;
    shared_ptr<TorchCpuNormal> rematerialization_torch_rng;

    RandomOp(NanoVector shape, NanoString dtype=ns_float32, NanoString type=ns_uniform,
             RandomGenerator* generator=nullptr, bool torch_normal=false);

    const char* name() const override { return "random"; }
    DECLARE_jit_run;
};

} // jittor
