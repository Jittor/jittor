// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "curand_wrapper.h"
#include "init.h"
#include "misc/cuda_flags.h"

namespace jittor {

curandGenerator_t gen;
static bool curand_gen_created = false;

// See cublas_shutdown: report, never raise, and idempotent.
void curand_shutdown() {
    if (!curand_gen_created) return;
    curand_gen_created = false;
    peekCudaErrorsAlways( curandDestroyGenerator(gen) );
    gen = nullptr;
    LOGv << "curandDestroy finished";
}

struct curand_initer {

inline curand_initer() {
    if (!get_device_count()) return;
    checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    curand_gen_created = true;
    add_set_seed_callback([](int seed) {
        // The callback list is a separate global: nothing orders it against
        // this generator at exit, so a set_seed after shutdown must not run.
        if (!gen) return;
        checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, seed) );
        // The seed alone does not rewind the generator: it keeps its position
        // in the sequence, so re-seeding with the same value after drawing
        // continues from where it left off and jt.set_seed() does not
        // reproduce. set_seed() resets the CPU side's offset for the same
        // reason; this is the CUDA half of it.
        checkCudaErrors( curandSetGeneratorOffset(gen, 0) );
    });
    LOGv << "curandCreate finished";
}

inline ~curand_initer() {
    curand_shutdown();
}

} init_;

} // jittor
