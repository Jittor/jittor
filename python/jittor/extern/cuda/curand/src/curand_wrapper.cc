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
// One generator per device; `gen` always names the current device's. Every
// generator is seeded the same way so a seed is reproducible on any device.
static curandGenerator_t gens[64];
static int last_seed = -1;

static void seed_generator(curandGenerator_t g, int seed) {
    checkCudaErrors( curandSetPseudoRandomGeneratorSeed(g, seed) );
    checkCudaErrors( curandSetGeneratorOffset(g, 0) );
}

struct curand_initer {

inline curand_initer() {
    if (!get_device_count()) return;
    register_device_switch_hook([](int device) {
        if (!gens[device]) {
            checkCudaErrors( curandCreateGenerator(&gens[device], CURAND_RNG_PSEUDO_DEFAULT) );
            if (last_seed >= 0) seed_generator(gens[device], last_seed);
        }
        gen = gens[device];
    });
    add_set_seed_callback([](int seed) {
        last_seed = seed;
        for (auto g : gens)
            if (g) seed_generator(g, seed);
        return;
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
    if (!get_device_count()) return;
    for (auto g : gens)
        if (g) checkCudaErrors( curandDestroyGenerator(g) );
    LOGv << "curandDestroy finished";
}

} init_;

} // jittor
