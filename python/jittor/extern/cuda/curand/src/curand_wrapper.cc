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
// One generator per device; the global is the current device's and the
// seed callback keeps every generator in step.
static vector<curandGenerator_t> gens;
static int curand_seed = -1;

static void curand_switch_device(int device) {
    if ((int)gens.size() <= device) gens.resize(device+1, nullptr);
    if (!gens[device]) {
        checkCudaErrors( curandCreateGenerator(&gens[device], CURAND_RNG_PSEUDO_DEFAULT) );
        if (curand_seed >= 0) {
            checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gens[device], curand_seed) );
            checkCudaErrors( curandSetGeneratorOffset(gens[device], 0) );
        }
    }
    gen = gens[device];
}

struct curand_initer {

inline curand_initer() {
    if (!get_device_count()) return;
    add_device_switch_hook(curand_switch_device);
    add_set_seed_callback([](int seed) {
        curand_seed = seed;
        for (auto g : gens) {
            if (!g) continue;
            checkCudaErrors( curandSetPseudoRandomGeneratorSeed(g, seed) );
            checkCudaErrors( curandSetGeneratorOffset(g, 0) );
        }
    });
    LOGv << "curandCreate finished";
}

inline ~curand_initer() {
    if (!get_device_count()) return;
    for (auto g : gens)
        if (g) checkCudaErrors( curandDestroyGenerator(g) );
    gens.clear();
    LOGv << "curandDestroy finished";
}

} init_;

} // jittor
