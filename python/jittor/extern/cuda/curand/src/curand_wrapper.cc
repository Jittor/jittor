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
#include "misc/cuda_streams.h"

namespace jittor {

curandGenerator_t gen;
// One generator per device; the global is the current device's. A generator
// draws from the device it was created on, so a single global one would make
// jt.rand() on device 1 either fail or fill device-0 memory.
static vector<curandGenerator_t> gens;
static vector<uint64> curand_stream_binds;
// The last seed, replayed onto a generator created after set_seed so every
// device answers the same seed the same way.
static int curand_last_seed = -1;

static void curand_seed_generator(curandGenerator_t g, int seed) {
    checkCudaErrors( curandSetPseudoRandomGeneratorSeed(g, seed) );
    // The seed alone does not rewind the generator: it keeps its position
    // in the sequence, so re-seeding with the same value after drawing
    // continues from where it left off and jt.set_seed() does not
    // reproduce. set_seed() resets the CPU side's offset for the same
    // reason; this is the CUDA half of it.
    checkCudaErrors( curandSetGeneratorOffset(g, 0) );
}

curandGenerator_t curand_bind_stream() {
    int device = current_device();
    checkCudaErrors(curandSetStream(gen, cuda_compute_stream(device)));
    if ((int)curand_stream_binds.size() <= device)
        curand_stream_binds.resize(device + 1);
    curand_stream_binds[device]++;
    return gen;
}

uint64 curand_stream_bind_count(int device) {
    return device >= 0 && device < (int)curand_stream_binds.size()
        ? curand_stream_binds[device] : 0;
}

static void curand_switch_device(int device) {
    if ((int)gens.size() <= device) gens.resize(device+1, nullptr);
    if (!gens[device]) {
        checkCudaErrors( curandCreateGenerator(&gens[device], CURAND_RNG_PSEUDO_DEFAULT) );
        if (curand_last_seed >= 0) curand_seed_generator(gens[device], curand_last_seed);
    }
    gen = gens[device];
}

// See cublas_shutdown: report, never raise, and idempotent.
void curand_shutdown() {
    if (gens.empty()) return;
    for (auto g : gens)
        if (g) peekCudaErrorsAlways( curandDestroyGenerator(g) );
    gens.clear();
    curand_stream_binds.clear();
    gen = nullptr;
    LOGv << "curandDestroy finished";
}

struct curand_initer {

inline curand_initer() {
    if (!get_device_count()) return;
    add_device_switch_hook(curand_switch_device);
    add_set_seed_callback([](int seed) {
        curand_last_seed = seed;
        // The callback list is a separate global: nothing orders it against
        // these generators at exit, so a set_seed after shutdown must not run.
        for (auto g : gens)
            if (g) curand_seed_generator(g, seed);
    });
    LOGv << "curandCreate finished";
}

inline ~curand_initer() {
    curand_shutdown();
}

} init_;

} // jittor
