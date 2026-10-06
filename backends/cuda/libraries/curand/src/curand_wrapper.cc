// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "stream_compat.h"
#include "curand_wrapper.h"
#include "runtime/init.h"
#include "runtime/device.h"
#include "runtime/cuda_streams.h"

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
// How far each device's generator has advanced since it was last seeded or
// restored, in `curandSetGeneratorOffset` units.
//
// The offset is what makes a CUDA RNG checkpoint possible: cuRAND can set a
// seed and an offset but will not tell you the offset it is at, so unless
// somebody counts, a resumed run restarts the sequence instead of continuing
// it -- silently. Measured on this box against CURAND_RNG_PSEUDO_DEFAULT:
//
//   uniform float32/float64   n elements cost n
//   normal  float32/float64   n elements cost n/2
//
// and a mixed history costs the sum of its parts. Verified by drawing a
// history, drawing a continuation, then reseeding, setting the summed offset
// and drawing again: the values match exactly, for a continuation of every
// one of the four kinds.
static vector<int64> curand_offsets;

void curand_advance(int device, int64 cost) {
    if (device < 0) return;
    if ((int)curand_offsets.size() <= device) curand_offsets.resize(device + 1, 0);
    curand_offsets[device] += cost;
}

int64 curand_generator_offset(int device) {
    return device >= 0 && device < (int)curand_offsets.size()
        ? curand_offsets[device] : 0;
}

int curand_generator_seed() { return curand_last_seed; }

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
        // Every library handle must agree with jittor's own launches on the
        // stream; see `compute_stream` in backends/cuda/runtime/driver.cc.
        // `cudaStreamPerThread` does not synchronise with the legacy stream,
        // so a handle left on the default would race with no error.
        checkCudaErrors(curandSetStream(gens[device], cudaStreamPerThread));
        if (curand_last_seed >= 0) curand_seed_generator(gens[device], curand_last_seed);
    }
    gen = gens[device];
}

// Put a device's generator back where a checkpoint left it.
//
// Seeding alone is not a restore: it rewinds to the start of the sequence, so
// the resumed run draws what the *original* run drew first rather than what it
// was about to draw. Seed, then set the offset the checkpoint recorded.
void curand_restore_state(int device, int seed, int64 offset) {
    CHECK(device >= 0) << "curand restore needs a device";
    int previous = current_device();
    if (device != previous) set_current_device(device);
    curand_switch_device(device);
    checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gens[device], seed) );
    checkCudaErrors( curandSetGeneratorOffset(gens[device], (unsigned long long)offset) );
    if ((int)curand_offsets.size() <= device) curand_offsets.resize(device + 1, 0);
    curand_offsets[device] = offset;
    curand_last_seed = seed;
    if (device != previous) set_current_device(previous);
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
        // Seeding rewinds every generator to offset 0 (see
        // curand_seed_generator), so the count has to rewind with it.
        for (auto& offset : curand_offsets) offset = 0;
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
