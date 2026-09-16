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
#include "core/var_holder.h"
#include <limits>
#include <sstream>

namespace jittor {

curandGenerator_t gen;
// One generator per device; the global is the current device's. A generator
// draws from the device it was created on, so a single global one would make
// jt.rand() on device 1 either fail or fill device-0 memory.
static vector<curandGenerator_t> gens;
static vector<uint64> curand_stream_binds;
static vector<uint64> curand_seeds;
static vector<uint64> curand_offsets;
static vector<uint8> curand_snapshot_safe;
// The last seed, replayed onto a generator created after set_seed so every
// device answers the same seed the same way.
static uint64 curand_last_seed = 0;
static bool curand_has_seed = false;

static void curand_seed_generator(curandGenerator_t g, uint64 seed) {
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
    if ((int)curand_seeds.size() <= device) {
        curand_seeds.resize(device+1);
        curand_offsets.resize(device+1);
        curand_snapshot_safe.resize(device+1, 1);
    }
    if (!gens[device]) {
        checkCudaErrors( curandCreateGenerator(&gens[device], CURAND_RNG_PSEUDO_DEFAULT) );
        if (curand_has_seed) curand_seed_generator(gens[device], curand_last_seed);
        curand_seeds[device] = curand_last_seed;
        curand_offsets[device] = 0;
        curand_snapshot_safe[device] = 1;
    }
    gen = gens[device];
}

void curand_check_offset_advance(uint64 count) {
    int device = current_device();
    USER_CHECK(!curand_snapshot_safe[device] ||
        curand_offsets[device] <= std::numeric_limits<uint64>::max() - count)
        << "cuRAND RNG offset exceeds uint64 range";
}

void curand_advance_offset(uint64 count, bool snapshot_safe) {
    if (!count) return;
    int device = current_device();
    if (!snapshot_safe) curand_snapshot_safe[device] = 0;
    if (curand_snapshot_safe[device]) curand_offsets[device] += count;
}

struct CurandSnapshot {
    uint64 seed;
    uint64 offset;
};

static int curand_version() {
    int version;
    checkCudaErrors(curandGetVersion(&version));
    return version;
}

static uint64 parse_uint64(const string& token) {
    USER_CHECK(!token.empty()) << "invalid cuRAND RNG state integer";
    uint64 value = 0;
    for (char c : token) {
        USER_CHECK(c >= '0' && c <= '9') << "invalid cuRAND RNG state integer";
        uint64 digit = c - '0';
        USER_CHECK(value <= (std::numeric_limits<uint64>::max() - digit) / 10)
            << "cuRAND RNG state integer exceeds uint64 range";
        value = value * 10 + digit;
    }
    return value;
}

static CurandSnapshot parse_curand_state(const string& state) {
    USER_CHECK(!state.empty() && state.size() <= 256) << "invalid cuRAND RNG state size";
    std::istringstream input(state);
    string version, seed, offset;
    int library_version;
    USER_CHECK(bool(input >> version >> library_version >> seed >> offset)
        && version == "JITTOR_CURAND_XORWOW_U32_V1" && library_version == curand_version())
        << "invalid or incompatible cuRAND RNG state";
    input >> std::ws;
    USER_CHECK(input.eof()) << "trailing data in cuRAND RNG state";
    return {parse_uint64(seed), parse_uint64(offset)};
}

struct CurandDeviceScope {
    int previous;
    explicit CurandDeviceScope(int device) : previous(current_device()) {
        USER_CHECK(device >= 0 && device < get_device_count()) << "invalid cuRAND RNG device";
        set_current_device(device);
    }
    ~CurandDeviceScope() { set_current_device(previous); }
};

void curand_validate_rng_state(const string& state) {
    parse_curand_state(state);
}

string curand_get_rng_state(int device) {
    sync_all(true);
    CurandDeviceScope scope(device);
    sync_devices(0);
    // Partial normal/double batches advance XORWOW's 4096 subsequences
    // unevenly; the public Host API cannot restore them with one offset.
    USER_CHECK(curand_snapshot_safe[device])
        << "complete CUDA RNG state is unsupported after normal or float64 draws; "
        << "cuRAND XORWOW checkpoints require only float32 uniform draws since seed/restore";
    std::ostringstream output;
    output << "JITTOR_CURAND_XORWOW_U32_V1 " << curand_version() << ' '
           << curand_seeds[device] << ' ' << curand_offsets[device] << '\n';
    return output.str();
}

void curand_set_rng_state(int device, const string& state) {
    auto snapshot = parse_curand_state(state);
    USER_CHECK(device >= 0 && device < get_device_count()) << "invalid cuRAND RNG device";
    sync_all(true);
    CurandDeviceScope scope(device);
    sync_devices(0);
    checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, snapshot.seed));
    checkCudaErrors(curandSetGeneratorOffset(gen, snapshot.offset));
    curand_seeds[device] = snapshot.seed;
    curand_offsets[device] = snapshot.offset;
    curand_snapshot_safe[device] = 1;
}

void curand_manual_seed(int device, uint64 seed) {
    sync_all(true);
    CurandDeviceScope scope(device);
    sync_devices(0);
    curand_seed_generator(gen, seed);
    curand_seeds[device] = seed;
    curand_offsets[device] = 0;
    curand_snapshot_safe[device] = 1;
}

uint64 curand_initial_seed(int device) {
    CurandDeviceScope scope(device);
    return curand_seeds[device];
}

// See cublas_shutdown: report, never raise, and idempotent.
void curand_shutdown() {
    if (gens.empty()) return;
    for (auto g : gens)
        if (g) peekCudaErrorsAlways( curandDestroyGenerator(g) );
    gens.clear();
    curand_stream_binds.clear();
    curand_seeds.clear();
    curand_offsets.clear();
    curand_snapshot_safe.clear();
    gen = nullptr;
    LOGv << "curandDestroy finished";
}

struct curand_initer {

inline curand_initer() {
    if (!get_device_count()) return;
    add_device_switch_hook(curand_switch_device);
    add_set_seed_callback([](int seed) {
        curand_last_seed = seed;
        curand_has_seed = true;
        // The callback list is a separate global: nothing orders it against
        // these generators at exit, so a set_seed after shutdown must not run.
        for (size_t device = 0; device < gens.size(); ++device) {
            if (gens[device]) curand_seed_generator(gens[device], seed);
            curand_seeds[device] = uint64(seed);
            curand_offsets[device] = 0;
            curand_snapshot_safe[device] = 1;
        }
    });
    LOGv << "curandCreate finished";
}

inline ~curand_initer() {
    curand_shutdown();
}

} init_;

} // jittor
