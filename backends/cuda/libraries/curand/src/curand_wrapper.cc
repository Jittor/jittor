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
#include "core/var_holder.h"
#include "runtime/init.h"
#include "runtime/device.h"
#include <limits>
#include <mutex>
#include <sstream>

namespace jittor {

static vector<uint64> curand_stream_binds;
static vector<uint64> curand_seeds;
static vector<uint64> curand_offsets;
static std::mutex curand_state_mutex;
// The last seed, replayed onto a generator created after set_seed so every
// device answers the same seed the same way.
static uint64 curand_last_seed = 0;
static bool curand_has_seed = false;

uint64 curand_stream_bind_count(int device) {
    std::lock_guard<std::mutex> lock(curand_state_mutex);
    return device >= 0 && device < (int)curand_stream_binds.size()
        ? curand_stream_binds[device] : 0;
}

static void curand_switch_device(int device) {
    std::lock_guard<std::mutex> lock(curand_state_mutex);
    if ((int)curand_seeds.size() <= device) {
        size_t old_size = curand_seeds.size();
        curand_seeds.resize(device+1);
        curand_offsets.resize(device+1);
        curand_stream_binds.resize(device+1);
        for (size_t index = old_size; index < curand_seeds.size(); ++index)
            curand_seeds[index] = curand_has_seed ? curand_last_seed : 0;
    }
}

void curand_reserve_philox_words(
        uint64 words, uint64 alignment, uint64& seed, uint64& offset) {
    int device = current_device();
    std::lock_guard<std::mutex> lock(curand_state_mutex);
    USER_CHECK(device >= 0 && device < (int)curand_offsets.size())
        << "CUDA RNG device is not initialized";
    USER_CHECK(alignment && !(alignment & (alignment - 1)))
        << "CUDA RNG alignment must be a nonzero power of two";
    uint64 padding = (-curand_offsets[device]) & (alignment - 1);
    USER_CHECK(curand_offsets[device] <= std::numeric_limits<uint64>::max() - padding)
        << "CUDA RNG Philox counter exceeds uint64 range";
    uint64 aligned_offset = curand_offsets[device] + padding;
    USER_CHECK(aligned_offset <= std::numeric_limits<uint64>::max() - words)
        << "CUDA RNG Philox counter exceeds uint64 range";
    seed = curand_seeds[device];
    offset = aligned_offset;
    curand_offsets[device] = aligned_offset + words;
    curand_stream_binds[device]++;
}

struct CurandSnapshot {
    uint64 seed;
    uint64 offset;
};

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
    USER_CHECK(!state.empty() && state.size() <= 256) << "invalid CUDA RNG state size";
    std::istringstream input(state);
    string version, seed, offset;
    USER_CHECK(bool(input >> version >> seed >> offset)
        && version == "JITTOR_CUDA_PHILOX4X32_10_V1")
        << "invalid or incompatible CUDA RNG state";
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
    std::lock_guard<std::mutex> lock(curand_state_mutex);
    std::ostringstream output;
    output << "JITTOR_CUDA_PHILOX4X32_10_V1 "
           << curand_seeds[device] << ' ' << curand_offsets[device] << '\n';
    return output.str();
}

void curand_set_rng_state(int device, const string& state) {
    auto snapshot = parse_curand_state(state);
    USER_CHECK(device >= 0 && device < get_device_count()) << "invalid cuRAND RNG device";
    sync_all(true);
    CurandDeviceScope scope(device);
    sync_devices(0);
    std::lock_guard<std::mutex> lock(curand_state_mutex);
    curand_seeds[device] = snapshot.seed;
    curand_offsets[device] = snapshot.offset;
}

void curand_manual_seed(int device, uint64 seed) {
    sync_all(true);
    CurandDeviceScope scope(device);
    sync_devices(0);
    std::lock_guard<std::mutex> lock(curand_state_mutex);
    curand_seeds[device] = seed;
    curand_offsets[device] = 0;
}

uint64 curand_initial_seed(int device) {
    CurandDeviceScope scope(device);
    std::lock_guard<std::mutex> lock(curand_state_mutex);
    return curand_seeds[device];
}

// See cublas_shutdown: report, never raise, and idempotent.
void curand_shutdown() {
    std::lock_guard<std::mutex> lock(curand_state_mutex);
    curand_stream_binds.clear();
    curand_seeds.clear();
    curand_offsets.clear();
}

struct curand_initer {

inline curand_initer() {
    if (!get_device_count()) return;
    add_device_switch_hook(curand_switch_device);
    add_set_seed_callback([](int seed) {
        std::lock_guard<std::mutex> lock(curand_state_mutex);
        curand_last_seed = seed;
        curand_has_seed = true;
        for (size_t device = 0; device < curand_seeds.size(); ++device) {
            curand_seeds[device] = uint64(seed);
            curand_offsets[device] = 0;
        }
    });
}

inline ~curand_initer() {
    curand_shutdown();
}

} init_;

} // jittor
