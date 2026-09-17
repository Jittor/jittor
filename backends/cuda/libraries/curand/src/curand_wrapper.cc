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
#include <mutex>
#include <sstream>

namespace jittor {

curandGenerator_t gen;
// One generator per device; the global is the current device's. A generator
// draws from the device it was created on, so a single global one would make
// jt.rand() on device 1 either fail or fill device-0 memory.
static vector<curandGenerator_t> gens;
static vector<uint64> curand_stream_binds;
static vector<uint64> curand_seeds;
struct CurandOperation {
    uint8 normal;
    uint8 double_precision;
    uint64 count;
};
static vector<vector<CurandOperation>> curand_history;
static std::mutex curand_history_mutex;
// The last seed, replayed onto a generator created after set_seed so every
// device answers the same seed the same way.
static uint64 curand_last_seed = 0;
static bool curand_has_seed = false;

static void curand_seed_generator(curandGenerator_t g, uint64 seed) {
    checkCudaErrors( curandSetPseudoRandomGeneratorSeed(g, seed) );
    // Re-seeding must rewind the native stream as well as update the seed.
    checkCudaErrors( curandSetGeneratorOffset(g, 0) );
}

curandGenerator_t curand_bind_stream() {
    int device = current_device();
    checkCudaErrors(curandSetStream(gen, cuda_compute_stream(device)));
    std::lock_guard<std::mutex> lock(curand_history_mutex);
    if ((int)curand_stream_binds.size() <= device)
        curand_stream_binds.resize(device + 1);
    curand_stream_binds[device]++;
    return gen;
}

uint64 curand_stream_bind_count(int device) {
    std::lock_guard<std::mutex> lock(curand_history_mutex);
    return device >= 0 && device < (int)curand_stream_binds.size()
        ? curand_stream_binds[device] : 0;
}

static void curand_switch_device(int device) {
    std::lock_guard<std::mutex> lock(curand_history_mutex);
    if ((int)gens.size() <= device) gens.resize(device+1, nullptr);
    if ((int)curand_seeds.size() <= device) {
        curand_seeds.resize(device+1);
        curand_history.resize(device+1);
    }
    if (!gens[device]) {
        checkCudaErrors( curandCreateGenerator(&gens[device], CURAND_RNG_PSEUDO_DEFAULT) );
        if (curand_has_seed) curand_seed_generator(gens[device], curand_last_seed);
        curand_seeds[device] = curand_last_seed;
        curand_history[device].clear();
    }
    gen = gens[device];
}

void curand_record_operation(uint64 count, bool normal, bool double_precision) {
    if (!count) return;
    int device = current_device();
    USER_CHECK(device >= 0 && device < (int)curand_history.size())
        << "cuRAND RNG device is not initialized";
    USER_CHECK(curand_history[device].size() < 1000000)
        << "CUDA RNG operation history exceeds the supported checkpoint size";
    curand_history[device].push_back({uint8(normal), uint8(double_precision), count});
}

struct CurandSnapshot {
    uint64 seed;
    vector<CurandOperation> history;
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
    USER_CHECK(!state.empty() && state.size() <= 64 * 1024 * 1024)
        << "invalid cuRAND RNG state size";
    std::istringstream input(state);
    string version, seed, operation_count;
    int library_version;
    USER_CHECK(bool(input >> version >> library_version >> seed >> operation_count)
        && version == "JITTOR_CURAND_REPLAY_V1" && library_version == curand_version())
        << "invalid or incompatible cuRAND RNG state";
    uint64 count = parse_uint64(operation_count);
    USER_CHECK(count <= 1000000) << "cuRAND RNG state has too many operations";
    CurandSnapshot snapshot{parse_uint64(seed), {}};
    snapshot.history.reserve((size_t)count);
    uint64 replay_bytes = 0;
    for (uint64 index = 0; index < count; ++index) {
        string kind, precision, length;
        USER_CHECK(bool(input >> kind >> precision >> length))
            << "truncated cuRAND RNG operation history";
        USER_CHECK(kind == "uniform" || kind == "normal")
            << "invalid cuRAND RNG operation kind";
        USER_CHECK(precision == "f32" || precision == "f64")
            << "invalid cuRAND RNG operation precision";
        uint64 n = parse_uint64(length);
        USER_CHECK(n > 0) << "invalid zero-length cuRAND RNG operation";
        const uint64 element_size = precision == "f64" ? sizeof(double) : sizeof(float);
        USER_CHECK(n <= std::numeric_limits<size_t>::max() / element_size)
            << "cuRAND RNG operation is too large for this process";
        USER_CHECK(!(kind == "normal" && (n & 1) && n == std::numeric_limits<uint64>::max())
                   && n <= std::numeric_limits<uint64>::max() - ((kind == "normal" && (n & 1)) ? 1 : 0))
            << "cuRAND RNG normal operation length overflows during replay";
        uint64 bytes = n * element_size;
        if (kind == "normal" && (n & 1)) bytes += element_size;
        USER_CHECK(replay_bytes <= 1024ull * 1024ull * 1024ull - bytes)
            << "cuRAND RNG replay state exceeds the 1 GiB safety limit";
        replay_bytes += bytes;
        snapshot.history.push_back({uint8(kind == "normal"), uint8(precision == "f64"), n});
    }
    input >> std::ws;
    USER_CHECK(input.eof()) << "trailing data in cuRAND RNG state";
    return snapshot;
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
    std::lock_guard<std::mutex> lock(curand_history_mutex);
    std::ostringstream output;
    output << "JITTOR_CURAND_REPLAY_V1 " << curand_version() << ' '
           << curand_seeds[device] << ' ' << curand_history[device].size() << '\n';
    for (const auto& op : curand_history[device]) {
        output << (op.normal ? "normal" : "uniform") << ' '
               << (op.double_precision ? "f64" : "f32") << ' '
               << op.count << '\n';
    }
    return output.str();
}

static void replay_operation(curandGenerator_t generator, const CurandOperation& op) {
    size_t element_size = op.double_precision ? sizeof(double) : sizeof(float);
    uint64 consumed = op.count;
    if (op.normal && (consumed & 1)) ++consumed;
    void* buffer = nullptr;
    checkCudaErrors(cudaMalloc(&buffer, size_t(consumed) * element_size));
    if (op.normal) {
        if (op.count & 1) {
            if (op.count > 1)
                if (op.double_precision)
                    checkCudaErrors(curandGenerateNormalDouble(generator, (double*)buffer, op.count - 1, 0, 1));
                else
                    checkCudaErrors(curandGenerateNormal(generator, (float*)buffer, op.count - 1, 0, 1));
            void* tail = (char*)buffer + size_t(op.count - 1) * element_size;
            if (op.double_precision)
                checkCudaErrors(curandGenerateNormalDouble(generator, (double*)tail, 2, 0, 1));
            else
                checkCudaErrors(curandGenerateNormal(generator, (float*)tail, 2, 0, 1));
        } else if (op.double_precision) {
            checkCudaErrors(curandGenerateNormalDouble(generator, (double*)buffer, op.count, 0, 1));
        } else {
            checkCudaErrors(curandGenerateNormal(generator, (float*)buffer, op.count, 0, 1));
        }
    } else if (op.double_precision) {
        checkCudaErrors(curandGenerateUniformDouble(generator, (double*)buffer, op.count));
    } else {
        checkCudaErrors(curandGenerateUniform(generator, (float*)buffer, op.count));
    }
    checkCudaErrors(cudaFree(buffer));
}

void curand_set_rng_state(int device, const string& state) {
    auto snapshot = parse_curand_state(state);
    USER_CHECK(device >= 0 && device < get_device_count()) << "invalid cuRAND RNG device";
    sync_all(true);
    CurandDeviceScope scope(device);
    sync_devices(0);
    checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, snapshot.seed));
    checkCudaErrors(curandSetGeneratorOffset(gen, 0));
    {
        std::lock_guard<std::mutex> lock(curand_history_mutex);
        curand_seeds[device] = snapshot.seed;
        curand_history[device] = snapshot.history;
    }
    checkCudaErrors(curandSetStream(gen, cuda_compute_stream(device)));
    for (const auto& op : snapshot.history) replay_operation(gen, op);
    checkCudaErrors(cudaStreamSynchronize(cuda_compute_stream(device)));
}

void curand_manual_seed(int device, uint64 seed) {
    sync_all(true);
    CurandDeviceScope scope(device);
    sync_devices(0);
    curand_seed_generator(gen, seed);
    {
        std::lock_guard<std::mutex> lock(curand_history_mutex);
        curand_seeds[device] = seed;
        curand_history[device].clear();
    }
}

uint64 curand_initial_seed(int device) {
    CurandDeviceScope scope(device);
    std::lock_guard<std::mutex> lock(curand_history_mutex);
    return curand_seeds[device];
}

// See cublas_shutdown: report, never raise, and idempotent.
void curand_shutdown() {
    if (gens.empty()) return;
    for (auto g : gens)
        if (g) peekCudaErrorsAlways( curandDestroyGenerator(g) );
    std::lock_guard<std::mutex> lock(curand_history_mutex);
    gens.clear();
    curand_stream_binds.clear();
    curand_seeds.clear();
    curand_history.clear();
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
        std::lock_guard<std::mutex> lock(curand_history_mutex);
        for (size_t device = 0; device < gens.size(); ++device) {
            if (gens[device]) curand_seed_generator(gens[device], seed);
            curand_seeds[device] = uint64(seed);
            curand_history[device].clear();
        }
    });
    LOGv << "curandCreate finished";
}

inline ~curand_initer() {
    curand_shutdown();
}

} init_;

} // jittor
