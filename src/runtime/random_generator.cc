// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include <chrono>
#include <limits>
#include "runtime/random_generator.h"
#include "runtime/rng_state.h"

namespace jittor {

// Marks the counter-based CUDA offset. Strings written before it carry a
// curand host-generator offset, which cannot be replayed on this stream.
static const char* const kCudaStreamTag = "philox";
static const char* const kAclStateTag = "JITTOR_ACL_RNG_1";
static const char* const kAclAlgorithm = "philox4x32-10-raw32-v1";

RandomGeneratorState::RandomGeneratorState(int64 seed, int device_id, BackendId backend)
    : seed_(seed), device_id(device_id), backend(backend), cpu_engine(seed) {}

void RandomGeneratorState::set_seed(int64 seed) {
    std::lock_guard<std::mutex> lock(mutex);
    seed_ = seed;
    cpu_engine.seed((uint64)seed);
    cpu_torch_normal.reset();
    cuda_offset = 0;
    acl_offset = 0;
}

int64 RandomGeneratorState::reserve_cuda(int64 raw_count) {
    USER_CHECK(backend != BackendId::Acl)
        << "An ACL generator cannot be used for a CUDA draw";
    ASSERT(raw_count >= 0);
    int64 base = cuda_offset;
    USER_CHECK(!__builtin_add_overflow(cuda_offset, raw_count, &cuda_offset))
        << "Generator: CUDA stream position overflows int64";
    return base;
}

AclPrivateRngSpan RandomGeneratorState::reserve_acl(int64 raw_count) {
    std::lock_guard<std::mutex> lock(mutex);
    USER_CHECK(backend == BackendId::Acl && raw_count >= 0)
        << "ACL RNG reservation requires an ACL generator and nonnegative count";
    USER_CHECK(raw_count <= std::numeric_limits<int64>::max() - acl_offset)
        << "Generator: ACL raw32 stream position overflows int64";
    AclPrivateRngSpan span{(uint64)seed_, acl_offset};
    acl_offset += raw_count;
    return span;
}

int64 RandomGeneratorState::get_offset() {
    std::lock_guard<std::mutex> lock(mutex);
    USER_CHECK(backend == BackendId::Acl)
        << "Generator.get_offset() currently supports private ACL streams only";
    return acl_offset;
}

void RandomGeneratorState::set_offset(int64 offset) {
    std::lock_guard<std::mutex> lock(mutex);
    USER_CHECK(backend == BackendId::Acl && offset >= 0)
        << "Generator.set_offset() requires a private ACL stream and a nonnegative int64 raw32 position";
    acl_offset = offset;
}

int64 RandomGeneratorState::get_seed() {
    std::lock_guard<std::mutex> lock(mutex);
    return seed_;
}

string RandomGeneratorState::get_state() {
    std::lock_guard<std::mutex> lock(mutex);
    if (backend == BackendId::Acl) {
        std::ostringstream oss;
        oss << kAclStateTag << ' ' << kAclAlgorithm << " acl " << device_id
            << ' ' << seed_ << ' ' << acl_offset;
        return oss.str();
    }
    if (cpu_torch_normal)
        return encode_cpu_rng_state((uint64)seed_, cpu_engine, cpu_torch_normal);
    std::ostringstream oss;
    oss << seed_ << ' ' << device_id << ' ' << cuda_offset << ' '
        << kCudaStreamTag << ' ' << cpu_engine;
    return oss.str();
}

void RandomGeneratorState::set_state(const string& s) {
    std::lock_guard<std::mutex> lock(mutex);
    if (backend == BackendId::Acl) {
        std::istringstream iss(s);
        string version, algorithm, new_backend;
        int new_device;
        int64 new_seed, new_offset;
        iss >> version >> algorithm >> new_backend >> new_device >> new_seed >> new_offset;
        USER_CHECK(!iss.fail() && version == kAclStateTag
                   && algorithm == kAclAlgorithm && new_backend == "acl"
                   && new_device == device_id && new_offset >= 0)
            << "Generator.set_state(): incompatible or malformed private ACL state";
        iss >> std::ws;
        USER_CHECK(iss.eof()) << "Generator.set_state(): unexpected trailing ACL state bytes";
        // Nothing mutates before all fields, including backend and ordinal,
        // have been validated. Outstanding ops retain their captured spans.
        seed_ = new_seed;
        acl_offset = new_offset;
        return;
    }
    if (s.compare(0, 12, "JITTOR_RNG_1") == 0 || s.compare(0, 12, "JITTOR_RNG_2") == 0) {
        USER_CHECK(device_id < 0)
            << "Generator.set_state(): the state belongs to cpu but this generator is on cuda";
        uint64 seed;
        std::default_random_engine engine;
        shared_ptr<TorchCpuNormal> normal;
        decode_cpu_rng_state(s, seed, engine, normal);
        seed_ = (int64)seed;
        cuda_offset = 0;
        cpu_engine = engine;
        cpu_torch_normal = move(normal);
        return;
    }
    std::istringstream iss(s);
    int64 new_seed, new_offset;
    int new_device;
    iss >> new_seed >> new_device >> new_offset >> std::ws;
    USER_CHECK(!iss.fail()) << "Generator.set_state(): malformed state string "
        "(expected the byte string previously returned by get_state()).";
    auto engine_start = iss.tellg();
    string tag;
    iss >> tag;
    bool tagged = !iss.fail() && tag == kCudaStreamTag;
    if (!tagged) {
        // A state from before the counter-based CUDA stream. Its CPU part is
        // still exact; a CUDA position into the old curand stream is not.
        USER_CHECK(new_device < 0 || new_offset == 0)
            << "Generator.set_state(): this CUDA state was saved by an older "
               "Jittor whose curand stream cannot be replayed here; reseed with "
               "manual_seed() instead of restoring it.";
        iss.clear();
        iss.seekg(engine_start);
    }
    std::default_random_engine new_engine;
    // std::default_random_engine's own operator>> does not skip leading
    // whitespace (unlike operator>> for the built-in numeric types before
    // it) -- without the explicit std::ws here it tries to parse starting
    // at the separating space itself and silently fails every time.
    iss >> std::ws >> new_engine;
    USER_CHECK(!iss.fail() && new_offset >= 0)
        << "Generator.set_state(): malformed state string "
        "(expected the byte string previously returned by get_state()).";
    iss >> std::ws;
    USER_CHECK(iss.eof()) << "Generator.set_state(): unexpected trailing state bytes";
    USER_CHECK(new_device == device_id)
        << "Generator.set_state(): the state belongs to"
        << (new_device < 0 ? string("cpu") : "cuda:" + std::to_string(new_device))
        << "but this generator is on"
        << (device_id < 0 ? string("cpu") : "cuda:" + std::to_string(device_id));
    seed_ = new_seed;
    cuda_offset = new_offset;
    cpu_engine = new_engine;
    cpu_torch_normal.reset();
}

static int parse_device(const string& device) {
    if (device == "cpu") return -1;
    if (device.substr(0, 4) == "cuda") {
        if (device.size() > 5 && device[4] == ':')
            return std::stoi(device.substr(5));
        return 0;
    }
    LOGf << "Generator: unknown device string" << device << "(expected \"cpu\", \"cuda\", or \"cuda:N\")";
    return -1;
}

static int64 fresh_seed() {
    // matches the intent of torch.Generator().seed(): unpredictable,
    // process/time-derived, not required to be cryptographically secure.
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::random_device rd;
    uint64 a = (uint64)now, b;
    try { b = ((uint64)rd() << 32) ^ (uint64)rd(); }
    catch (...) { b = 0; }
    return (int64)(a ^ b ^ (a << 21) ^ (b >> 13));
}

RandomGenerator::RandomGenerator(const string& device) : device_str(device) {
    if (device == "acl" || device.compare(0, 4, "acl:") == 0) {
        int ordinal = 0;
        if (device != "acl") {
            USER_CHECK(device.size() > 4) << "Generator: expected acl:N with a nonnegative ordinal";
            for (size_t i = 4; i < device.size(); ++i) {
                const char c = device[i];
                USER_CHECK(c >= '0' && c <= '9'
                    && ordinal <= (std::numeric_limits<int>::max() - (c - '0')) / 10)
                    << "Generator: invalid ACL device ordinal";
                ordinal = ordinal * 10 + (c - '0');
            }
        }
        device_str = "acl:" + std::to_string(ordinal);
        state = std::make_shared<RandomGeneratorState>(fresh_seed(), ordinal, BackendId::Acl);
        return;
    }
    int device_id = parse_device(device);
    state = std::make_shared<RandomGeneratorState>(fresh_seed(), device_id,
        device_id < 0 ? BackendId::Cpu : BackendId::Cuda);
}

RandomGenerator* RandomGenerator::manual_seed(int64 seed) {
    state->set_seed(seed);
    return this;
}

int64 RandomGenerator::reseed() {
    int64 s = fresh_seed();
    state->set_seed(s);
    return s;
}

int64 RandomGenerator::initial_seed() {
    return state->get_seed();
}

string RandomGenerator::get_state() {
    return state->get_state();
}

string RandomGenerator::get_cpu_rng_state() {
    std::lock_guard<std::mutex> lock(state->mutex);
    USER_CHECK(state->device_id < 0 && state->cuda_offset == 0)
        << "CPU RNG snapshot requires a CPU-only generator stream";
    return encode_cpu_rng_state((uint64)state->seed_, state->cpu_engine, state->cpu_torch_normal);
}

RandomGenerator* RandomGenerator::set_state(const string& s) {
    state->set_state(s);
    return this;
}

int64 RandomGenerator::get_offset() {
    return state->get_offset();
}

RandomGenerator* RandomGenerator::set_offset(int64 offset) {
    state->set_offset(offset);
    return this;
}

string RandomGenerator::get_device() {
    return device_str;
}

} // jittor
