// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include <chrono>
#include "runtime/random_generator.h"

namespace jittor {

// Marks the counter-based CUDA offset. Strings written before it carry a
// curand host-generator offset, which cannot be replayed on this stream.
static const char* const kCudaStreamTag = "philox";

RandomGeneratorState::RandomGeneratorState(int64 seed, int device_id)
    : seed_(seed), device_id(device_id), cpu_engine(seed) {}

void RandomGeneratorState::set_seed(int64 seed) {
    seed_ = seed;
    cpu_engine.seed((uint64)seed);
    cuda_offset = 0;
}

int64 RandomGeneratorState::reserve_cuda(int64 raw_count) {
    ASSERT(raw_count >= 0);
    int64 base = cuda_offset;
    USER_CHECK(!__builtin_add_overflow(cuda_offset, raw_count, &cuda_offset))
        << "Generator: CUDA stream position overflows int64";
    return base;
}

string RandomGeneratorState::get_state() {
    std::ostringstream oss;
    oss << seed_ << ' ' << device_id << ' ' << cuda_offset << ' '
        << kCudaStreamTag << ' ' << cpu_engine;
    return oss.str();
}

void RandomGeneratorState::set_state(const string& s) {
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
    USER_CHECK(new_device == device_id)
        << "Generator.set_state(): the state belongs to"
        << (new_device < 0 ? string("cpu") : "cuda:" + std::to_string(new_device))
        << "but this generator is on"
        << (device_id < 0 ? string("cpu") : "cuda:" + std::to_string(device_id));
    seed_ = new_seed;
    cuda_offset = new_offset;
    cpu_engine = new_engine;
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
    int device_id = parse_device(device);
    state = std::make_shared<RandomGeneratorState>(fresh_seed(), device_id);
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
    return state->seed_;
}

string RandomGenerator::get_state() {
    return state->get_state();
}

RandomGenerator* RandomGenerator::set_state(const string& s) {
    state->set_state(s);
    return this;
}

string RandomGenerator::get_device() {
    return device_str;
}

} // jittor
