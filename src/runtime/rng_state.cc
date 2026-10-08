#include "runtime/rng_state.h"
#include "runtime/runtime.h"
#include "runtime/backend.h"
#include "runtime/device.h"
#include "runtime/executor_entry.h"
#include <cstring>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace jittor {

EXTERN_LIB void sync_all(bool device_sync);

RuntimeRngState& runtime_rng_state() { return native_runtime().rng(); }

namespace {

int stream_count(const string& backend) {
    if (backend == "cpu") return 1;
    if (backend != "acl")
        throw std::invalid_argument("RNG state supports CPU and ACL, not " + backend);
    const int count = backend_device_count("acl");
    if (count <= 0) throw std::runtime_error("ACL RNG requires an available ACL device");
    return count;
}

int stream_device(const string& backend, int device) {
    const int count = stream_count(backend);
    if (device == -1) device = backend == "cpu" ? 0 : current_device();
    if (device < 0 || device >= count)
        throw std::invalid_argument("RNG device index is outside the visible backend");
    return device;
}

AclRngStream& acl_stream(int device) {
    auto& owner = runtime_rng_state();
    auto found = owner.acl_streams.find(device);
    if (found == owner.acl_streams.end()) {
        AclRngStream initial;
        initial.seed = owner.global_seed;
        found = owner.acl_streams.emplace(device, initial).first;
    }
    return found->second;
}

struct ParsedState {
    uint64 seed = 0;
    int64 offset = 0;
    std::default_random_engine engine;
};

// Entire malformed payload is rejected before flushing queued work or changing
// any generator. The engine's parameters identify this serialization format.
ParsedState parse_state(const string& payload, const string& backend, int device) {
    if (payload.empty() || payload.size() > 65536)
        throw std::invalid_argument("Invalid RNG state byte length");
    std::istringstream input(payload);
    input.imbue(std::locale::classic());
    string version, saved_backend;
    int saved_device = -1;
    ParsedState result;
    if (!(input >> version >> saved_backend >> saved_device >> result.seed >> result.offset)
            || version != "JITTOR_RNG_1" || saved_backend != backend
            || saved_device != device || result.offset < 0)
        throw std::invalid_argument("RNG state version/backend/device/header mismatch");
    if (backend == "cpu") {
        uint64 minimum, maximum;
        if (result.offset != 0 || !(input >> minimum >> maximum >> std::ws >> result.engine)
                || minimum != std::default_random_engine::min()
                || maximum != std::default_random_engine::max())
            throw std::invalid_argument("RNG host engine state mismatch");
    } else {
        string kind;
        if (!(input >> kind) || kind != "ACL_COUNTER")
            throw std::invalid_argument("RNG ACL counter state mismatch");
    }
    input >> std::ws;
    if (!input.eof()) throw std::invalid_argument("Unexpected trailing RNG state bytes");
    return result;
}

string encode_state(const string& backend, int device) {
    auto& owner = runtime_rng_state();
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "JITTOR_RNG_1 " << backend << ' ' << device << ' ';
    if (backend == "cpu") {
        output << owner.host_seed << " 0 " << std::default_random_engine::min()
               << ' ' << std::default_random_engine::max() << ' ' << owner.host_engine;
    } else {
        const auto& stream = acl_stream(device);
        output << stream.seed << ' ' << stream.offset << " ACL_COUNTER";
    }
    return output.str();
}

void apply_state(const string& backend, int device, const ParsedState& state) {
    auto& owner = runtime_rng_state();
    if (backend == "cpu") {
        owner.host_seed = state.seed;
        owner.host_engine = state.engine;
    } else {
        auto& stream = acl_stream(device);
        stream.seed = state.seed;
        stream.offset = state.offset;
    }
}

void seed_stream(const string& backend, int device, uint64 seed) {
    auto& owner = runtime_rng_state();
    if (backend == "cpu") {
        owner.host_seed = seed;
        owner.host_engine.seed(seed);
    } else {
        auto& stream = acl_stream(device);
        stream.seed = seed;
        stream.offset = 0;
    }
}

} // namespace

AclRngSpan reserve_acl_random(int device, int64 elements) {
    // Kernel launchers already own the executor. Compatibility generator
    // hooks can reserve counters from the host before a graph is submitted,
    // so acquire the same process-wide entry lock when called externally.
    // ExecutorEntryScope is recursive by thread; the recursive call below
    // reaches the state mutation with the lock held in either case.
    if (!inside_executor()) {
        ExecutorEntryScope lock;
        return reserve_acl_random(device, elements);
    }
    if (device < 0 || elements < 0) throw std::invalid_argument("Invalid ACL random reservation");
    auto& state = acl_stream(device);
    if (elements > std::numeric_limits<int64>::max() - state.offset)
        throw std::overflow_error("ACL RNG offset exhausted");
    AclRngSpan span;
    // CANN receives a signed int64 seed carrying the original 64 seed bits.
    static_assert(sizeof(span.seed) == sizeof(state.seed), "seed width mismatch");
    std::memcpy(&span.seed, &state.seed, sizeof(span.seed));
    span.offset = state.offset;
    state.offset += elements;
    return span;
}

string rng_state(const string& backend, int device) {
    ExecutorEntryScope lock;
    device = stream_device(backend, device);
    sync_all(true);
    return encode_state(backend, device);
}

void set_rng_state(const string& backend, const string& state, int device) {
    ExecutorEntryScope lock;
    device = stream_device(backend, device);
    auto parsed = parse_state(state, backend, device);
    sync_all(true);
    apply_state(backend, device, parsed);
}

vector<string> rng_states(const string& backend) {
    ExecutorEntryScope lock;
    const int count = stream_count(backend);
    sync_all(true);
    vector<string> result;
    result.reserve(count);
    for (int device = 0; device < count; ++device) result.push_back(encode_state(backend, device));
    return result;
}

void set_rng_states(const string& backend, const vector<string>& states) {
    ExecutorEntryScope lock;
    const int count = stream_count(backend);
    if (states.size() != static_cast<size_t>(count))
        throw std::invalid_argument("RNG state count must match every visible device");
    vector<ParsedState> parsed;
    parsed.reserve(count);
    for (int device = 0; device < count; ++device)
        parsed.push_back(parse_state(states[device], backend, device));
    sync_all(true);
    for (int device = 0; device < count; ++device) apply_state(backend, device, parsed[device]);
}

uint64 rng_initial_seed(const string& backend, int device) {
    ExecutorEntryScope lock;
    device = stream_device(backend, device);
    return backend == "cpu" ? runtime_rng_state().host_seed : acl_stream(device).seed;
}

void rng_manual_seed(const string& backend, uint64 seed, int device) {
    ExecutorEntryScope lock;
    device = stream_device(backend, device);
    sync_all(true);
    seed_stream(backend, device, seed);
}

void rng_manual_seed_all(const string& backend, uint64 seed) {
    ExecutorEntryScope lock;
    const int count = stream_count(backend);
    sync_all(true);
    for (int device = 0; device < count; ++device) seed_stream(backend, device, seed);
}

} // namespace jittor
