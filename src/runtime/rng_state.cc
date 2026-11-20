#include "runtime/rng_state.h"
#include "runtime/runtime.h"
#include "runtime/backend.h"
#include "runtime/device.h"
#include "runtime/executor_entry.h"
#include "runtime/async_exec.h"
#include <cmath>
#include <cstring>
#include <iomanip>
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
    shared_ptr<TorchCpuNormal> normal;
};

shared_ptr<TorchCpuNormal> parse_torch_normal(std::istream& input, uint64 seed) {
    int has_spare;
    double spare;
    if (!(input >> has_spare >> spare) || (has_spare != 0 && has_spare != 1)
            || !std::isfinite(spare))
        throw std::invalid_argument("Invalid Torch CPU normal cache");
    std::ostringstream engine_state;
    engine_state.imbue(std::locale::classic());
    bool has_nonzero_word = false;
    for (size_t i = 0; i < std::mt19937::state_size; ++i) {
        uint64 word;
        if (!(input >> word) || word > std::numeric_limits<uint32>::max())
            throw std::invalid_argument("Invalid Torch CPU normal engine state");
        engine_state << word << ' ';
        has_nonzero_word |= word != 0;
    }
    if (!has_nonzero_word) throw std::invalid_argument("Invalid Torch CPU normal engine");
    input >> std::ws;
    // libstdc++ appends a cursor. libc++ serializes the state in draw order.
    if (!input.eof()) {
        size_t cursor;
        if (!(input >> cursor) || cursor > std::mt19937::state_size)
            throw std::invalid_argument("Invalid Torch CPU normal engine cursor");
        engine_state << cursor;
    }
    auto normal = std::make_shared<TorchCpuNormal>(seed);
    std::istringstream words(engine_state.str());
    words.imbue(std::locale::classic());
    if (!(words >> normal->engine))
        throw std::invalid_argument("Invalid Torch CPU normal engine");
    normal->has_spare = has_spare;
    normal->spare = spare;
    return normal;
}

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
    input >> version;
    if (version == "JTCPU1") {
        // Checkpoints before the runtime RNG API stored a signed 32-bit seed
        // and this same complete host engine. Restore its position, not a
        // freshly seeded engine; this format never described an ACL stream.
        int32 legacy_seed;
        if (backend != "cpu" || device != 0
                || !(input >> legacy_seed >> std::ws >> result.engine))
            throw std::invalid_argument("RNG legacy host engine state mismatch");
        result.seed = static_cast<uint64>(static_cast<int64>(legacy_seed));
    } else {
        const bool has_normal = version == "JITTOR_RNG_2" && backend == "cpu";
        if (!(input >> saved_backend >> saved_device >> result.seed >> result.offset)
                || (version != "JITTOR_RNG_1" && !has_normal) || saved_backend != backend
                || saved_device != device || result.offset < 0)
            throw std::invalid_argument("RNG state version/backend/device/header mismatch");
        if (backend == "cpu") {
            uint64 minimum, maximum;
            if (result.offset != 0 || !(input >> minimum >> maximum >> std::ws >> result.engine)
                    || minimum != std::default_random_engine::min()
                    || maximum != std::default_random_engine::max())
                throw std::invalid_argument("RNG host engine state mismatch");
            if (has_normal) result.normal = parse_torch_normal(input, result.seed);
        } else {
            string kind;
            if (!(input >> kind) || kind != "ACL_COUNTER")
                throw std::invalid_argument("RNG ACL counter state mismatch");
        }
    }
    input >> std::ws;
    if (!input.eof()) throw std::invalid_argument("Unexpected trailing RNG state bytes");
    return result;
}

string encode_state(const string& backend, int device) {
    auto& owner = runtime_rng_state();
    if (backend == "cpu")
        return encode_cpu_rng_state(owner.host_seed, owner.host_engine, owner.host_torch_normal);
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "JITTOR_RNG_1 " << backend << ' ' << device << ' ';
    const auto& stream = acl_stream(device);
    output << stream.seed << ' ' << stream.offset << " ACL_COUNTER";
    return output.str();
}

void apply_state(const string& backend, int device, const ParsedState& state) {
    auto& owner = runtime_rng_state();
    if (backend == "cpu") {
        owner.host_seed = state.seed;
        owner.host_engine = state.engine;
        owner.host_torch_normal = state.normal;
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
        owner.host_torch_normal.reset();
    } else {
        auto& stream = acl_stream(device);
        stream.seed = seed;
        stream.offset = 0;
    }
}

} // namespace

string encode_cpu_rng_state(uint64 seed, const std::default_random_engine& engine,
                            const shared_ptr<TorchCpuNormal>& normal) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << (normal ? "JITTOR_RNG_2 cpu 0 " : "JITTOR_RNG_1 cpu 0 ") << seed << " 0 "
           << std::default_random_engine::min() << ' '
           << std::default_random_engine::max() << ' ' << engine;
    if (normal)
        output << ' ' << int(normal->has_spare) << ' '
               << std::setprecision(std::numeric_limits<double>::max_digits10)
               << normal->spare << ' ' << normal->engine;
    return output.str();
}

void decode_cpu_rng_state(const string& payload, uint64& seed,
                          std::default_random_engine& engine,
                          shared_ptr<TorchCpuNormal>& normal) {
    const auto parsed = parse_state(payload, "cpu", 0);
    seed = parsed.seed;
    engine = parsed.engine;
    normal = parsed.normal;
}

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

template<class T>
T torch_uniform(TorchCpuNormal& state) {
    constexpr int bits = std::numeric_limits<T>::digits;
    uint64 raw = state.engine();
    if (bits > 32) raw = (raw << 32) | state.engine();
    constexpr uint64 range = uint64(1) << bits;
    return T(raw & (range - 1)) / T(range);
}

double torch_scalar_normal(TorchCpuNormal& state) {
    if (state.has_spare) {
        state.has_spare = false;
        return state.spare;
    }
    const double angle = kTwoPi * torch_uniform<double>(state);
    const double radius = std::sqrt(-2.0 * std::log1p(-torch_uniform<double>(state)));
    state.spare = radius * std::sin(angle);
    state.has_spare = true;
    return radius * std::cos(angle);
}

template<class T>
void torch_normal_block(T* values) {
    for (int lane = 0; lane < 8; ++lane) {
        const T radius = std::sqrt(T(-2) * std::log(T(1) - values[lane]));
        const T angle = T(kTwoPi) * values[lane + 8];
        values[lane] = radius * std::cos(angle);
        values[lane + 8] = radius * std::sin(angle);
    }
}

template<class T>
void fill_torch_normal(TorchCpuNormal& state, T* output, int64 size) {
    // ATen CPU uses a cached scalar Box–Muller for short tensors. Contiguous
    // draws from 16 elements use blocks of eight radii and eight angles.
    // When the tail overlaps a full block, ATen redraws the last 16 entries.
    // Its architecture-specific vector math can round bulk FP32 differently.
    if (size < 16) {
        for (int64 i = 0; i < size; ++i) output[i] = T(torch_scalar_normal(state));
        return;
    }
    for (int64 i = 0; i < size; ++i) output[i] = torch_uniform<T>(state);
    for (int64 i = 0; i <= size - 16; i += 16) torch_normal_block(output + i);
    if (size % 16) {
        auto* tail = output + size - 16;
        for (int i = 0; i < 16; ++i) tail[i] = torch_uniform<T>(state);
        torch_normal_block(tail);
    }
}

} // namespace

void fill_torch_cpu_normal(TorchCpuNormal& state, void* output, int64 size, bool is_double) {
    if (is_double) fill_torch_normal(state, static_cast<double*>(output), size);
    else fill_torch_normal(state, static_cast<float*>(output), size);
}

AclRngSpan reserve_acl_random(int device, int64 elements) {
    // Kernel launchers already own the executor. Compatibility generator
    // hooks can reserve counters from the host before a graph is submitted,
    // so acquire the same process-wide entry lock when called externally.
    // ExecutorEntryScope is recursive by thread; the recursive call below
    // reaches the state mutation with the lock held in either case.
    //
    // The async-execution worker is the exception at both ends. It runs a
    // batch whose entry scope is held by no thread -- the caller let go of it
    // when it handed the batch over -- so it looks external, and taking the
    // scope here would deadlock against a Python thread that holds it and is
    // waiting for this batch inside `sync_all`. It needs no scope: one batch
    // runs at a time and every executor entry waits for it first, so the
    // worker and the Python thread never reach this state together. An
    // external caller does have to wait for that batch, since its counters
    // are reserved as the batch runs.
    if (!inside_executor() && !on_async_worker()) {
        async_exec_wait();
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
    // The worker reserves counters as it runs, and takes no entry scope.
    async_exec_wait();
    ExecutorEntryScope lock;
    device = stream_device(backend, device);
    sync_all(true);
    return encode_state(backend, device);
}

void set_rng_state(const string& backend, const string& state, int device) {
    // The worker reserves counters as it runs, and takes no entry scope.
    async_exec_wait();
    ExecutorEntryScope lock;
    device = stream_device(backend, device);
    auto parsed = parse_state(state, backend, device);
    sync_all(true);
    apply_state(backend, device, parsed);
}

vector<string> rng_states(const string& backend) {
    // The worker reserves counters as it runs, and takes no entry scope.
    async_exec_wait();
    ExecutorEntryScope lock;
    const int count = stream_count(backend);
    sync_all(true);
    vector<string> result;
    result.reserve(count);
    for (int device = 0; device < count; ++device) result.push_back(encode_state(backend, device));
    return result;
}

void set_rng_states(const string& backend, const vector<string>& states) {
    // The worker reserves counters as it runs, and takes no entry scope.
    async_exec_wait();
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
    // The worker reserves counters as it runs, and takes no entry scope.
    async_exec_wait();
    ExecutorEntryScope lock;
    device = stream_device(backend, device);
    return backend == "cpu" ? runtime_rng_state().host_seed : acl_stream(device).seed;
}

void rng_manual_seed(const string& backend, uint64 seed, int device) {
    // The worker reserves counters as it runs, and takes no entry scope.
    async_exec_wait();
    ExecutorEntryScope lock;
    device = stream_device(backend, device);
    sync_all(true);
    seed_stream(backend, device, seed);
}

void rng_manual_seed_all(const string& backend, uint64 seed) {
    // The worker reserves counters as it runs, and takes no entry scope.
    async_exec_wait();
    ExecutorEntryScope lock;
    const int count = stream_count(backend);
    sync_all(true);
    for (int device = 0; device < count; ++device) seed_stream(backend, device, seed);
}

} // namespace jittor
