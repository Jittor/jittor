#pragma once
#include "core/common.h"
#include <map>
#include <random>

namespace jittor {

struct AclRngStream {
    uint64 seed = 0;
    int64 offset = 0;
};

// One physical owner. Host and each visible ACL device have independent state;
// default device streams inherit the last global seed until explicitly seeded.
struct RuntimeRngState {
    uint64 global_seed = 0;
    uint64 host_seed = 0;
    std::default_random_engine host_engine;
    std::map<int, AclRngStream> acl_streams;
};

EXTERN_LIB RuntimeRngState& runtime_rng_state();
struct AclRngSpan { int64 seed; int64 offset; };
// Called under the executor lock by the actual ACL RandomOp launcher.
EXTERN_LIB AclRngSpan reserve_acl_random(int device, int64 elements);

// These checkpoint APIs are synchronized state operations, not passive
// jt.introspection getters. Payloads are versioned and runtime-format specific.
// backend is exactly "cpu" or "acl"; cpu device is 0, -1 selects current ACL.
// @pyjt(rng_state)
string rng_state(const string& backend, int device=-1);
// @pyjt(set_rng_state)
void set_rng_state(const string& backend, const string& state, int device=-1);
// @pyjt(rng_states)
vector<string> rng_states(const string& backend);
// Validate every payload before changing any stream.
// @pyjt(set_rng_states)
void set_rng_states(const string& backend, const vector<string>& states);
// @pyjt(rng_initial_seed)
uint64 rng_initial_seed(const string& backend, int device=-1);
// @pyjt(rng_manual_seed)
void rng_manual_seed(const string& backend, uint64 seed, int device=-1);
// @pyjt(rng_manual_seed_all)
void rng_manual_seed_all(const string& backend, uint64 seed);

} // namespace jittor
