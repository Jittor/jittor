#pragma once
#include "runtime/backend.h"

namespace jittor {

struct BackendFallbackState {
    string policy = "warn";
    uint64 count = 0;
};

EXTERN_LIB BackendFallbackState& runtime_backend_fallback();
DECLARE_RUNTIME_FLAG(string, backend_fallback);

// Call once before an unsupported implementation takes another backend's
// computation path. Host staging and device copies are not fallback events.
EXTERN_LIB void check_backend_fallback(const string& operation,
    BackendId requested, BackendId target, const string& reason);

// Number of cross-backend decisions, including attempts rejected by "error".
// This is not the number of completed fallback computations.
// @pyjt(backend_fallback_count)
uint64 backend_fallback_count();

} // namespace jittor
