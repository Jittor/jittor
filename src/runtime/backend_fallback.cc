#include "runtime/backend_fallback.h"

namespace jittor {

DEFINE_RUNTIME_FLAG_WITH_SETTER(string, backend_fallback, "warn",
    "Backend fallback policy: error rejects unsupported kernels, warn reports fallback, allow permits fallback without warning.");

void setter_backend_fallback(const string&, const string& requested) {
    USER_CHECK(requested == "error" || requested == "warn" || requested == "allow")
        << "backend_fallback must be error, warn, or allow";
}

void check_backend_fallback(const string& operation, BackendId requested,
                           BackendId target, const string& reason) {
    if (requested == target) return;
    const auto& source = backend_ops(requested);
    const auto& destination = backend_ops(target);
    auto& state = runtime_backend_fallback();
    ++state.count;
    if (state.policy == "allow") return;
    const string message = "Backend fallback: op=" + operation
        + " backend=" + source.name + " target=" + destination.name
        + " reason=" + reason;
    USER_CHECK(state.policy == "warn") << message;
    LOGw << message;
}

uint64 backend_fallback_count() { return runtime_backend_fallback().count; }

} // namespace jittor
