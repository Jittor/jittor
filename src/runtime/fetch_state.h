#pragma once
#include <list>
#include "core/var.h"

namespace jittor {

// Fetch roots must exist even when an environment-driven device setter calls
// sync_all during shared-library initialization. NativeRuntime constructs both
// queues on first access; no cross-translation-unit initialization is required.
// Mutations retain the existing serialized runtime/executor requirement.
class RuntimeFetchState {
public:
    using Queue = std::list<VarPtr>;

    RuntimeFetchState() = default;
    RuntimeFetchState(const RuntimeFetchState&) = delete;
    RuntimeFetchState& operator=(const RuntimeFetchState&) = delete;

    Queue& pending() { return pending_; }
    Queue& deferred() { return deferred_; }

private:
    Queue pending_;
    Queue deferred_;
};

EXTERN_LIB RuntimeFetchState& runtime_fetch_state();

} // namespace jittor
