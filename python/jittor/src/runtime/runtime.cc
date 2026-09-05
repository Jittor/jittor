#include "runtime/runtime.h"

namespace jittor {

NativeRuntime& native_runtime() {
    // Backends and static holders may access state during late teardown.
    // Keep this core-owned instance alive until the process exits.
    static auto* state = new NativeRuntime();
    return *state;
}

Executor& runtime_executor() {
    return native_runtime().executor();
}

} // namespace jittor
