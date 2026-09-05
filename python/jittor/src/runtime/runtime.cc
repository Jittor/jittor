#include "runtime/runtime.h"

namespace jittor {

RuntimeDeviceState& runtime_device_state() {
    return native_runtime().devices();
}

int& runtime_flag_use_cuda() { return runtime_device_state().use_cuda; }
int& runtime_flag_device_id() { return runtime_device_state().device_id; }
int& runtime_flag_sync_run() { return runtime_device_state().sync_run; }

NativeRuntime& native_runtime() {
    // Backends and static holders may access state during late teardown.
    // Keep this core-owned instance alive until the process exits.
    static auto* state = new NativeRuntime();
    return *state;
}

Executor& runtime_executor() {
    return native_runtime().executor();
}

RuntimeTraversalState& runtime_traversal_state() {
    return native_runtime().traversals();
}

} // namespace jittor
