#pragma once
#include "executor.h"
#include "runtime/holder_state.h"
#include "runtime/traversal_state.h"
#include "runtime/device_state.h"
#include "runtime/jit_policy.h"
#include "runtime/configuration.h"

namespace jittor {

// Native execution state shared by the core and dynamically loaded backends.
// State mutations retain the existing serialized runtime requirement.
class NativeRuntime {
public:
    NativeRuntime() = default;
    NativeRuntime(const NativeRuntime&) = delete;
    NativeRuntime& operator=(const NativeRuntime&) = delete;

    Executor& executor() { return executor_; }
    RuntimeHolderState& holders() { return holders_; }
    RuntimeTraversalState& traversals() { return traversals_; }
    RuntimeDeviceState& devices() { return devices_; }
    RuntimeJitPolicy& jit_policy() { return jit_policy_; }
    StartupConfigState& startup_config() { return startup_config_; }

private:
    Executor executor_;
    RuntimeHolderState holders_;
    RuntimeTraversalState traversals_;
    RuntimeDeviceState devices_;
    RuntimeJitPolicy jit_policy_;
    StartupConfigState startup_config_;
};

EXTERN_LIB NativeRuntime& native_runtime();

} // namespace jittor
