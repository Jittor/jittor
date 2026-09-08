#pragma once
#include "core/executor.h"
#include "runtime/holder_state.h"
#include "runtime/submission_pipeline.h"
#include "runtime/traversal_state.h"
#include "runtime/device_state.h"
#include "runtime/jit_policy.h"
#include "runtime/configuration.h"
#include "runtime/backend.h"
#include "runtime/backend_fallback.h"
#include "runtime/launch_diagnostics.h"

namespace jittor {

// Native execution state shared by the core and dynamically loaded backends.
// State mutations retain the existing serialized runtime requirement.
class NativeRuntime {
public:
    NativeRuntime() = default;
    NativeRuntime(const NativeRuntime&) = delete;
    NativeRuntime& operator=(const NativeRuntime&) = delete;

    Executor& executor() { return executor_; }
    SubmissionPipeline& submissions() { return submissions_; }
    RuntimeHolderState& holders() { return holders_; }
    RuntimeTraversalState& traversals() { return traversals_; }
    RuntimeDeviceState& devices() { return devices_; }
    RuntimeJitPolicy& jit_policy() { return jit_policy_; }
    StartupConfigState& startup_config() { return startup_config_; }
    BackendRegistry& backends() { return backends_; }
    BackendFallbackState& fallbacks() { return fallbacks_; }
    LaunchHistory& launches() { return launches_; }

private:
    Executor executor_;
    SubmissionPipeline submissions_;
    RuntimeHolderState holders_;
    RuntimeTraversalState traversals_;
    RuntimeDeviceState devices_;
    RuntimeJitPolicy jit_policy_;
    StartupConfigState startup_config_;
    BackendRegistry backends_;
    BackendFallbackState fallbacks_;
    LaunchHistory launches_;
};

EXTERN_LIB NativeRuntime& native_runtime();

} // namespace jittor
