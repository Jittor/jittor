#pragma once
#include "executor.h"
#include "runtime/holder_state.h"

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

private:
    Executor executor_;
    RuntimeHolderState holders_;
};

EXTERN_LIB NativeRuntime& native_runtime();

} // namespace jittor
