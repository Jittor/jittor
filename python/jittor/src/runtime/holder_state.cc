#include "runtime/runtime.h"

namespace jittor {

RuntimeHolderState& runtime_holder_state() {
    return native_runtime().holders();
}

} // namespace jittor
