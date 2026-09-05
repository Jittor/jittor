#include "runtime/holder_state.h"

namespace jittor {

RuntimeHolderState& runtime_holder_state() {
    // Extension/static holders may unregister during process teardown after
    // ordinary function-local statics have been destroyed. Keep the registry
    // (not the holders) alive until process exit, shared through this core API.
    static auto* state = new RuntimeHolderState();
    return *state;
}

} // namespace jittor
