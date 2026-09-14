#include "runtime/device.h"
#include <algorithm>
#include <stdexcept>

namespace jittor {

uint64 push_device_mode_scope() {
    auto& state = runtime_device_state();
    const auto token = ++state.next_mode_scope_token;
    state.mode_scope_snapshots.push_back({token, state.use_cuda});
    return token;
}

void pop_device_mode_scope(uint64 token, bool restore) {
    auto& state = runtime_device_state();
    auto snapshot = std::find_if(
        state.mode_scope_snapshots.begin(), state.mode_scope_snapshots.end(),
        [token](const RuntimeDeviceState::ModeScopeSnapshot& saved) {
            return saved.token == token;
        });
    if (snapshot == state.mode_scope_snapshots.end())
        throw std::invalid_argument("device mode scope token is not an active snapshot");
    const int previous = snapshot->use_cuda;
    // Python threads can interleave equal-mode scopes without changing device
    // policy. Each exit consumes its own snapshot, not a process-wide stack top.
    state.mode_scope_snapshots.erase(snapshot);
    if (restore) {
        // Unlike a requested mode change, rollback must not submit failed
        // pending graphs. setter_use_cuda does not switch the active device or
        // library handles, and the accelerator backend is fixed at build time.
        // Other scoped flags (including device_id) retain their own setters.
        state.use_cuda = previous;
    }
}

} // namespace jittor
