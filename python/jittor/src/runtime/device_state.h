#pragma once
#include "common.h"

namespace jittor {

using device_switch_hook_t = void (*)(int);

// Device policy and bookkeeping share the native runtime's lifetime.
struct RuntimeDeviceState {
    int use_cuda = 0;
    int device_id = -1;
    int sync_run = 1;
    int device_count = -1;
    int current_device = -1;
    vector<device_switch_hook_t> switch_hooks;
    vector<char> peer_enabled;
};

EXTERN_LIB RuntimeDeviceState& runtime_device_state();
DECLARE_RUNTIME_FLAG(int, use_cuda);
DECLARE_RUNTIME_FLAG(int, device_id);
DECLARE_RUNTIME_FLAG(int, sync_run);

} // namespace jittor
