// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "runtime/device.h"
#include "runtime/backend.h"

namespace jittor {

DEFINE_RUNTIME_FLAG_WITH_SETTER(int, use_cuda, 0,
    "Use cuda or not. 1 for trying to use cuda, 2 for forcing to use cuda.");
DEFINE_RUNTIME_FLAG_WITH_SETTER(int, device_id, -1,
    "Current accelerator device for new Vars; setting it switches device in place without restarting the process.");
DEFINE_RUNTIME_FLAG(int, sync_run, 1, "Enable per-op-sync or not");

EXTERN_LIB void sync_all(bool device_sync);

#ifdef HAS_ACCELERATOR
int get_device_count() { return backend_ops(accelerator_backend_id()).device_count(); }
int current_device() { return backend_ops(accelerator_backend_id()).current_device(); }
void set_current_device(int device) { backend_ops(accelerator_backend_id()).set_device(device); }
void enable_peer_access(int from, int to) { backend_ops(accelerator_backend_id()).enable_peer(from, to); }
void sync_devices(uint64 devices) { backend_ops(accelerator_backend_id()).synchronize(devices); }

void add_device_switch_hook(device_switch_hook_t hook) {
    if (!get_device_count()) return;
    runtime_device_state().switch_hooks.push_back(hook);
    int device = current_device();
    if (device >= 0) hook(device);
}
#endif

void setter_use_cuda(const int& old_value, const int& requested) {
    if (old_value == requested) return;
    int value = requested;
#ifdef HAS_ACCELERATOR
    if (value) {
        if (backend_ops(accelerator_backend_id()).device_count() == 0) {
            LOGw << "No CUDA device available; falling back to CPU (use_cuda=0).";
            value = 0;
        } else {
            LOGi << "CUDA enabled.";
            current_device();
        }
    } else {
        LOGv << "CUDA disabled.";
    }
#else
    CHECK(value == 0) << "No CUDA found.";
#endif
    if (old_value != value) {
        // Pending graphs were prepared for the old backend. The generated
        // setter rolls back to this value if submission throws.
        runtime_device_state().use_cuda = old_value;
        sync_all(false);
    }
    runtime_device_state().use_cuda = value;
}

void setter_device_id(const int& old_value, const int& value) {
#ifdef HAS_ACCELERATOR
    if (value < 0) return;
    if (!get_device_count()) {
        LOGw << "No CUDA device available; ignoring device_id" << value;
        return;
    }
    set_current_device(value);
#else
    CHECK(value < 0) << "No CUDA found.";
#endif
}

} // namespace jittor
