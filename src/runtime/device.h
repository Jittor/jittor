// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "runtime/device_state.h"


#ifdef HAS_ACCELERATOR

namespace jittor {

inline int runtime_use_cuda() { return runtime_device_state().use_cuda; }

// @pyjt(get_device_count)
int get_device_count();

// ---- Device placement -------------------------------------------------
// One process uses every visible device of the selected backend. `device_id` is the *current*
// device: new Vars are placed on it (Var::device_id) and kernels launch on
// it until an op says otherwise. Setting it calls the backend and lets
// every library wrapper swap in that device's handle; it never restarts the
// process, and the other devices stay visible and usable.

// The current device, or -1 when no accelerator is visible.
// @pyjt(current_device)
int current_device();
// Make `device` current through the backend and every registered switch hook.
// @pyjt(set_device)
void set_current_device(int device);

// Registered by the cuDNN/cuBLAS/cuRAND/... wrappers so the one global handle
// their ops read always belongs to the current device. A hook is called once
// on registration for the device that is current then, and after every switch.
void add_device_switch_hook(device_switch_hook_t hook);

// Let `to` read `from`'s memory directly, once per ordered pair. Where the
// hardware cannot peer this does nothing and copies fall back to staging
// through the backend's supported staging path.
void enable_peer_access(int from, int to);

// Synchronize every device in the bitmask (bit d = device d),
// restoring the current device afterwards. An empty mask means "the current
// device only".
void sync_devices(uint64 devices);

} // jittor

#else

namespace jittor {

constexpr int runtime_use_cuda() { return 0; }

inline int get_device_count() { return 0; }
inline int current_device() { return -1; }
inline void set_current_device(int) {}
inline void add_device_switch_hook(device_switch_hook_t) {}
inline void enable_peer_access(int, int) {}
inline void sync_devices(uint64) {}

} // jittor
#endif
