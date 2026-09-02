// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"


#ifdef HAS_CUDA
#include <cuda_runtime.h>

namespace jittor {

DECLARE_FLAG(int, use_cuda);
DECLARE_FLAG(int, sync_run);
DECLARE_FLAG(int, device_id);

// @pyjt(get_device_count)
int get_device_count();

// ---- Device placement -------------------------------------------------
// One process uses every visible CUDA device. `device_id` is the *current*
// device: new Vars are placed on it (Var::device_id) and kernels launch on
// it until an op says otherwise. Setting it calls cudaSetDevice and lets
// every library wrapper swap in that device's handle; it never restarts the
// process, and the other devices stay visible and usable.

// The current device, or -1 when no CUDA device is visible.
// @pyjt(current_device)
int current_device();
// Make `device` current: cudaSetDevice plus every registered switch hook.
// @pyjt(set_device)
void set_current_device(int device);

typedef void (*device_switch_hook_t)(int device);
// Registered by the cuDNN/cuBLAS/cuRAND/... wrappers so the one global handle
// their ops read always belongs to the current device. A hook is called once
// on registration for the device that is current then, and after every switch.
void add_device_switch_hook(device_switch_hook_t hook);

// Let `to` read `from`'s memory directly, once per ordered pair. Where the
// hardware cannot peer this does nothing and copies fall back to staging
// through the host, which cudaMemcpy does on its own.
void enable_peer_access(int from, int to);

// cudaDeviceSynchronize on every device in the bitmask (bit d = device d),
// restoring the current device afterwards. An empty mask means "the current
// device only".
void sync_devices(uint64 devices);

} // jittor

#if defined(CUDART_VERSION) && CUDART_VERSION < 10000
    #define _cudaLaunchHostFunc(a,b,c) \
        cudaStreamAddCallback(a,b,c,0)
    #define CUDA_HOST_FUNC_ARGS cudaStream_t stream, cudaError_t status, void*
#else
    #define _cudaLaunchHostFunc(a,b,c) \
        cudaLaunchHostFunc(a,b,c)
    #define CUDA_HOST_FUNC_ARGS void*
#endif

#else

namespace jittor {

constexpr int use_cuda = 0;
constexpr int device_id = -1;

inline int get_device_count() { return 0; }
inline int current_device() { return -1; }
inline void set_current_device(int) {}
typedef void (*device_switch_hook_t)(int device);
inline void add_device_switch_hook(device_switch_hook_t) {}
inline void enable_peer_access(int, int) {}
inline void sync_devices(uint64) {}

} // jittor
#endif
