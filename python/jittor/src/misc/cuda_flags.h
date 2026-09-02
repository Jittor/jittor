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
#include <functional>

namespace jittor {

DECLARE_FLAG(int, use_cuda);
DECLARE_FLAG(int, device_id);
DECLARE_FLAG(int, sync_run);

// @pyjt(get_device_count)
int get_device_count();

// ---- Device selection -------------------------------------------------
// Every Var carries the CUDA device it lives on (Var::cuda_device); a new
// Var takes the current default, ``jt.flags.device_id`` (0 when unset).
// The executor makes a Var's device current before touching it, and
// libraries that keep per-device state (handles, streams) bind it through
// a switch hook. Nothing here restarts the process.

// The device new Vars are placed on: jt.flags.device_id, or 0 when unset.
int default_cuda_device();
// The device the executor most recently made current, -1 before any switch.
int current_cuda_device();
// cudaSetDevice(device) plus the registered hooks; no-op when already current.
void switch_cuda_device(int device);
// Register a hook called after every device switch, and once right away for
// the device that is (or becomes) current, so per-device state is bound
// before it is used.
void register_device_switch_hook(std::function<void(int)> hook);
// cudaDeviceSynchronize on every device the executor has used so far.
void synchronize_all_devices();

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

inline int get_device_count() { return 0; }
inline int default_cuda_device() { return 0; }
inline int current_cuda_device() { return -1; }
inline void switch_cuda_device(int) {}
inline void synchronize_all_devices() {}
template <class F> inline void register_device_switch_hook(F&&) {}

} // jittor
#endif
