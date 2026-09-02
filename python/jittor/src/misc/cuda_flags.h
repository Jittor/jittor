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

// The CUDA device new Vars are placed on and kernels are launched to until
// an op says otherwise; see Var::device_id. Setting it is cheap and does
// not restart the process: it calls cudaSetDevice and lets every library
// wrapper swap in that device's handle through a registered hook.
// @pyjt(current_device)
int current_device();
// @pyjt(set_device)
void set_current_device(int device);

typedef void (*device_switch_hook_t)(int device);
// Registered by the cuDNN/cuBLAS/... wrappers so the global handle each op
// uses always belongs to the current device. Called after cudaSetDevice.
void add_device_switch_hook(device_switch_hook_t hook);

// Make peer memory access between two devices available once. When the
// pair cannot peer this does nothing and copies go through the host.
void enable_peer_access(int from, int to);

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

} // jittor
#endif
