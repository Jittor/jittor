// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include "common.h"
#include "misc/cuda_flags.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#endif

namespace jittor {

DEFINE_FLAG_WITH_SETTER(int, use_cuda, 0,
    "Use cuda or not. 1 for trying to use cuda, 2 for forcing to use cuda.");
DEFINE_FLAG_WITH_SETTER(int, device_id, -1,
    "The CUDA device new Vars are placed on, torch.cuda.current_device. Setting it switches the current device in place; it never restarts the process. It reads -1 until a device has been set or queried; jt.current_device() is always the truth.");
DEFINE_FLAG_WITH_SETTER(int, sync_run, 1,
    "Enable per-op-sync or not");

EXTERN_LIB void sync_all(bool device_sync);

#ifdef HAS_CUDA
int get_device_count() {
    static int count=-1;
    if (count==-1) {
        // cudaGetDeviceCount returns cudaErrorNoDevice (and may leave `count`
        // untouched at -1) when no GPU is visible (e.g. CUDA_VISIBLE_DEVICES="").
        // Treat any error as 0 devices so callers (the array_op static Init,
        // setter_use_cuda) take the CPU path instead of aborting on a CUDA call.
        if (cudaGetDeviceCount(&count) != cudaSuccess)
            count = 0;
    }
    return count;
}
#endif

void setter_sync_run(int value) {
    if(sync_run == value) return;
    sync_run = value;
}

void setter_use_cuda(int value) {
    if (use_cuda == value) return;
#ifdef HAS_CUDA
    if (value) {
        int count=0;
        cudaGetDeviceCount(&count);
        if (count == 0) {
            // No CUDA device visible at runtime (e.g. CUDA_VISIBLE_DEVICES="" in
            // a CPU-only process such as a Ray orchestrator actor with
            // num_gpus=0). Fall back to CPU instead of aborting, so importing
            // jittor / the torch-shim does not crash where there is no GPU.
            LOGw << "No CUDA device available; falling back to CPU (use_cuda=0).";
            value = 0;
        } else {
            LOGi << "CUDA enabled.";
        }
    } else {
        LOGv << "CUDA disabled.";
    }
#else
    CHECK(value==0) << "No CUDA found.";
#endif
    if (use_cuda != value)
        sync_all(0);
    // jtorch will call this directly
    use_cuda = value;
}

#ifdef HAS_CUDA
static int cur_device = -1;   // -1: not yet queried from the runtime
static vector<device_switch_hook_t> device_switch_hooks;

int current_device() {
    if (cur_device < 0) {
        int d = 0;
        if (get_device_count() <= 0 || cudaGetDevice(&d) != cudaSuccess) {
            cudaGetLastError();
            return -1;
        }
        cur_device = d;
        device_id = d;
    }
    return cur_device;
}

void add_device_switch_hook(device_switch_hook_t hook) {
    device_switch_hooks.push_back(hook);
    // The hook must see the device that is current right now.
    int d = current_device();
    if (d >= 0) hook(d);
}

void set_current_device(int device) {
    int count = get_device_count();
    CHECK(device >= 0 && device < count)
        << "Invalid CUDA device index" << device << ", device count is" << count;
    int cur = current_device();
    // Keep the flag readable as the current device even when nothing moves.
    device_id = device;
    if (device == cur) return;
    checkCudaErrors(cudaSetDevice(device));
    cur_device = device;
    for (auto hook : device_switch_hooks) hook(device);
}

void enable_peer_access(int from, int to) {
    static vector<char> enabled;  // (from, to) pairs already handled
    if (from == to || from < 0 || to < 0) return;
    int n = get_device_count();
    if (enabled.size() < (size_t)n*n) enabled.resize(n*n, 0);
    auto& done = enabled[from*n+to];
    if (done) return;
    done = 1;
    int can = 0;
    if (cudaDeviceCanAccessPeer(&can, to, from) != cudaSuccess || !can) {
        cudaGetLastError();
        return;
    }
    int prev = current_device();
    checkCudaErrors(cudaSetDevice(to));
    auto err = cudaDeviceEnablePeerAccess(from, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled)
        LOGw << "cudaDeviceEnablePeerAccess(" << from << "->" << to << ") failed:" << cudaGetErrorString(err);
    cudaGetLastError();
    checkCudaErrors(cudaSetDevice(prev));
}
#endif

void setter_device_id(int value) {
#ifdef HAS_CUDA
    // Below zero is "unset": the runtime default stays whatever device is
    // current, which the flag reports once it has been queried or set.
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

} // jittor