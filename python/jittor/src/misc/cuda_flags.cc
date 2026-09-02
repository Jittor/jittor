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
#ifdef __linux__
#include <fstream>
#include <unistd.h>
#endif
#endif

namespace jittor {

DEFINE_FLAG_WITH_SETTER(int, use_cuda, 0,
    "Use cuda or not. 1 for trying to use cuda, 2 for forcing to use cuda.");
DEFINE_FLAG_WITH_SETTER(int, device_id, -1,
    "number of the device to used");
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

static int exec_device = -1;
static uint64 used_devices = 0;
static vector<std::function<void(int)>>& device_hooks() {
    static vector<std::function<void(int)>> hooks;
    return hooks;
}

int default_cuda_device() {
    return device_id < 0 ? 0 : device_id;
}

int current_cuda_device() {
    return exec_device;
}

void switch_cuda_device(int device) {
#ifdef HAS_CUDA
    if (device == exec_device) return;
    CHECK(device >= 0 && device < get_device_count())
        << "cuda:" << device << "is not a visible device," << get_device_count() << "visible";
    checkCudaErrors(cudaSetDevice(device));
    exec_device = device;
    used_devices |= 1ull << device;
    for (auto& hook : device_hooks())
        hook(device);
#endif
}

void register_device_switch_hook(std::function<void(int)> hook) {
#ifdef HAS_CUDA
    if (!get_device_count()) return;
    device_hooks().push_back(hook);
    if (exec_device < 0)
        // Runs every hook, this one included, for the initial device.
        switch_cuda_device(default_cuda_device());
    else
        hook(exec_device);
#endif
}

void synchronize_all_devices() {
#ifdef HAS_CUDA
    if (!use_cuda) return;
    if (exec_device < 0 || used_devices == (1ull << exec_device)) {
        checkCudaErrors(cudaDeviceSynchronize());
        return;
    }
    for (int i = 0; i < 64; i++)
        if (used_devices & (1ull << i)) {
            checkCudaErrors(cudaSetDevice(i));
            checkCudaErrors(cudaDeviceSynchronize());
        }
    checkCudaErrors(cudaSetDevice(exec_device));
#endif
}

void setter_device_id(int value) {
    // The current device, as torch.cuda.set_device: new Vars are placed on it
    // and it becomes the CUDA current device. Other devices stay visible and
    // reachable through Var.to_device; nothing is re-executed.
    if (value < 0) {
        device_id = value;
        return;
    }
#ifdef HAS_CUDA
    int count = get_device_count();
    CHECK(count == 0 || value < count)
        << "device_id" << value << "is out of range:" << count << "CUDA devices visible";
    device_id = value;
    if (count) switch_cuda_device(value);
#else
    device_id = value;
#endif
}

} // jittor