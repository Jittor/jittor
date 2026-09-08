// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <map>
#include <memory>
#include "stream_compat.h"
#include "cudnn_wrapper.h"
#include "runtime/device.h"
#include "cudnn_conv_plan.h"
#include "cache_device_scope.h"

namespace jittor {

cudnnHandle_t cudnn_handle = nullptr;
int max_cache_size = 100;
float max_workspace_ratio = 0.25;
int cudnn_benchmark = -1;

struct CudnnDeviceState {
    int device;
    cudnnHandle_t handle = nullptr;
    uint64 binds = 0, plan_destroys = 0;
    ConvAlgoCache<cudnnConvolutionFwdAlgo_t> fwd;
    ConvAlgoCache<cudnnConvolutionBwdDataAlgo_t> bwdx;
    ConvAlgoCache<cudnnConvolutionBwdFilterAlgo_t> bwdw;
    ConvPlanCache plans;
    explicit CudnnDeviceState(int device) : device(device) {}
    ~CudnnDeviceState() {
        // Plans capture this device's handle; release their descriptors first.
        plans.clear();
        if (handle) {
            CacheDeviceScope scope(device, true);
            if (scope.active) peekCudaErrorsAlways(cudnnDestroy(handle));
        }
    }
};
static std::map<int, std::unique_ptr<CudnnDeviceState>> cudnn_devices;

static CudnnDeviceState& cudnn_state(int device) {
    auto found = cudnn_devices.find(device);
    if (found != cudnn_devices.end()) return *found->second;
    auto state = std::unique_ptr<CudnnDeviceState>(new CudnnDeviceState(device));
    auto* result = state.get();
    cudnn_devices.emplace(device, std::move(state));
    return *result;
}

static void cudnn_switch_device(int device) {
    auto& state = cudnn_state(device);
    if (!state.handle) {
        checkCudaErrors(cudnnCreate(&state.handle));
        LOGv << "cudnnCreate finished for device" << device;
    }
    cudnn_handle = state.handle;
}

cudnnHandle_t cudnn_bind_stream() {
    int device = current_device();
    cudnn_switch_device(device);
    auto& state = cudnn_state(device);
    checkCudaErrors(cudnnSetStream(state.handle, cuda_compute_stream(device)));
    state.binds++;
    return state.handle;
}

uint64 cudnn_stream_bind_count(int device) {
    ExecutorEntryScope entry;
    auto found = cudnn_devices.find(device);
    return found == cudnn_devices.end() ? 0 : found->second->binds;
}

ConvAlgoCache<cudnnConvolutionFwdAlgo_t>& cudnn_fwd_algo_cache() {
    return cudnn_state(current_device()).fwd;
}
ConvAlgoCache<cudnnConvolutionBwdDataAlgo_t>& cudnn_bwdx_algo_cache() {
    return cudnn_state(current_device()).bwdx;
}
ConvAlgoCache<cudnnConvolutionBwdFilterAlgo_t>& cudnn_bwdw_algo_cache() {
    return cudnn_state(current_device()).bwdw;
}
ConvPlanCache& conv_plan_cache() {
    return cudnn_state(current_device()).plans;
}

void initialize_conv_plan_owner(ConvPlanEntry& entry) {
    entry.device = current_device();
    entry.stream = cuda_compute_stream(entry.device);
    entry.destroy_counter = &cudnn_state(entry.device).plan_destroys;
}

int cudnn_algorithm_cache_size(int device) {
    ExecutorEntryScope entry;
    int size = 0;
    for (const auto& bank : cudnn_devices)
        if (device < 0 || bank.first == device)
            size += bank.second->fwd.size() + bank.second->bwdx.size() + bank.second->bwdw.size();
    return size;
}

void cudnn_clear_algorithm_cache(int device) {
    ExecutorEntryScope entry;
    for (auto& bank : cudnn_devices) {
        if (device >= 0 && bank.first != device) continue;
        bank.second->fwd.clear(); bank.second->bwdx.clear(); bank.second->bwdw.clear();
    }
}

int cudnn_plan_cache_size(int device) {
    ExecutorEntryScope entry;
    int size = 0;
    for (const auto& bank : cudnn_devices)
        if (device < 0 || bank.first == device)
            for (const auto& entry : bank.second->plans)
                if (entry.second.valid && entry.second.plan) size++;
    return size;
}

uint64 cudnn_plan_destroy_count(int device) {
    ExecutorEntryScope entry;
    uint64 count = 0;
    for (const auto& bank : cudnn_devices)
        if (device < 0 || bank.first == device) count += bank.second->plan_destroys;
    return count;
}

void cudnn_clear_plan_cache(int device) {
    ExecutorEntryScope entry;
    for (auto& bank : cudnn_devices)
        if (device < 0 || bank.first == device) bank.second->plans.clear();
}

void set_algorithm_cache_size(int size) {
    ExecutorEntryScope entry;
    USER_CHECK(size >= 0) << "cuDNN algorithm cache size must be nonnegative";
    max_cache_size = size;
    for (auto& bank : cudnn_devices) {
        if (bank.second->fwd.size() > (size_t)size) bank.second->fwd.clear();
        if (bank.second->bwdx.size() > (size_t)size) bank.second->bwdx.clear();
        if (bank.second->bwdw.size() > (size_t)size) bank.second->bwdw.clear();
    }
}

void set_max_workspace_ratio(float64 ratio) { max_workspace_ratio = ratio; }
void set_benchmark(int enabled) { cudnn_benchmark = enabled < 0 ? -1 : (enabled ? 1 : 0); }
int get_benchmark() { return cudnn_benchmark; }

void cudnn_shutdown() {
    cudnn_devices.clear();
    cudnn_handle = nullptr;
    LOGv << "cudnnDestroy finished";
}

struct cudnn_initer {
    cudnn_initer() {
        if (get_device_count()) add_device_switch_hook(cudnn_switch_device);
    }
    ~cudnn_initer() { cudnn_shutdown(); }
} init;

} // namespace jittor
