// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <mutex>
#include "misc/cuda_flags.h"
#include "mem/allocator/sfrl_allocator.h"
#include "mem/allocator/cuda_dual_allocator.h"
#include "event_queue.h"
#endif
#include "fetcher.h"
#include "mem/allocator.h"

namespace jittor {

#ifdef HAS_CUDA

#pragma GCC visibility push(hidden)
namespace fetcher_local {

cudaStream_t stream;
cudaEvent_t event;

volatile int64 n_to_fetch;
std::mutex m;
list<FetchResult> fetch_tasks;

static void fetch_caller() {
    fetch_tasks.front().call();
    fetch_tasks.pop_front();
}

static void to_fetch(CUDA_HOST_FUNC_ARGS) {
    event_queue.push(fetch_caller);
}

struct Init {
Init() {
    if (!get_device_count()) return;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    checkCudaErrors(cudaEventCreate(&event, cudaEventDisableTiming));
}
~Init() {
    if (!get_device_count()) return;
    // do not call deleter on exit
    for (auto& f : fetch_tasks)
        f.func.deleter = nullptr;
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaStreamDestroy(stream));
    checkCudaErrors(cudaEventDestroy(event));
}
};

}
using namespace fetcher_local;

#endif

void fetch(const vector<VarHolder*>& vh, FetchFunc&& func) {
    #ifdef HAS_CUDA
    static Init init;
    #endif
    sync(vh);
    vector<Allocation> allocations(vh.size());
    vector<ArrayArgs> arrays(vh.size());
    #ifdef HAS_CUDA
    bool has_cuda_memcpy = false;
    event_queue.flush();
    #endif
    for (int i=0; i<vh.size(); i++) {
        auto v = vh[i]->var;
        auto& allocation = allocations[i];
        #ifdef HAS_CUDA
        if (v->allocator->is_cuda()) {
            checkCudaErrors(cudaEventRecord(event, 0));
            checkCudaErrors(cudaStreamWaitEvent(stream, event, 0));
            new (&allocation) Allocation(&cuda_dual_allocator, v->size);
            // mostly device to device
            checkCudaErrors(cudaMemcpyAsync(
                allocation.ptr, v->mem_ptr, v->size, cudaMemcpyDefault, stream));
            auto host_ptr = cuda_dual_allocator.get_dual_allocation(
                allocation.allocation).host_ptr;
            // device to host
            checkCudaErrors(cudaMemcpyAsync(
                host_ptr, allocation.ptr, v->size, cudaMemcpyDefault, stream));
            allocation.ptr = host_ptr;
            has_cuda_memcpy = true;
        } else
        #endif
        {
            new (&allocation) Allocation(cpu_allocator, v->size);
            std::memcpy(allocation.ptr, v->mem_ptr, v->size);
        }
        arrays[i].ptr = allocation.ptr;
        arrays[i].shape = v->shape;
        arrays[i].dtype = v->dtype();
    }
    #ifdef HAS_CUDA
    if (has_cuda_memcpy) {
        fetch_tasks.push_back({move(func), move(allocations), move(arrays)});
        checkCudaErrors(_cudaLaunchHostFunc(stream, &to_fetch, 0));
    } else
    #endif
    {
        FetchResult fr{move(func), move(allocations), move(arrays)};
        fr.call();
    }
}

} // jittor
