// ***************************************************************
// Copyright (c) 2021 Jittor. 
// All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <mutex>
#include "misc/cuda_flags.h"
#include "mem/allocator/sfrl_allocator.h"
#include "mem/allocator/cuda_dual_allocator.h"
#include "event_queue.h"
#endif
#include "ops/fetch_op.h"
#include "mem/allocator.h"
#include "executor.h"

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
    peekCudaErrors(cudaDeviceSynchronize());
    peekCudaErrors(cudaStreamDestroy(stream));
    peekCudaErrors(cudaEventDestroy(event));
}
} ;

}
using namespace fetcher_local;

#endif

list<VarPtr> fetcher;
// this list will be free at each execution
list<VarPtr> fetcher_to_free;

FetchOp::FetchOp(vector<Var*>&& inputs, FetchFunc&& func) 
: fetch_vars(inputs), func(move(func)) {
    #ifdef HAS_CUDA
    // stream needs to be created after nccl plugin
    static Init init_fetch;
    #endif
    VarPtr vp(0, ns_int32);
    outputs_holder.emplace_back(vp);
    fetcher.emplace_front(move(vp));
    fetcher_iter = fetcher.begin();
    bool all_finished = true;
    for (auto v : fetch_vars)
        if (!v->is_finished()) {
            all_finished = false;
            v->flags.set(NodeFlags::_stop_fuse);
            v->flags.set(NodeFlags::_fetch);
        }
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    flags.set(NodeFlags::_fetch);
    flags.set(NodeFlags::_stop_grad);
    fetcher_iter->ptr->flags.set(NodeFlags::_fetch);
    // fetcher_to_free.clear();
    if (all_finished) {
        // if all finished, run immediately
        run();
    }
    // if too many fetchers are bufferd, force flush
    while (fetcher.size() > 20) {
        LOGvvvv << "too many fetchers(">>fetcher.size() >> 
            ") are bufferd, force flush";
        exe.run_sync({fetcher.back().ptr}, false);
    }
}

void FetchOp::run() {
    vector<Allocation> allocations(fetch_vars.size());
    vector<ArrayArgs> arrays(fetch_vars.size());
    #ifdef HAS_CUDA
    bool has_cuda_memcpy = false;
    event_queue.flush();
    #endif
    LOGvvvv << "fetch" << fetch_vars.size() << "vars" << fetch_vars;
    int i = 0;
    for (auto v : fetch_vars) {    
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
        i++;
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
    fetcher_to_free.emplace_front(move(*fetcher_iter));
    fetcher.erase(fetcher_iter);
}

} // jittor
