// ***************************************************************
// Copyright (c) 2023 Jittor. 
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
// `event` only orders the fetch stream *after* the default stream. The other
// direction is missing: the staging copies below read the source vars on the
// fetch stream, and those vars are released as soon as the next run_sync
// clears fetcher_to_free -- which it does *before* any device sync. Nothing
// records that a copy is still in flight, so the blocks come straight back out
// of the free list and the next kernels overwrite them mid-copy. The fix is to
// hold a reference on the source blocks (see FetchOp::run); this event is the
// fallback for allocators that cannot express one block with two owners.
cudaEvent_t copy_done_event;

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
    checkCudaErrors(cudaEventCreate(&copy_done_event, cudaEventDisableTiming));
    // stream = aclstream;
}
~Init() {
    if (!get_device_count()) return;
    // do not call deleter on exit
    for (auto& f : fetch_tasks)
        f.func.deleter = nullptr;
    peekCudaErrors(cudaDeviceSynchronize());
    peekCudaErrors(cudaStreamDestroy(stream));
    peekCudaErrors(cudaEventDestroy(event));
    peekCudaErrors(cudaEventDestroy(copy_done_event));
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
    // References taken on the source vars' blocks so they cannot be handed out
    // again while the staging copies are still queued; they ride along in the
    // fetch task and are released with it, after the callback has run.
    vector<Allocation> pinned;
    // Set when some source could not be pinned, and the default stream has to
    // be held back instead.
    bool need_copy_fence = false;
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
            // This staging copy is the only read of the source var's own
            // memory; the device-to-host leg reads the staging block instead,
            // which is why the two legs are separate loops now.
            #if IS_CUDA
            checkCudaErrors(cudaMemcpyAsync(
                allocation.ptr, v->mem_ptr, v->size, cudaMemcpyDefault, stream));
            // checkCudaErrors(cudaMemcpyAsync(
            //     allocation.ptr, v->size, v->mem_ptr, v->size, cudaMemcpyDefault, aclstream));
            // checkCudaErrors(aclrtSynchronizeStream(aclstream));
            #else
            checkCudaErrors(cudaMemcpyAsync(
                allocation.ptr, v->mem_ptr, v->size, cudaMemcpyDeviceToDevice, stream));
            #endif
            // The copy is queued, not done. Keep the source block reserved
            // until this fetch task is destroyed, which happens after the host
            // callback and so after the copy. Holding memory is much cheaper
            // than the alternative below, which would stop the default stream
            // from running ahead at all -- the very overlap fetch exists for.
            if (v->allocator->can_share()) {
                v->allocator->share_with(v->size, v->allocation);
                pinned.emplace_back(v->mem_ptr, v->allocation, v->size,
                                    v->allocator);
            } else
                need_copy_fence = true;
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
        if (PREDICT_BRANCH_NOT_TAKEN(need_copy_fence)) {
            // Some source could not be pinned (an allocator with no notion of
            // sharing), so the only way left to keep its block from being
            // handed out mid-copy is to hold the default stream back until the
            // staging copies have run. Recorded before the device-to-host leg
            // so at least the PCIe transfers stay off the critical path.
            checkCudaErrors(cudaEventRecord(copy_done_event, stream));
            checkCudaErrors(cudaStreamWaitEvent(0, copy_done_event, 0));
        }
        for (uint j=0; j<allocations.size(); j++) {
            auto& allocation = allocations[j];
            if (allocation.allocator != (Allocator*)&cuda_dual_allocator)
                continue;
            auto host_ptr = cuda_dual_allocator.get_dual_allocation(
                allocation.allocation).host_ptr;
            // device to host
            checkCudaErrors(cudaMemcpyAsync(host_ptr, allocation.ptr,
                allocation.size, cudaMemcpyDeviceToHost, stream));
            // checkCudaErrors(aclrtMemcpyAsync(
            //     host_ptr, v->size, allocation.ptr, v->size, cudaMemcpyDeviceToHost, aclstream));
            // checkCudaErrors(aclrtSynchronizeStream(aclstream));
            allocation.ptr = host_ptr;
            arrays[j].ptr = host_ptr;
        }
        // appended last: the loop above must not mistake them for staging
        for (auto& p : pinned)
            allocations.emplace_back(move(p));
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
