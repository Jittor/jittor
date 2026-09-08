// ***************************************************************
// Copyright (c) 2023 Jittor. 
// All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_ACCELERATOR
#include <exception>
#include <mutex>
#include "runtime/device.h"
#include "runtime/backend_streams.h"
#include "mem/allocator/sfrl_allocator.h"
#include "mem/allocator/cuda_dual_allocator.h"
#include "core/event_queue.h"
#endif
#include "ops/composite/fetch_op.h"
#include "mem/allocator.h"
#include "core/executor.h"
#include "runtime/backend.h"

namespace jittor {

#ifdef HAS_ACCELERATOR

#pragma GCC visibility push(hidden)
namespace fetcher_local {

volatile int64 n_to_fetch;
std::mutex m;
list<FetchResult> fetch_tasks;

static void fetch_caller() {
    fetch_tasks.front().call();
    fetch_tasks.pop_front();
}

static void to_fetch(void* user_data) {
    event_queue.push(fetch_caller);
}

struct Init {
Init() {
    if (!get_device_count()) return;
    backend_stream({accelerator_backend_id(), 0}, BackendStreamKind::Copy);
}
~Init() {
    if (!get_device_count()) return;
    // do not call deleter on exit
    for (auto& f : fetch_tasks)
        f.func.deleter = nullptr;
    try {
        backend_synchronize({accelerator_backend_id(), current_device()});
    } catch (const std::exception& error) {
        LOGe << "Fetch shutdown synchronization failed:" << error.what();
    }
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
    #ifdef HAS_ACCELERATOR
    // Side streams are lazy so CUDA is initialized before they are created.
    static Init init_fetch;
    #endif
    VarPtr vp(0, ns_int32);
    outputs_holder.emplace_back(vp);
    fetcher.emplace_front(move(vp));
    fetcher_iter = fetcher.begin();
    for (auto v : fetch_vars)
        if (!v->is_finished()) {
            v->set_flag(VarFlags::_stop_fuse);
            v->flags.set(NodeFlags::_fetch);
        }
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_device);
    flags.set(NodeFlags::_fetch);
    flags.set(NodeFlags::_stop_grad);
    fetcher_iter->ptr->flags.set(NodeFlags::_fetch);
}

// Called only after the Python fetch constructor has returned and the graph
// edges are installed. Callback execution never observes a half-built Op.
void submit_pending_fetches() {
    if (!fetcher.empty()) {
        auto target = fetcher.front();
        auto* op = target->input();
        bool ready = op != nullptr;
        if (op)
            for (auto* v : op->inputs()) ready &= v->is_finished();
        if (ready) runtime_executor().run_sync({target.ptr}, false, false);
    }
    while (fetcher.size() > 20) {
        LOGvvvv << "too many fetchers(">>fetcher.size() >> 
            ") are bufferd, force flush";
        runtime_executor().run_sync({fetcher.back().ptr}, false, false);
    }
}

void FetchOp::run() {
    vector<Allocation> allocations(fetch_vars.size());
    vector<ArrayArgs> arrays(fetch_vars.size());
    #ifdef HAS_ACCELERATOR
    bool has_device_copy = false;
    // References taken on the source vars' blocks so they cannot be handed out
    // again while the staging copies are still queued; they ride along in the
    // fetch task and are released with it, after the callback has run.
    vector<Allocation> pinned;
    // Set when some source could not be pinned, and the default stream has to
    // be held back instead.
    bool need_copy_fence = false;
    // Devices this fetch read from, and the device to come back to.
    uint64 src_devices = 0;
    int entry_device = current_device();
    int copy_device = 0;
    auto copy_stream = backend_stream(
        {accelerator_backend_id(), copy_device}, BackendStreamKind::Copy);
    event_queue.flush();
    #endif
    LOGvvvv << "fetch" << fetch_vars.size() << "vars" << fetch_vars;
    int i = 0;
    for (auto v : fetch_vars) {    
        auto& allocation = allocations[i];

        #ifdef HAS_ACCELERATOR
        if (v->allocator->is_cuda()) {
            // The event that orders this fetch after the kernels that
            // produced v has to be recorded on v's *own* device: stream 0 is
            // whichever device is current, and an event of another device
            // cannot be recorded on it at all.
            auto source = allocation_device(v->allocator);
            int src = source.index;
            if (src >= 0) {
                if (src != current_device()) set_current_device(src);
                if (src < 64) src_devices |= 1ull << src;
            }
            backend_side_stream_wait_default(
                BackendStreamKind::Copy, copy_device, src);
            new (&allocation) Allocation(&cuda_dual_allocator, v->size);
            // mostly device to device
            // This staging copy is the only read of the source var's own
            // memory; the device-to-host leg reads the staging block instead,
            // which is why the two legs are separate loops now.
            Device target{accelerator_backend_id(), cuda_dual_device_allocator.device()};
            backend_copy_async(allocation.ptr, target, v->mem_ptr, source, v->size,
                copy_stream);
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
            has_device_copy = true;
        } else
        #endif
        {
            new (&allocation) Allocation(cpu_allocator, v->size);
            backend_copy(allocation.ptr, {}, v->mem_ptr, {}, v->size);
        }
        arrays[i].ptr = allocation.ptr;
        arrays[i].shape = v->shape;
        arrays[i].dtype = v->dtype();
        i++;
    }
    #ifdef HAS_ACCELERATOR
    if (has_device_copy) {
        if (PREDICT_BRANCH_NOT_TAKEN(need_copy_fence)) {
            // Some source could not be pinned (an allocator with no notion of
            // sharing), so the only way left to keep its block from being
            // handed out mid-copy is to hold the default stream back until the
            // staging copies have run. Recorded before the device-to-host leg
            // so at least the PCIe transfers stay off the critical path.
            // Hold back the default stream of every device the sources came
            // from, not only whichever one happens to be current.
            for (int d = 0; d < 64; d++) {
                if (!((src_devices >> d) & 1)) continue;
                backend_default_stream_wait_side(
                    BackendStreamKind::Copy, copy_device, d);
            }
        }
        for (uint j=0; j<allocations.size(); j++) {
            auto& allocation = allocations[j];
            if (allocation.allocator != (Allocator*)&cuda_dual_allocator)
                continue;
            auto host_ptr = cuda_dual_allocator.get_dual_allocation(
                allocation.allocation).host_ptr;
            // device to host
            Device source{accelerator_backend_id(), cuda_dual_device_allocator.device()};
            backend_copy_async(host_ptr, {}, allocation.ptr, source, allocation.size,
                copy_stream);
            allocation.ptr = host_ptr;
            arrays[j].ptr = host_ptr;
        }
        // appended last: the loop above must not mistake them for staging
        for (auto& p : pinned)
            allocations.emplace_back(move(p));
        fetch_tasks.push_back({move(func), move(allocations), move(arrays)});
        if (current_device() != copy_device) set_current_device(copy_device);
        backend_ops(copy_stream.device.backend).host_callback(copy_stream, &to_fetch, nullptr);
        if (entry_device >= 0 && entry_device != current_device())
            set_current_device(entry_device);
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
