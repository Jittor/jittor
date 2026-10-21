// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in LICENSE.txt.
#include "core/rematerialization.h"
#include "core/executor.h"
#include "runtime/async_exec.h"
#include "runtime/executor_entry.h"

namespace jittor {

DECLARE_FLAG(int, keep_graph);

namespace {

struct RetainTopology {
    int previous = keep_graph;
    RetainTopology() { keep_graph = 2; }
    ~RetainTopology() { keep_graph = previous; }
};

// Called only immediately after our successful, device-synchronous batch.
// Neither a pointer nor _kept alone proves that a producer wrote its bytes.
VarPtr ready_alias(VarHolder* source) {
    Var* original = source->var;
    USER_CHECK(original->num >= 0 && !original->flag(VarFlags::_execution_failed)
            && original->allocator && (original->mem_ptr || !original->size))
        << "rematerialization needs successfully materialized output storage";

    VarPtr ready(original->shape, original->dtype());
    ready->storage_strides = original->storage_strides;
    ready->device_id = original->device_id;
    ready->placement = original->placement;
    ready->set_flag(VarFlags::_host_resident, original->flag(VarFlags::_host_resident));
    ready->set_stop_grad();
    ready->share_with(original);
    USER_CHECK(ready->alloc(original->allocator)
            && ready->mem_ptr == original->mem_ptr
            && (!original->size || ready->shares_allocation_with(original)))
        << "allocator cannot share rematerialized output storage";

    // Share mutation detection, not trainability or the source graph's owner.
    ready->autograd_alias = std::make_shared<AutogradAliasState>();
    ready->autograd_alias->version = source->ensure_autograd_alias()->version;
    ready->autograd_alias->creation_version = *ready->autograd_alias->version;
    // Complete only this new inputless leaf. Original nodes remain untouched.
    ready->finish_pending_liveness();
    return ready;
}

} // namespace

vector<VarHolder*> materialize_remat_outputs(const vector<VarHolder*>& outputs) {
    // Match ordinary executor entry ordering; do not hold its entry lock while
    // waiting for a previous async submission to release that lock.
    async_exec_wait();
    ExecutorEntryScope entry;
    RetainTopology retain;
    // One strong-root batch handles shared/duplicate outputs without replaying
    // their providers once per output. Device completion is our ready receipt.
    runtime_executor().run_sync(convert(outputs), true, false, false);

    vector<VarPtr> ready;
    ready.reserve(outputs.size());
    for (auto* output : outputs) ready.emplace_back(ready_alias(output));
    return make_vh_vector(move(ready));
}

} // namespace jittor
