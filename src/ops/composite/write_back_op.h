#pragma once
#include "core/op.h"

namespace jittor {

// Writes each of `values` into the storage of the matching `targets` entry,
// all in one launch; its outputs share the targets' storage, so the write is
// what they read. What a captured step replays in place of one copy per
// piece of state it updates (see `jittor._runtime.step_capture`).
struct WriteBackOp : Op {
    static constexpr bool mutates_storage_inputs = true;
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    vector<Var*> targets, values;
    vector<Var*> written;
    // Whether `order_after_readers` has run: once, at construction.
    bool ordered = false;

    // @attrs(multiple_outputs)
    WriteBackOp(vector<Var*>&& targets, vector<Var*>&& values);

    const char* name() const override { return "write_back"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
