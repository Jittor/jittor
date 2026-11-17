#pragma once
#include "core/op.h"

namespace jittor {

// An indexed write whose output names the root storage. Returning the root
// gives state capture an ordinary data dependency for the storage mutation.
struct IndexedStorageWriteOp : Op {
    static constexpr bool accepts_storage_strides = true;
    static constexpr bool mutates_storage_inputs = true;
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var *x, *values, *out, *storage_root;
    vector<Var*> indices;
    NanoVector axes;
    bool ordered = false;

    // @pybind(None)
    IndexedStorageWriteOp(Var* storage_root, Var* x, Var* values,
                         vector<Var*>&& indices, NanoVector axes);
    const char* name() const override { return "indexed_storage_write"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

} // namespace jittor
