#pragma once
#include "core/op.h"

namespace jittor {
struct RocmArgReduceOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var *x, *offsets, *indices, *values;
    NanoString operation;
    bool keepdims;
    // @attrs(multiple_outputs)
    RocmArgReduceOp(Var* x, Var* offsets, NanoString operation, bool keepdims);
    const char* name() const override { return "rocm_arg_reduce"; }
    void infer_shape() override;
    void run() override;
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
};
} // namespace jittor
