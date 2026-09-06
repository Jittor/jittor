#pragma once

#include "op.h"

namespace jittor {

struct RocprimCumsumOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var* x;
    Var* y;
    bool reverse;

    RocprimCumsumOp(Var* x, bool reverse=false);
    const char* name() const override { return "rocprim_cumsum"; }
    void infer_shape() override;
    void run() override;
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
};

} // namespace jittor
