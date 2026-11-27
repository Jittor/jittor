#pragma once

#include "core/op.h"

namespace jittor {

struct MiopenConv2dOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var *a, *b, *result;
    NanoVector output_shape, stride, padding, dilation;
    int direction;

    MiopenConv2dOp(Var* a, Var* b, NanoVector output_shape, NanoVector stride,
                   NanoVector padding, NanoVector dilation, int direction);
    const char* name() const override { return "miopen_conv2d"; }
    void infer_shape() override;
    void run() override;
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
};

} // namespace jittor
