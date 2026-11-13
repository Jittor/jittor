#pragma once
#include "core/op.h"

namespace jittor {

// FP32 NCHW inference only. Differentiable convolution uses the graph of
// reindex and GEMM operators in kernels/convolution.py instead.
struct HipblasConvInferenceOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var* x;
    Var* weight;
    Var* bias;
    Var* y;
    NanoVector stride, padding, dilation;
    HipblasConvInferenceOp(Var* x, Var* weight, vector<Var*> biases, NanoVector stride, NanoVector padding, NanoVector dilation);
    const char* name() const override { return "hipblas_conv_inference"; }
    void infer_shape() override;
    void run() override;
};

} // namespace jittor
