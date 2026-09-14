// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"

namespace jittor {

/** One matrix product, executed by the backend's own GEMM as a single node.

    The portable definition of a product is broadcast * multiply + reduce.
    Every accelerator has a GEMM that does it in one library call instead, and
    each backend family already models that call as a graph node of its own --
    `cublas_matmul` on CUDA, `mkl_matmul` on CPU. Those live under `backends/`
    because their sources are compiled through `compile_custom_ops`, which
    scans their headers for an operator. The ACL backend compiles its sources
    into the core instead, and the core build only scans `src/ops` for op
    headers, so the class lives here; the backend supplies the execution by
    mapping this op's name in its own table (`acl_ops`, in
    `backends/acl/src/acl_op_exec.cc`). `backend_mask` keeps the op off the
    CPU, and the JIT body refuses rather than producing numbers on an
    accelerator that has not mapped it -- the `fused_sgd` arrangement.

    Batched and plain products are the same node. Which library entry point
    runs is a function of the operand rank -- exactly the choice the frontend
    already makes between its `matmul` and `batched_matmul` kernel slots -- and
    one node means one gradient rather than two that have to agree.
 */
struct MappedMatmulOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var* a, * b, * c;
    // c = op(a) @ op(b), where op transposes the last two axes of an operand.
    // At most one of the two: the descriptors these backends build present a
    // single operand transposed, and the gradient below never asks for both.
    bool trans_a, trans_b;
    // Whether the backend may accumulate this float32 product at reduced
    // precision -- ACL's cubeMathType ALLOW_FP32_DOWN_PRECISION (HF32), which
    // is what `jt.acl_allow_hf32` asks for and what torch_npu does by default.
    // Per node, not per process: a caller may flip the flag between two
    // products, and a gradient has to use the arithmetic of the forward it
    // differentiates rather than whatever the flag says when it is built.
    bool allow_reduced_precision;

    MappedMatmulOp(Var* a, Var* b, bool trans_a, bool trans_b, bool allow_reduced_precision);

    const char* name() const override { return "mapped_matmul"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
