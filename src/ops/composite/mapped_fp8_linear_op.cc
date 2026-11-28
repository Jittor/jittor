// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/composite/mapped_fp8_linear_op.h"
#include "core/var.h"
#include "runtime/dispatch_context.h"
#include "runtime/device.h"

namespace jittor {

#ifndef JIT

DECLARE_FLAG(bool, no_grad);

MappedFp8LinearOp::MappedFp8LinearOp(
    Var* x, Var* weight, Var* weight_scale,
    int64 mode, int64 block_size, NanoString out_dtype)
    : x(x), weight(weight), weight_scale(weight_scale),
      mode(mode), block_size(block_size), out_dtype(out_dtype) {
    USER_CHECK(no_grad)
        << "mapped_fp8_linear requires no_grad; use the differentiable path for training";
    USER_CHECK(mode == 0 || mode == 1)
        << "mapped_fp8_linear mode must be 0 (block128) or 1 (mxfp8), got" << mode;
    USER_CHECK(block_size > 0)
        << "mapped_fp8_linear block_size must be positive, got" << block_size;
    USER_CHECK(x->dtype() == ns_bfloat16 || x->dtype() == ns_float16)
        << "mapped_fp8_linear expects bfloat16 or float16 input, got" << x->dtype();
    USER_CHECK(weight->dtype() == ns_float8_e4m3fn || weight->dtype() == ns_float8_e5m2)
        << "mapped_fp8_linear expects float8_e4m3fn or float8_e5m2 weight, got" << weight->dtype();
    if (mode == 0) {
        USER_CHECK(weight_scale->dtype() == ns_float32)
            << "mapped_fp8_linear block mode expects float32 weight_scale, got" << weight_scale->dtype();
    } else {
        USER_CHECK(weight_scale->dtype() == ns_float8_e8m0fnu)
            << "mapped_fp8_linear mxfp8 mode expects float8_e8m0fnu weight_scale, got" << weight_scale->dtype();
    }
    USER_CHECK(out_dtype == ns_bfloat16 || out_dtype == ns_float16 || out_dtype == ns_float32)
        << "mapped_fp8_linear expects bfloat16/float16/float32 out_dtype, got" << out_dtype;
    USER_CHECK(runtime_use_cuda()) << "mapped_fp8_linear requires the ACL device backend";
    const auto target = query_dispatch_context({x, weight, weight_scale});
    USER_CHECK(target.backend == "acl" || target.backend == "acl_legacy")
        << "mapped_fp8_linear requires the ACL device backend, but got" << target.backend;
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    y = create_output(nullptr, out_dtype);
}

VarPtr MappedFp8LinearOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    USER_ERROR << "mapped_fp8_linear is inference-only and has no backward";
    return nullptr;
}

void MappedFp8LinearOp::infer_shape() {
    const auto rank = x->shape.size();
    USER_CHECK(rank >= 2) << "mapped_fp8_linear input must have rank >= 2, got" << x->shape;
    USER_CHECK(weight->shape.size() == 2)
        << "mapped_fp8_linear weight must be 2-D [N, K], got" << weight->shape;
    int64 m_dim = 1;
    for (int axis = 0; axis + 1 < rank; ++axis) {
        USER_CHECK(x->shape[axis] > 0)
            << "mapped_fp8_linear requires positive input dimensions, got" << x->shape;
        m_dim *= x->shape[axis];
    }
    const int64 k_dim = x->shape[rank - 1];
    const int64 n_dim = weight->shape[0];
    USER_CHECK(k_dim == weight->shape[1] && k_dim > 0 && n_dim > 0)
        << "mapped_fp8_linear dimension mismatch: x=" << x->shape << "weight=" << weight->shape;

    NanoVector out_shape;
    for (int axis = 0; axis + 1 < rank; ++axis)
        out_shape.push_back_check_overflow(x->shape[axis]);
    out_shape.push_back_check_overflow(n_dim);
    y->set_shape(out_shape);

    NanoVector qx_shape;
    qx_shape.push_back_check_overflow(m_dim);
    qx_shape.push_back_check_overflow(k_dim);
    if (!qx_scratch || !(qx_scratch->shape == qx_shape) || qx_scratch->dtype() != weight->dtype())
        qx_scratch = VarPtr(qx_shape, weight->dtype());

    NanoVector sx_shape;
    NanoString sx_dtype = (mode == 0) ? ns_float32 : ns_float8_e8m0fnu;
    if (mode == 0) {
        USER_CHECK(k_dim % block_size == 0 && n_dim % block_size == 0)
            << "mapped_fp8_linear block mode requires K and N divisible by" << block_size;
        USER_CHECK(weight_scale->shape.size() == 2
                   && weight_scale->shape[0] == n_dim / block_size
                   && weight_scale->shape[1] == k_dim / block_size)
            << "mapped_fp8_linear block weight_scale shape mismatch:" << weight_scale->shape;
        sx_shape.push_back_check_overflow(m_dim);
        sx_shape.push_back_check_overflow(k_dim / block_size);
    } else {
        const int64 pair_block = 2 * block_size;
        USER_CHECK(k_dim % pair_block == 0)
            << "mapped_fp8_linear mxfp8 mode requires K divisible by" << pair_block;
        USER_CHECK(weight_scale->shape.size() == 3
                   && weight_scale->shape[0] == n_dim
                   && weight_scale->shape[1] == k_dim / pair_block
                   && weight_scale->shape[2] == 2)
            << "mapped_fp8_linear mxfp8 weight_scale shape mismatch:" << weight_scale->shape;
        sx_shape.push_back_check_overflow(m_dim);
        sx_shape.push_back_check_overflow(k_dim / pair_block);
        sx_shape.push_back_check_overflow(2);
    }
    if (!sx_scratch || !(sx_scratch->shape == sx_shape) || sx_scratch->dtype() != sx_dtype)
        sx_scratch = VarPtr(sx_shape, sx_dtype);
}

void MappedFp8LinearOp::jit_prepare(JK& jk) {
    jk << "«M:" << mode << "«Tx:" << x->dtype()
       << "«Tw:" << weight->dtype() << "«Ty:" << out_dtype;
}

#else
void MappedFp8LinearOp::jit_run() {
    USER_ERROR << "mapped_fp8_linear is only available through a mapped backend";
}
#endif

} // jittor
