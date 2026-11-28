// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/composite/mapped_flash_attention_op.h"
#include "core/var.h"
#include "runtime/device.h"
#include "runtime/dispatch_context.h"

namespace jittor {

#ifndef JIT

DECLARE_FLAG(bool, no_grad);

MappedFlashAttentionOp::MappedFlashAttentionOp(
    Var* q, Var* k, Var* v, Var* atten_mask,
    double scale, int64 sparse_mode, bool has_mask, bool mask_is_realshift)
    : q(q), k(k), v(v), atten_mask(atten_mask),
      scale(scale), sparse_mode(sparse_mode),
      has_mask(has_mask), mask_is_realshift(mask_is_realshift) {
    USER_CHECK(no_grad)
        << "mapped_flash_attention requires no_grad; use FlashAttentionACL for training";
    USER_CHECK(q->dtype().is_float() && k->dtype() == q->dtype() && v->dtype() == q->dtype())
        << "mapped_flash_attention requires matching float q/k/v dtypes";
    USER_CHECK(runtime_use_cuda())
        << "mapped_flash_attention requires an accelerator backend";
    const auto target = query_dispatch_context({q, k, v});
    USER_CHECK(target.backend == "acl" || target.backend == "acl_legacy")
        << "mapped_flash_attention requires the ACL backend, got" << target.backend;
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    out = create_output(nullptr, q->dtype());
}

void MappedFlashAttentionOp::infer_shape() {
    USER_CHECK(q->shape.size() == 4 && k->shape.size() == 4 && v->shape.size() == 4)
        << "mapped_flash_attention requires 4-D BNSD q/k/v";
    out->set_shape(q->shape);
    NanoVector stat_shape{q->shape[0], q->shape[1], q->shape[2], 8};
    if (!softmax_max_scratch || !(softmax_max_scratch->shape == stat_shape))
        softmax_max_scratch = VarPtr(stat_shape, ns_float32);
    if (!softmax_sum_scratch || !(softmax_sum_scratch->shape == stat_shape))
        softmax_sum_scratch = VarPtr(stat_shape, ns_float32);
}

VarPtr MappedFlashAttentionOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    USER_ERROR << "mapped_flash_attention is inference-only and has no backward";
    return nullptr;
}

void MappedFlashAttentionOp::jit_prepare(JK& jk) {
    jk << "«T:" << q->dtype() << "«S:" << sparse_mode
       << "«M:" << (has_mask ? (mask_is_realshift ? 2 : 1) : 0);
}

#else
void MappedFlashAttentionOp::jit_run() {
    USER_ERROR << "mapped_flash_attention is only available through a mapped backend";
}
#endif

} // jittor
