// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/composite/mapped_grouped_matmul_op.h"
#include "core/var.h"
#include "runtime/device.h"
#include "runtime/dispatch_context.h"

namespace jittor {

#ifndef JIT

bool MappedGroupedMatmulOp::accepts_strided_input(const Var* v) {
    if (!v || v->num <= 0) return false;
    if (v->is_contiguous()) return true;
    // Accept 3-D expert weights [E, K, N] that are a last-2-axis transpose view
    // of contiguous [E, N, K] so MoE layers never materialize a full 3-D copy.
    if (v->shape.size() != 3 || !v->dtype().is_float()) return false;
    const int64 e = v->shape[0];
    const int64 dk = v->shape[1];
    const int64 dn = v->shape[2];
    const int64 se = v->storage_stride(0);
    const int64 sk = v->storage_stride(1);
    const int64 sn = v->storage_stride(2);
    if (se < 0 || sk < 0 || sn < 0) return false;
    const bool col_major = (sk == 1) && (dn <= 1 || sn == dk);
    const bool row_major = (sn == 1) && (dk <= 1 || sk == dn);
    if (!col_major && !row_major) return false;
    if (e > 1 && se != dk * dn) return false;
    return true;
}

MappedGroupedMatmulOp::MappedGroupedMatmulOp(Var* x, Var* weight, Var* offs)
    : x(x), weight(weight), offs(offs) {
    USER_CHECK(x->dtype().is_float() && weight->dtype() == x->dtype())
        << "mapped_grouped_matmul requires matching floating-point x and weight, got"
        << x->dtype() << "and" << weight->dtype();
    USER_CHECK(offs->dtype() == ns_int64)
        << "mapped_grouped_matmul requires int64 offs, got" << offs->dtype();
    USER_CHECK(runtime_use_cuda())
        << "mapped_grouped_matmul requires an accelerator backend";
    const auto target = query_dispatch_context({x, weight, offs});
    USER_CHECK(target.backend == "acl" || target.backend == "acl_legacy")
        << "mapped_grouped_matmul requires the ACL backend, got" << target.backend;
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    y = create_output(nullptr, x->dtype());
}

void MappedGroupedMatmulOp::infer_shape() {
    USER_CHECK(x->shape.size() == 2)
        << "mapped_grouped_matmul requires 2-D x [S, K], got" << x->shape;
    USER_CHECK(weight->shape.size() == 3)
        << "mapped_grouped_matmul requires 3-D weight [E, K, N], got" << weight->shape;
    USER_CHECK(offs->shape.size() == 1)
        << "mapped_grouped_matmul requires 1-D offs [E], got" << offs->shape;
    USER_CHECK(weight->shape[0] == offs->shape[0])
        << "mapped_grouped_matmul expert count mismatch:" << weight->shape[0] << "vs" << offs->shape[0];
    USER_CHECK(x->shape[1] == weight->shape[1])
        << "mapped_grouped_matmul K dimension mismatch:" << x->shape[1] << "vs" << weight->shape[1];
    y->set_shape({x->shape[0], weight->shape[2]});
}

VarPtr MappedGroupedMatmulOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    USER_ERROR << "mapped_grouped_matmul is inference-only and has no backward";
    return nullptr;
}

void MappedGroupedMatmulOp::jit_prepare(JK& jk) {
    jk << "«T:" << x->dtype();
}

#else
void MappedGroupedMatmulOp::jit_run() {
    USER_ERROR << "mapped_grouped_matmul is only available through a mapped backend";
}
#endif

} // jittor
