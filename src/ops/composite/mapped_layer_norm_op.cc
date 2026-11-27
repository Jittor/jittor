// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include "ops/composite/mapped_layer_norm_op.h"
#include "core/var.h"
#include "ops/op_register.h"
#include "runtime/device.h"
#include "runtime/dispatch_context.h"

namespace jittor {

#ifndef JIT

DECLARE_FLAG(bool, no_grad);

static auto make_mapped_layer_norm_backward =
    op_constructor<vector<VarPtr>, Var*, Var*, Var*, Var*, Var*, Var*, int, double>(
        "mapped_layer_norm_backward");

MappedLayerNormOp::MappedLayerNormOp(
    Var* x, Var* weight, Var* bias,
    int normalized_ndim, double eps)
    : x(x), weight(weight), bias(bias),
      y(nullptr), mean(nullptr), rstd(nullptr),
      normalized_ndim(normalized_ndim), eps(eps) {
    USER_CHECK(x->dtype().is_float() && weight->dtype() == x->dtype() && bias->dtype() == x->dtype())
        << "mapped_layer_norm requires matching floating-point x, weight, bias";
    USER_CHECK(normalized_ndim >= 1)
        << "mapped_layer_norm normalized_ndim must be >= 1, got" << normalized_ndim;
    USER_CHECK(std::isfinite(eps) && eps > 0.0)
        << "mapped_layer_norm eps must be finite and positive, got" << eps;
    USER_CHECK(runtime_use_cuda()) << "mapped_layer_norm requires an accelerator backend";
    const auto target = query_dispatch_context({x, weight, bias});
    USER_CHECK(target.backend == "acl" || target.backend == "acl_legacy")
        << "mapped_layer_norm requires the ACL backend, got" << target.backend;
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    set_flag(OpFlags::_grads);
    y = create_output(nullptr, x->dtype());
    if (!no_grad) {
        mean = create_output(nullptr, x->dtype());
        rstd = create_output(nullptr, x->dtype());
        x->set_flag(VarFlags::_needed_by_backward);
        weight->set_flag(VarFlags::_needed_by_backward);
        bias->set_flag(VarFlags::_needed_by_backward);
        mean->set_flag(VarFlags::_needed_by_backward);
        rstd->set_flag(VarFlags::_needed_by_backward);
    }
}

void MappedLayerNormOp::infer_shape() {
    const int rank = static_cast<int>(x->shape.size());
    USER_CHECK(normalized_ndim >= 1 && normalized_ndim <= rank)
        << "mapped_layer_norm normalized_ndim out of range:" << normalized_ndim << "for rank" << rank;
    USER_CHECK(static_cast<int>(weight->shape.size()) == normalized_ndim
               && static_cast<int>(bias->shape.size()) == normalized_ndim)
        << "mapped_layer_norm weight/bias rank mismatch";
    for (int i = 0; i < normalized_ndim; ++i) {
        const int64 expected = x->shape[rank - normalized_ndim + i];
        USER_CHECK(weight->shape[i] == expected && bias->shape[i] == expected)
            << "mapped_layer_norm weight/bias shape mismatch with trailing input shape";
    }
    y->set_shape(x->shape);
    NanoVector reduced;
    for (int i = 0; i < rank - normalized_ndim; ++i)
        reduced.push_back_check_overflow(x->shape[i]);
    for (int i = 0; i < normalized_ndim; ++i)
        reduced.push_back_check_overflow(1);
    if (mean) {
        mean->set_shape(reduced);
        rstd->set_shape(reduced);
    } else {
        if (!mean_scratch || !(mean_scratch->shape == reduced) || mean_scratch->dtype() != x->dtype())
            mean_scratch = VarPtr(reduced, x->dtype());
        if (!rstd_scratch || !(rstd_scratch->shape == reduced) || rstd_scratch->dtype() != x->dtype())
            rstd_scratch = VarPtr(reduced, x->dtype());
    }
}

void MappedLayerNormOp::grads(Var** douts, VarPtr* dins) {
    if (!douts[0]) return;
    USER_CHECK(mean && rstd)
        << "mapped_layer_norm backward requires forward to run with gradients enabled";
    auto outs = make_mapped_layer_norm_backward(
        douts[0], x, mean, rstd, weight, bias, normalized_ndim, eps);
    dins[0] = move(outs[0]);
    dins[1] = move(outs[1]);
    dins[2] = move(outs[2]);
}

void MappedLayerNormOp::jit_prepare(JK& jk) {
    jk << "«T:" << x->dtype() << "«ND:" << normalized_ndim;
}

#else
void MappedLayerNormOp::jit_run() {
    USER_ERROR << "mapped_layer_norm is only available through a mapped backend";
}
#endif

} // jittor
