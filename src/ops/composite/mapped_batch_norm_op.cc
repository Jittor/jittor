// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include "ops/composite/mapped_batch_norm_op.h"
#include "core/var.h"
#include "ops/op_register.h"
#include "runtime/device.h"
#include "runtime/dispatch_context.h"

namespace jittor {

#ifndef JIT

DECLARE_FLAG(bool, no_grad);

static auto make_mapped_batch_norm_backward =
    op_constructor<vector<VarPtr>, Var*, Var*, Var*, Var*, Var*, Var*, Var*, double, bool>(
        "mapped_batch_norm_backward");

MappedBatchNormOp::MappedBatchNormOp(
    Var* x, Var* weight, Var* bias,
    Var* running_mean, Var* running_var,
    double momentum, double eps,
    bool is_train, bool return_running_stats)
    : x(x), weight(weight), bias(bias),
      running_mean(running_mean), running_var(running_var),
      y(nullptr), new_running_mean(nullptr), new_running_var(nullptr),
      save_mean(nullptr), save_invstd(nullptr),
      momentum(momentum), eps(eps),
      is_train(is_train), return_running_stats(return_running_stats) {
    USER_CHECK(x->dtype().is_float() && weight->dtype() == x->dtype() && bias->dtype() == x->dtype())
        << "mapped_batch_norm requires matching floating-point x, weight, bias";
    USER_CHECK(running_mean->dtype() == x->dtype() || running_mean->dtype() == ns_float32)
        << "mapped_batch_norm unsupported running_mean dtype" << running_mean->dtype();
    USER_CHECK(running_var->dtype() == x->dtype() || running_var->dtype() == ns_float32)
        << "mapped_batch_norm unsupported running_var dtype" << running_var->dtype();
    USER_CHECK(std::isfinite(eps) && eps > 0.0)
        << "mapped_batch_norm eps must be finite and positive, got" << eps;
    USER_CHECK(std::isfinite(momentum) && momentum >= 0.0 && momentum <= 1.0)
        << "mapped_batch_norm momentum must be in [0, 1], got" << momentum;
    USER_CHECK(runtime_use_cuda()) << "mapped_batch_norm requires an accelerator backend";
    const auto target = query_dispatch_context({x, weight, bias, running_mean, running_var});
    USER_CHECK(target.backend == "acl" || target.backend == "acl_legacy")
        << "mapped_batch_norm requires the ACL backend, got" << target.backend;
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    set_flag(OpFlags::_grads);
    y = create_output(nullptr, x->dtype());
    if (return_running_stats) {
        new_running_mean = create_output(nullptr, running_mean->dtype());
        new_running_var = create_output(nullptr, running_var->dtype());
        new_running_mean->set_stop_grad();
        new_running_var->set_stop_grad();
    }
    if (!no_grad) {
        save_mean = create_output(nullptr, x->dtype());
        save_invstd = create_output(nullptr, x->dtype());
        x->set_flag(VarFlags::_needed_by_backward);
        weight->set_flag(VarFlags::_needed_by_backward);
        save_mean->set_flag(VarFlags::_needed_by_backward);
        save_invstd->set_flag(VarFlags::_needed_by_backward);
        if (!is_train) {
            running_mean->set_flag(VarFlags::_needed_by_backward);
            running_var->set_flag(VarFlags::_needed_by_backward);
        }
    }
}

void MappedBatchNormOp::infer_shape() {
    USER_CHECK(x->shape.size() == 4)
        << "mapped_batch_norm requires 4-D NCHW input, got" << x->shape;
    const int64 channels = x->shape[1];
    for (int d = 0; d < 4; ++d)
        USER_CHECK(x->shape[d] > 0) << "mapped_batch_norm requires positive dimensions, got" << x->shape;
    USER_CHECK(weight->shape.size() == 1 && weight->shape[0] == channels
               && bias->shape.size() == 1 && bias->shape[0] == channels
               && running_mean->shape.size() == 1 && running_mean->shape[0] == channels
               && running_var->shape.size() == 1 && running_var->shape[0] == channels)
        << "mapped_batch_norm parameter/buffer channel count mismatch";
    y->set_shape(x->shape);
    NanoVector stat_shape{channels};
    if (new_running_mean) new_running_mean->set_shape(stat_shape);
    if (new_running_var) new_running_var->set_shape(stat_shape);
    if (save_mean) {
        save_mean->set_shape(stat_shape);
        save_invstd->set_shape(stat_shape);
    } else {
        if (!save_mean_scratch || !(save_mean_scratch->shape == stat_shape)
                || save_mean_scratch->dtype() != x->dtype())
            save_mean_scratch = VarPtr(stat_shape, x->dtype());
        if (!save_invstd_scratch || !(save_invstd_scratch->shape == stat_shape)
                || save_invstd_scratch->dtype() != x->dtype())
            save_invstd_scratch = VarPtr(stat_shape, x->dtype());
    }
}

void MappedBatchNormOp::grads(Var** douts, VarPtr* dins) {
    if (!douts[0]) return;
    USER_CHECK(save_mean && save_invstd)
        << "mapped_batch_norm backward requires forward to run with gradients enabled";
    // In training mode aclnnBatchNormBackward uses save_mean/save_invstd and ignores
    // runningMean/runningVar; passing save_mean/save_invstd keeps old running stats freeable.
    Var* rm_arg = is_train ? save_mean : running_mean;
    Var* rv_arg = is_train ? save_invstd : running_var;
    auto outs = make_mapped_batch_norm_backward(
        douts[0], x, weight, rm_arg, rv_arg, save_mean, save_invstd, eps, is_train);
    dins[0] = move(outs[0]);
    dins[1] = move(outs[1]);
    dins[2] = move(outs[2]);
}

void MappedBatchNormOp::jit_prepare(JK& jk) {
    jk << "«T:" << x->dtype() << "«TR:" << (is_train ? 1 : 0)
       << "«RS:" << (return_running_stats ? 1 : 0);
}

#else
void MappedBatchNormOp::jit_run() {
    USER_ERROR << "mapped_batch_norm is only available through a mapped backend";
}
#endif

} // jittor
