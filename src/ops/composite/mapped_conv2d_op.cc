// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/composite/mapped_conv2d_op.h"
#include "core/var.h"
#include "ops/op_register.h"
#include "runtime/device.h"
#include "runtime/dispatch_context.h"

namespace jittor {

#ifndef JIT

DECLARE_FLAG(bool, no_grad);

static auto make_mapped_conv2d_backward =
    op_constructor<vector<VarPtr>, Var*, Var*, Var*, int, int, int, int, int, int, int, bool, int>(
        "mapped_conv2d_backward");

MappedConv2dOp::MappedConv2dOp(
    vector<Var*>&& inputs,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dil_h, int dil_w,
    int groups, int cube_math_type)
    : inputs(inputs),
      x(nullptr), weight(nullptr), bias(nullptr), y(nullptr),
      stride_h(stride_h), stride_w(stride_w),
      pad_h(pad_h), pad_w(pad_w),
      dil_h(dil_h), dil_w(dil_w),
      groups(groups), cube_math_type(cube_math_type) {
    USER_CHECK(this->inputs.size() == 2 || this->inputs.size() == 3)
        << "mapped_conv2d expects 2 or 3 inputs [x, weight(, bias)], got" << this->inputs.size();
    x = this->inputs[0];
    weight = this->inputs[1];
    bias = this->inputs.size() == 3 ? this->inputs[2] : nullptr;
    USER_CHECK(x->dtype().is_float() && weight->dtype() == x->dtype())
        << "mapped_conv2d requires matching floating-point x and weight, got"
        << x->dtype() << "and" << weight->dtype();
    if (bias) {
        USER_CHECK(bias->dtype() == x->dtype())
            << "mapped_conv2d bias dtype must match x, got" << bias->dtype() << "vs" << x->dtype();
    }
    USER_CHECK(stride_h > 0 && stride_w > 0 && dil_h > 0 && dil_w > 0 && pad_h >= 0 && pad_w >= 0 && groups > 0)
        << "mapped_conv2d invalid convolution geometry";
    USER_CHECK(runtime_use_cuda()) << "mapped_conv2d requires an accelerator backend";
    const auto target = query_dispatch_context(this->inputs);
    USER_CHECK(target.backend == "acl" || target.backend == "acl_legacy")
        << "mapped_conv2d requires the ACL backend, got" << target.backend;
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    set_flag(OpFlags::_grads);
    if (!no_grad) {
        x->set_flag(VarFlags::_needed_by_backward);
        weight->set_flag(VarFlags::_needed_by_backward);
    }
    y = create_output(nullptr, x->dtype());
}

void MappedConv2dOp::infer_shape() {
    USER_CHECK(x->shape.size() == 4 && weight->shape.size() == 4)
        << "mapped_conv2d requires 4-D x and weight, got" << x->shape << "and" << weight->shape;
    USER_CHECK(x->shape[1] == weight->shape[1] * groups && weight->shape[0] % groups == 0)
        << "mapped_conv2d channel/group mismatch:" << x->shape << "vs" << weight->shape << "groups=" << groups;
    if (bias) {
        USER_CHECK(bias->shape.size() == 1 && bias->shape[0] == weight->shape[0])
            << "mapped_conv2d bias shape mismatch:" << bias->shape << "vs out_channels=" << weight->shape[0];
    }
    const int64 oh = (x->shape[2] + 2 * pad_h - dil_h * (weight->shape[2] - 1) - 1) / stride_h + 1;
    const int64 ow = (x->shape[3] + 2 * pad_w - dil_w * (weight->shape[3] - 1) - 1) / stride_w + 1;
    USER_CHECK(oh > 0 && ow > 0) << "mapped_conv2d non-positive output spatial size:" << oh << ow;
    y->set_shape({x->shape[0], weight->shape[0], oh, ow});
}

void MappedConv2dOp::grads(Var** douts, VarPtr* dins) {
    if (!douts[0]) return;
    auto outs = make_mapped_conv2d_backward(
        douts[0], x, weight,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
        groups, bias != nullptr, cube_math_type);
    dins[0] = move(outs[0]);
    dins[1] = move(outs[1]);
    if (bias && outs.size() == 3)
        dins[2] = move(outs[2]);
}

void MappedConv2dOp::jit_prepare(JK& jk) {
    jk << "«T:" << x->dtype() << "«B:" << (bias ? 1 : 0) << "«C:" << cube_math_type;
}

#else
void MappedConv2dOp::jit_run() {
    USER_ERROR << "mapped_conv2d is only available through a mapped backend";
}
#endif

} // jittor
