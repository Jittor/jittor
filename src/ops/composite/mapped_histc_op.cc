// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "ops/composite/mapped_histc_op.h"
#include "core/var.h"
#include "runtime/device.h"
#include "runtime/dispatch_context.h"

namespace jittor {

#ifndef JIT

MappedHistcOp::MappedHistcOp(Var* x, int64 bins, double min_val, double max_val)
    : x(x), bins(bins), min_val(min_val), max_val(max_val) {
    USER_CHECK(bins > 0) << "mapped_histc requires bins > 0, got" << bins;
    USER_CHECK(min_val <= max_val) << "mapped_histc requires min <= max";
    USER_CHECK(runtime_use_cuda()) << "mapped_histc requires an accelerator backend";
    const auto target = query_dispatch_context({x});
    USER_CHECK(target.backend == "acl" || target.backend == "acl_legacy")
        << "mapped_histc requires the ACL backend, got" << target.backend;
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_manual_set_vnbb);
    y = create_output(nullptr, x->dtype());
}

void MappedHistcOp::infer_shape() {
    y->set_shape({bins});
}

VarPtr MappedHistcOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return nullptr;
}

void MappedHistcOp::jit_prepare(JK& jk) {
    jk << "«T:" << x->dtype();
}

#else
void MappedHistcOp::jit_run() {
    USER_ERROR << "mapped_histc is only available through a mapped backend";
}
#endif

} // jittor
