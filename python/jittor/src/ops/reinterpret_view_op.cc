// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "ops/op_register.h"
#include "ops/reinterpret_view_op.h"

namespace jittor {

static auto make_reinterpret_view = op_constructor<VarPtr, Var*, NanoVector, NanoString>("reinterpret_view");

ReinterpretViewOp::ReinterpretViewOp(Var* x, NanoVector shape, NanoString dtype)
    : x(x), shape(shape), dtype(dtype) {
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);
    CHECK(dtype.is_dtype()) << "reinterpret_view expects dtype, got" << dtype;
    y = create_output(nullptr, dtype);
}

VarPtr ReinterpretViewOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    if (!((x->dtype() == ns_complex64 && dtype == ns_float32) ||
          (x->dtype() == ns_float32 && dtype == ns_complex64)))
        return nullptr;
    return make_reinterpret_view(dout, x->shape, x->dtype());
}

void ReinterpretViewOp::infer_shape() {
    int64 known_items = 1;
    int infer_dim = -1;
    NanoVector yshape = shape;
    for (uint i=0; i<shape.size(); i++) {
        if (shape[i] < 0) {
            CHECK(infer_dim < 0) << "reinterpret_view allows at most one -1 dimension";
            infer_dim = i;
        } else {
            known_items *= shape[i];
        }
    }

    CHECK(x->num >= 0) << "reinterpret_view requires known input size";
    int64 x_bytes = x->num * x->dsize();
    int64 ydsize = dtype.dsize();
    CHECK(ydsize > 0);
    if (infer_dim >= 0) {
        CHECK(known_items > 0 && x_bytes % (known_items * ydsize) == 0)
            << "reinterpret_view cannot infer shape" << shape << "for byte size" << x_bytes;
        yshape.set_data(infer_dim, x_bytes / (known_items * ydsize));
    }
    y->set_shape(yshape);
    CHECKop(y->size,==,x_bytes)
        << "reinterpret_view byte size mismatch, input" << x->shape << x->dtype()
        << "target" << yshape << dtype;
    if (x->dtype() == ns_complex64) {
        CHECK(dtype == ns_float32 && yshape.size() && yshape[yshape.size()-1] == 2)
            << "complex64 -> float32 reinterpret_view requires target shape [..., 2]";
    } else if (dtype == ns_complex64) {
        CHECK(x->dtype() == ns_float32 && x->shape.size() && x->shape[x->shape.size()-1] == 2)
            << "float32 -> complex64 reinterpret_view requires input shape [..., 2]";
    }
    y->share_with(x);
}

} // jittor
