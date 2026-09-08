// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/var.h"
#include "ops/composite/array_op.h"
#include "ops/op_register.h"
#include "ops/composite/reshape_op.h"

namespace jittor {

static auto make_reshape = op_constructor<VarPtr, Var*, NanoVector>("reshape");

ReshapeOp::ReshapeOp(Var* x, NanoVector shape) : x(x), shape(shape) {
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);
    y = create_output(nullptr, x->dtype());
}

VarPtr ReshapeOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return make_reshape(dout, x->shape);
}

void ReshapeOp::infer_shape() {
    size_t uncertain_dim = 0;
    int64_t y_items = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] < 0) {
            ++uncertain_dim;
        } else
            y_items *= shape[i];
    }
    USER_CHECK(uncertain_dim <= 1) << "max number of -1 is 1, but get" << uncertain_dim << ".";
    int64_t x_items = x->num;
    auto yshape = shape;
    if (uncertain_dim == 0) {
        USER_CHECKop(x_items,==,y_items) << "reshape shape is invalid for input of size";
    } else {
        if (x_items == 0) {
            uncertain_dim = 0;
        } else {
            USER_CHECK(y_items != 0 && x_items % y_items == 0) << "reshape shape is invalid for input of size " << x_items;
            uncertain_dim = x_items / y_items;
        }        
        yshape.clear();
        for (auto a : shape)
            yshape.push_back(a<0 ? uncertain_dim : a);
    }
    y->set_shape(yshape);
    if (!x->is_contiguous() && x->num) {
        vector<int64> strides(yshape.size());
        int vd = int(yshape.size())-1;
        int64 chunk_stride = x->storage_stride(x->shape.size()-1);
        int64 old_count = 1, new_count = 1;
        for (int d=int(x->shape.size())-1; d>=0; --d) {
            old_count *= x->shape[d];
            if (d && (x->shape[d-1] == 1 || x->storage_stride(d-1) == old_count*chunk_stride))
                continue;
            while (vd>=0 && (new_count<old_count || yshape[vd] == 1)) {
                strides[vd] = new_count*chunk_stride;
                new_count *= yshape[vd--];
            }
            USER_CHECK(new_count == old_count)
                << "view shape is incompatible with storage strides; call contiguous() first";
            if (d) chunk_stride = x->storage_stride(d-1);
            old_count = new_count = 1;
        }
        USER_CHECK(vd == -1) << "view shape is incompatible with storage strides";
        y->set_storage_strides(NanoVector::make(strides.data(), strides.size()));
    }
    y->share_with(x);
}
} // jittor
