// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include "var.h"
#include "ops/arg_reduce_op.h"
#include <vector>
#include "executor.h"
#include "misc/cuda_flags.h"
#include "ops/op_register.h"

namespace jittor {

#ifndef JIT

#ifdef HAS_CUDA
static auto make_array = get_op_info("array")
    .get_constructor<VarPtr, const void*, NanoVector, NanoString>();
static auto make_binary = get_op_info("binary")
    .get_constructor<VarPtr, Var*, Var*, NanoString>();
static auto make_transpose = get_op_info("transpose")
    .get_constructor<VarPtr, Var*, NanoVector>();
#endif

static auto make_index = get_op_info("index")
    .get_constructor<VarPtr, NanoVector, int64, NanoString>();
static auto make_reshape = get_op_info("reshape")
    .get_constructor<VarPtr, Var*, NanoVector>();
static auto make_reindex_reduce = get_op_info("reindex_reduce")
    .get_constructor<VarPtr, Var*, NanoString, NanoVector, vector<string>&&, vector<string>&&, vector<Var*>&&>();

ArgReduceOp::ArgReduceOp(Var* x, NanoString op, int dim, bool keepdims)
    : x(x), op(op), dim(dim), keepdims(keepdims) {
    if  (this->dim == -1)
        this->dim = x->shape.size() - 1;
    dim = this->dim;
    #ifdef HAS_CUDA
    if (use_cuda) {
        static auto cub_arg_reduce = has_op("cub_arg_reduce") ?
            get_op_info("cub_arg_reduce").get_constructor<std::vector<VarPtr>, Var*, Var*, NanoString, bool>()
            : nullptr;
        if (cub_arg_reduce) {
            if (x->num<0) exe.run_sync(vector<Var*>({x}), true);
            int dims = x->shape.size();
            vector<int64> axes;
            axes.reserve(dims);
            for (int i = 0; i < dims; ++i)
                if (i != dim)
                    axes.push_back(i);
            axes.push_back(dim);
            auto tranpose1 = make_transpose(x, axes);

            int m = 1;
            for (int i = 0; i < dims - 1; ++i) {
                m *= tranpose1->shape[i];
            }
            int n = tranpose1->shape[dims - 1];
            auto one = make_array(&n, 1, ns_int32);
            auto offsets1 = make_index({m+1}, 0, ns_int32);
            auto offsets = make_binary(one, offsets1, ns_multiply);
            auto var = cub_arg_reduce(tranpose1, offsets, op, keepdims);
            if (keepdims) {
                vector<int64> axes2;
                axes2.reserve(dims);
                for (int i = 0; i < dims; ++i) {
                    if (i == dim) axes2.push_back(dims - 1);
                    if (i < dims - 1) axes2.push_back(i);
                }
                auto tranpose2_0 = make_transpose(var[0], axes2);
                auto tranpose2_1 = make_transpose(var[1], axes2);
                forward(tranpose2_0);
                forward(tranpose2_1);
            } else {
                auto tranpose2_0 = var[0];
                auto tranpose2_1 = var[1];
                forward(tranpose2_0);
                forward(tranpose2_1);
            }
            return;
        }
    }
    #endif
    y = create_output(nullptr, ns_int32);
    y_key = create_output(nullptr, x->dtype());
}
VarPtr ArgReduceOp::get_grad(Var* out, Var* dout, Var* v, int v_index, int dim, Var* y) {
    // Do not have grad to extras input
    if (v_index) return nullptr;
    vector<int64> shape;
    shape.reserve(v->shape.size());
    for (int i = 0; i < v->shape.size(); ++i)
        if (i == dim) {
            shape.push_back(1);
        } else {
            shape.push_back(v->shape[i]);
        }
    auto reshape1 = make_reshape(dout, NanoVector(shape));
    auto reshapey = make_reshape(y, shape);

    vector<VarPtr> indexes;
    vector<Var*> indexes_;
    vector<string> indexes__;
    // auto& shape = v->shape;
    for (int i = 0; i < shape.size(); ++i) {
        if (i == dim) {
            indexes.push_back(reshapey);
        } else {
            indexes.push_back(make_index(shape, i, ns_int32));
        }
        indexes_.push_back(indexes.back());
    }

    string temp;
    temp.reserve(6+3*shape.size()); // @e0(i0,i1)
    temp += "@e0(";
    for (uint i=0; i<shape.size(); i++) {
        if (i) temp += ',';
        temp += 'i';
        temp += '0'+i;
    };
    temp += ')';
    indexes__.reserve(indexes.size());
    for (uint i=0; i<indexes.size(); i++) {
        temp[2] = '0'+i; // @ei
        indexes__.emplace_back(temp);
    }
    
    return make_reindex_reduce(reshape1, ns_add, v->shape, move(indexes__), {}, move(indexes_));
}

VarPtr ArgReduceOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return get_grad(out, dout, v, v_index, dim, y);
}

void ArgReduceOp::infer_shape() {
    ASSERTop(dim,>=,0);
    ASSERTop(dim,<,(int)x->shape.size());
    NanoVector shape;
    for (int i = 0; i < x->shape.size(); ++i) {
        if (i == dim) {
            if (keepdims)
                shape.push_back(1);
        } else {
            shape.push_back(x->shape[i]);
        }
    }
    if (shape.size() == 0)
        shape.push_back(1);
    y->set_shape(shape);
    y_key->set_shape(shape);
}

void ArgReduceOp::jit_prepare(JK& jk) {
    jk << _CS("[Tx:") << x->dtype();
    jk << _CS("][Ty:") << y->dtype();
    jk << _CS("][XDIM=") << JK::hex1(x->shape.size());
    jk << _CS("][YDIM=") << JK::hex1(y->shape.size());
    jk << _CS("][KEEPDIMS:") << (keepdims ? '1' : '0');
    jk << _CS("][DIM=") << JK::hex1(dim);
    jk << _CS("][CMP:") << (op==ns_minimum ? "<" : ">");
    jk << ']';
}

#else // JIT
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-variable"
void ArgReduceOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    // define x shape
    @for(i, 0, XDIM, index_t xshape@i = x->shape[@i];)
    // define x stride
    index_t xstride@{XDIM-1} = 1;
    @for(i, XDIM-2, -1, -1, auto xstride@i = xstride@{i+1} * xshape@{i+1};)

    // define y shape
    @for(i, 0, YDIM, index_t yshape@i = y->shape[@i];)
    // define y stride
    index_t ystride@{YDIM-1} = 1;
    @for(i, YDIM-2, -1, -1, auto ystride@i = ystride@{i+1} * yshape@{i+1};)

    auto* __restrict__ yp = y->ptr<Ty>();
    auto* __restrict__ y_keyp = y_key->ptr<Tx>();

    @for(d, 0, DIM, for (index_t i@d=0; i@d < xshape@d; i@d++))
    @for(d, DIM+1, XDIM, for (index_t i@d=0; i@d < xshape@d; i@d++)) {
        auto yid = 0@for(d, 0, DIM, + i@d * ystride@d);
        @if(KEEPDIMS, yid += 0 @for(d, DIM + 1, XDIM, + i@d * ystride@d), yid += 0 @for(d, DIM + 1, XDIM, + i@d * ystride@{d-1}));

        auto x0id = 0@for(d, 0, DIM, + i@d * xstride@d);
        x0id += 0 @for(d, DIM + 1, XDIM, + i@d * xstride@d);

        y_keyp[yid] = xp[x0id];
        yp[yid] = 0;

        for (index_t i@DIM=0; i@DIM < xshape@DIM; i@DIM++){
            auto xid = @for(d, 0, XDIM, + i@d * xstride@d);
            if (xp[xid]@CMP@@y_keyp[yid]) {
                y_keyp[yid] = xp[xid];
                yp[yid] = i@DIM;
            }
        }
    }
}
#endif // JIT

} // jittor
