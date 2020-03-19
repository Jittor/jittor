// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guoye Yang <498731903@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include "var.h"
#include "ops/argsort_op.h"
#include <vector>
#include "executor.h"
#include "misc/cuda_flags.h"
#include "ops/op_register.h"
namespace jittor {

#ifndef JIT

static auto make_index = get_op_info("index")
    .get_constructor<VarPtr, NanoVector, int64, NanoString>();
static auto make_reindex_reduce = get_op_info("reindex_reduce")
    .get_constructor<VarPtr, Var*, NanoString, NanoVector, vector<string>&&, vector<string>&&, vector<Var*>&&>();

#ifdef HAS_CUDA
static auto make_array = get_op_info("array")
    .get_constructor<VarPtr, const void*, NanoVector, NanoString>();
static auto make_binary = get_op_info("binary")
    .get_constructor<VarPtr, Var*, Var*, NanoString>();
static auto make_transpose = get_op_info("transpose")
    .get_constructor<VarPtr, Var*, NanoVector>();
#endif

ArgsortOp::ArgsortOp(Var* x, int dim, bool descending, NanoString dtype)
    : x(x), dim(dim), descending(descending) {
    if  (this->dim == -1)
        this->dim = x->shape.size() - 1;
    dim = this->dim;
    #ifdef HAS_CUDA
    if (use_cuda) {
        static std::vector<VarPtr>(*cub_argsort)(Var*, Var*, Var*, bool, NanoString) = nullptr;
        if (!cub_argsort && has_op("cub_argsort")) {
            cub_argsort = get_op_info("cub_argsort")
                .get_constructor<std::vector<VarPtr>, Var*, Var*, Var*, bool, NanoString>();
        }
        if (cub_argsort) {
            if (x->num<0) exe.run_sync(vector<Var*>({x}), true);
            int dims = x->shape.size();
            vector<int64> axes;
            axes.reserve(dims);
            for (int i = 0; i < dims; ++i)
                if (i != dim)
                    axes.push_back(i);
            axes.push_back(dim);
            auto tranpose1 = make_transpose(x, axes);

            auto indexes = make_index(tranpose1->shape, dims - 1, ns_int32);
            int m = 1;
            for (int i = 0; i < dims - 1; ++i) {
                m *= tranpose1->shape[i];
            }
            int n = tranpose1->shape[dims - 1];
            auto one = make_array(&n, 1, ns_int32);
            auto offsets1 = make_index({m+1}, 0, ns_int32);
            auto offsets = make_binary(one, offsets1, ns_multiply);
            auto var = cub_argsort(tranpose1, indexes, offsets, descending, dtype);
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
            return;
        }
    }
    #endif
    y = create_output(nullptr, dtype);
    y_key = create_output(nullptr, x->dtype());
}

VarPtr ArgsortOp::get_grad(Var* out, Var* dout, Var* v, int v_index, int dim, Var* y) {
    // Do not have grad to extras input
    if (v_index) return nullptr;
    vector<VarPtr> indexes;
    vector<Var*> indexes_;
    vector<string> indexes__;
    auto& shape = v->shape;
    for (int i = 0; i < v->shape.size(); ++i) {
        if (i == dim) {
            indexes.push_back(y);
        } else {
            indexes.push_back(make_index(v->shape, i, ns_int32));
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
    
    return make_reindex_reduce(dout, ns_add, v->shape, move(indexes__), {}, move(indexes_));
}

VarPtr ArgsortOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    return get_grad(out, dout, v, v_index, dim, y);
}

void ArgsortOp::infer_shape() {
    ASSERTop(dim,>=,0);
    ASSERTop(dim,<,(int)x->shape.size());
    y->set_shape(x->shape);
    y_key->set_shape(x->shape);
}

void ArgsortOp::jit_prepare() {
    add_jit_define("Tx", x->dtype());
    add_jit_define("Ty", y->dtype());
    add_jit_define("XDIM", JK::hex1(x->shape.size()));
    add_jit_define("DIM", JK::hex1(dim));
    add_jit_define("CMP", descending ? ">" : "<");
}

#else // JIT
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-variable"
void ArgsortOp::jit_run() {
    auto* __restrict__ xp = x->ptr<Tx>();
    // define x shape
    @for(i, 0, XDIM, index_t xshape@i = x->shape[@i];)
    // define x stride
    index_t xstride@{XDIM-1} = 1;
    @for(i, XDIM-2, -1, -1, auto xstride@i = xstride@{i+1} * xshape@{i+1};)

    auto* __restrict__ yp = y->ptr<Ty>();
    auto* __restrict__ y_keyp = y_key->ptr<Tx>();
    std::vector<Tx> tempx(xshape@DIM);
    std::vector<Ty> tempy(xshape@DIM);

    @for(d, 0, DIM, for (index_t i@d=0; i@d < xshape@d; i@d++))
    @for(d, DIM+1, XDIM, for (index_t i@d=0; i@d < xshape@d; i@d++)) {
        for (index_t i@DIM=0; i@DIM < xshape@DIM; i@DIM++){
            auto xid = @for(d, 0, XDIM, + i@d * xstride@d);
            tempx[i@DIM] = xp[xid];
            tempy[i@DIM] = i@DIM;
        }
        std::sort(tempy.begin(), tempy.end(), [&](Ty i, Ty j) -> bool { return tempx[i]@CMP@@tempx[j]; });

        for (index_t i@DIM=0; i@DIM < xshape@DIM; i@DIM++){
            auto xid = @for(d, 0, XDIM, + i@d * xstride@d);
            yp[xid] = tempy[i@DIM];
            y_keyp[xid] = tempx[tempy[i@DIM]];
        }
    }
}
#endif // JIT

} // jittor