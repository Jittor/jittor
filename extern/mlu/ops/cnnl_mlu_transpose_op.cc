// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "cnnl_mlu_transpose_op.h"
#include "ops/op_register.h"
#include "misc/stack_vector.h"
#include "mlu_warper.h"

namespace jittor {

#ifndef JIT

static auto make_cnnl_transpose = get_op_info("cnnl_mlu_transpose")
    .get_constructor<VarPtr, Var*, NanoVector>();

CnnlMluTransposeOp::CnnlMluTransposeOp(Var* x, NanoVector axes) : x(x), axes(axes) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    int i=0;
    for (; i<axes.size(); i++)
        if (i!=axes[i]) break;
    if (i==axes.size() && axes.size()) {
        forward(x);
        return;
    }
    y = create_output(nullptr, x->dtype());
}

void CnnlMluTransposeOp::infer_shape() {
    auto xdim = x->shape.size();
    CHECK(xdim);
    if (!axes.size()) {
        for (int i=0; i<(int)xdim; i++)
            axes.push_back(xdim-1-i);
    } else {
        CHECKop(axes.size(),==,xdim);
        int64_t mask=0;
        for (auto i : axes) mask |= 1<<i;
        CHECK(mask==((1ll<<xdim)-1)) << "Invalid axes" << axes;
    }
    NanoVector shape;
    for (uint i=0; i<xdim; i++)
        shape.push_back(x->shape[axes[i]]);
    y->set_shape(shape);
}

VarPtr CnnlMluTransposeOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    NanoVector reverse;
    reverse.reserve(axes.size(), axes.size());
    for (uint i=0; i<axes.size(); i++)
        reverse.set_data(axes[i], i);
    return make_cnnl_transpose(dout, reverse);
}

unordered_map<string, unsigned int> cutt_plan_cache;


void CnnlMluTransposeOp::jit_prepare(JK& jk) {
    jk << _CS("[Tx:") << x->dtype() << ']';
}

#else // JIT
void CnnlMluTransposeOp::jit_run() {
    Tx* input_mlu_ptr = (Tx*)x->mem_ptr;
    Tx *output_mlu_ptr = (Tx*)y->mem_ptr;
    
    cnnlTensorDescriptor_t input_desc = nullptr;
    cnnlTensorDescriptor_t output_desc = nullptr;
    cnnlTransposeDescriptor_t transpose_desc = nullptr;

    cnnlCreateTensorDescriptor(&input_desc);
    cnnlCreateTensorDescriptor(&output_desc);

    int dims=x->shape.size();
    vector<int> xshape;
    vector<int> yshape;
    vector<int> permute;
    for (int i=0;i<dims;i++){
        xshape.push_back(x->shape[i]);
        yshape.push_back(y->shape[i]);
        permute.push_back(axes[i]);
    }

    @define(Tx2,
        @if(@strcmp(@Tx,half)==0 || @strcmp(@Tx,float16)==0, CNNL_DTYPE_HALF)
        @if(@strcmp(@Tx,float)==0 || @strcmp(@Tx,float32)==0, CNNL_DTYPE_FLOAT)
        @if(@strcmp(@Tx,int8)==0, CNNL_DTYPE_INT8)
        @if(@strcmp(@Tx,int16)==0, CNNL_DTYPE_INT16)
        @if(@strcmp(@Tx,int)==0 || @strcmp(@Tx,int32)==0, CNNL_DTYPE_INT32)
        @if(@strcmp(@Tx,int64)==0, CNNL_DTYPE_INT64)
    )
    cnnlSetTensorDescriptor(input_desc, CNNL_LAYOUT_NCHW, @Tx2, dims, &xshape[0]);
    cnnlSetTensorDescriptor(output_desc, CNNL_LAYOUT_NCHW, @Tx2, dims, &yshape[0]);

    cnnlCreateTransposeDescriptor(&transpose_desc);
    cnnlSetTransposeDescriptor(transpose_desc, dims, &permute[0]);

    cnnlTranspose(
        /* handle         */ mlu_handle,
        /* transpose_desc */ transpose_desc,
        /* x_desc         */ input_desc,
        /* x              */ input_mlu_ptr,
        /* y_desc         */ output_desc,
        /* y              */ output_mlu_ptr);
    return;
}

#endif // JIT

} // jittor