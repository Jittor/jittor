// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Wenyang Zhou <576825820@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "cnnl_max_pool_op.h"

#include <string.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include "mlu_warper.h"
#include "executor.h"
// #include "/data/zwy/jittor/src/profiler/simple_profiler.h"

using namespace std;

namespace jittor {

static inline int findc(const string& format, const char& c) {
    if (c==format[0]) return 0;
    if (c==format[1]) return 1;
    if (c==format[2]) return 2;
    ASSERT(c==format[3]) << "Not a valid format" << format << c;
    return 3;
}

static inline void get_shape(Var* x, const char* f, const string& format, int& a, int& b, int &c, int& d) {
    auto& shape = x->shape;
    a = shape[findc(format, f[0])];
    b = shape[findc(format, f[1])];
    c = shape[findc(format, f[2])];
    d = shape[findc(format, f[3])];
}

#ifndef JIT

static inline void set_shape(Var* x, const char* f, const string& format, int a, int b, int c, int d) {
    int64 shape[4];
    shape[findc(format, f[0])] = a;
    shape[findc(format, f[1])] = b;
    shape[findc(format, f[2])] = c;
    shape[findc(format, f[3])] = d;
    x->set_shape(NanoVector(
        shape[0], shape[1], shape[2], shape[3]));
}

CnnlMaxPoolOp::CnnlMaxPoolOp(Var* x, int kernel_size, int stride, int padding, int dilation, int pool_mode_row, bool ceil_mode, bool count_include_pad, string xformat, string yformat)
    : x(x), kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation), pool_mode_row(pool_mode_row), ceil_mode(ceil_mode), count_include_pad(count_include_pad),
      xformat(move(xformat)), yformat(move(yformat)) {
    y = create_output(nullptr, ns_float32);
    index = create_output(nullptr, ns_int32);
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    if (!this->yformat.size())
        this->yformat = this->xformat;
}

void CnnlMaxPoolOp::infer_shape() {
    ASSERTop(x->shape.size(),==,4);
    int xn, xc, xh, xw, yn, yc, yh, yw;
    get_shape(x, "abcd", xformat, xn, xc, xh, xw);
    yn = xn, yc = xc;
    if (ceil_mode == false) {
        yh = (xh+padding*2-kernel_size)/stride+1;
        yw = (xw+padding*2-kernel_size)/stride+1;
    }
    else {
        yh = (xh+padding*2-kernel_size + stride - 1)/stride+1;
        yw = (xw+padding*2-kernel_size + stride - 1)/stride+1;
    }
    set_shape(y, "abcd", yformat, yn, yc, yh, yw);
    set_shape(index, "abcd", yformat, yn, yc, yh, yw);
}

static const char* short_type(Var* x) {
    if (x->is_float()) {
        if (x->dsize()==4) return "f32";
        if (x->dsize()==8) return "f64";
        if (x->dsize()==2) return "f16";
        return "f8";
    } else {
        if (x->dsize()==4) return "s32";
        if (x->dsize()==8) return "s64";
        if (x->dsize()==2) return "s16";
        return "s8";
    }
}

void CnnlMaxPoolOp::jit_prepare(JK& jk) {
    jk << _CS("[Txd:") << x->dtype();
    jk << _CS("][Tyd:") << y->dtype();
    jk << _CS("][Tx:") << short_type(x);
    jk << _CS("][Ty:") << short_type(y);
    jk << _CS("][XFORMAT:") << xformat;
    jk << _CS("][YFORMAT:") << yformat;
    jk << ']';
}

#else // JIT
#ifdef JIT_cpu
#pragma clang diagnostic ignored "-Wtautological-compare"

void CnnlMaxPoolOp::jit_run() {
    // LOGw << "CnnlMaxPoolOp::jit_run";
    ASSERT(dilation==0);

    int ni, ci, hi, wi, no, co, ho, wo;
    get_shape(x, "abcd", xformat, ni, ci, hi, wi);
    get_shape(y, "abcd", yformat, no, co, ho, wo);

    // cnnlHandle_t handle = nullptr;
    cnnlTensorDescriptor_t input_desc = nullptr;
    cnnlTensorDescriptor_t output_desc = nullptr;
    cnnlTensorDescriptor_t index_desc = nullptr;
    cnnlPoolingDescriptor_t pooling_desc = nullptr;
    

    cnnlCreateTensorDescriptor(&input_desc);
    cnnlCreateTensorDescriptor(&output_desc);
    cnnlCreateTensorDescriptor(&index_desc);

    int input_dim[4] = {ni, ci, hi, wi};
    int output_dim[4] = {no, co, ho, wo};
    cnnlSetTensorDescriptor(input_desc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, 4, input_dim);
    cnnlSetTensorDescriptor(output_desc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, 4, output_dim);
    cnnlSetTensorDescriptor(index_desc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT32, 4, output_dim);

    cnnlCreatePoolingDescriptor(&pooling_desc);
    cnnlSetPooling2dDescriptor(pooling_desc, CNNL_POOLING_MAX, CNNL_PROPAGATE_NAN, kernel_size, kernel_size, padding, padding, padding, padding, stride, stride);
    
    void *workspace = nullptr;
    size_t workspace_size = 0;
    cnnlGetPoolingWithIndexWorkspaceSize(mlu_handle, input_desc, output_desc, &workspace_size);
    // if (workspace_size != 0) {
    //     cnrtMalloc(&workspace, workspace_size);
    //     cnrtMemset(workspace, 0, workspace_size);
    // }
    size_t allocation;
    if (workspace_size > 0) {
        workspace = exe.temp_allocator->alloc(workspace_size, allocation);
    }

    const void * alpha = nullptr;
    const void * beta = nullptr;
    cnnlPoolingForwardWithIndex(
    /* handle         */ mlu_handle,
    /* pooling_desc   */ pooling_desc,
    /* alpha          */ alpha,
    /* x_desc         */ input_desc,
    /* x              */ x->mem_ptr,
    /* beta           */ beta,
    /* y_desc         */ output_desc,
    /* y              */ y->mem_ptr,
    /* index_desc     */ index_desc,
    /* index          */ index->mem_ptr,
    /* workspace      */ workspace,
    /* workspace_size */ workspace_size);
    cnrtSyncQueue(mlu_queue);
    if (workspace)
        exe.temp_allocator->free(workspace, workspace_size, allocation);
    return;
}
#endif
#endif // JIT

} // jittor
