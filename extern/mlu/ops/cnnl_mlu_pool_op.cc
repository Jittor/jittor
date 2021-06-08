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
#include "cnnl_mlu_pool_op.h"

#include <string.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include "mlu_warper.h"

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

CnnlMluPoolOp::CnnlMluPoolOp(Var* x, int kernel_size, int stride, int padding, int dilation, int pool_mode_row, bool ceil_mode, bool count_include_pad, string xformat, string yformat, string op)
    : x(x), kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation), pool_mode_row(pool_mode_row), ceil_mode(ceil_mode), count_include_pad(count_include_pad),
      xformat(move(xformat)), yformat(move(yformat)), op(move(op)) {
    y = create_output(nullptr, ns_float32);
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    if (!this->yformat.size())
        this->yformat = this->xformat;
}

void CnnlMluPoolOp::infer_shape() {
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

void CnnlMluPoolOp::jit_prepare(JK& jk) {
    jk << _CS("[Txd:") << x->dtype();
    jk << _CS("][Tyd:") << y->dtype();
    jk << _CS("][Tx:") << short_type(x);
    jk << _CS("][Ty:") << short_type(y);
    jk << _CS("][XFORMAT:") << xformat;
    jk << _CS("][YFORMAT:") << yformat;
    jk << ']';
}

unordered_map<string, CnnlMluPool_t> cnnl_mlu_pool_cache;

#else // JIT
#ifdef JIT_cpu
#pragma clang diagnostic ignored "-Wtautological-compare"
extern unordered_map<string, CnnlMluPool_t> cnnl_mlu_pool_cache;

void CnnlMluPoolOp::jit_run() {
    ASSERT(dilation==0);
    float* input_mlu_ptr = (float*)x->mem_ptr;
    float *output_mlu_ptr = (float*)y->mem_ptr;

    int ni, ci, hi, wi, no, co, ho, wo;
    get_shape(x, "abcd", xformat, ni, ci, hi, wi);
    get_shape(y, "abcd", yformat, no, co, ho, wo);


    cnnlTensorDescriptor_t input_desc = nullptr;
    cnnlTensorDescriptor_t output_desc = nullptr;
    cnnlPoolingDescriptor_t pooling_desc = nullptr;
    void *workspace = nullptr;
    size_t workspace_size = 0;

    cnnlCreateTensorDescriptor(&input_desc);
    cnnlCreateTensorDescriptor(&output_desc);

    int input_dim[4] = {ni, ci, hi, wi};
    int output_dim[4] = {no, co, ho, wo};
    cnnlSetTensorDescriptor(input_desc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, 4, input_dim);
    cnnlSetTensorDescriptor(output_desc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, 4, output_dim);

    // Determine the pooling mode
    cnnlPoolingMode_t mode;
    if (op == ns_maximum) mode = CNNL_POOLING_MAX;
    else mode =  CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

    cnnlCreatePoolingDescriptor(&pooling_desc);
    cnnlSetPooling2dDescriptor(pooling_desc, mode, CNNL_PROPAGATE_NAN, kernel_size, kernel_size, padding, padding, padding, padding, stride, stride);
    
    const void * alpha = nullptr;
    const void * beta = nullptr;

    cnnlPoolingForward(
        /* handle         */ mlu_handle,
        /* pooling_desc   */ pooling_desc,
        /* alpha          */ alpha,
        /* x_desc         */ input_desc,
        /* x              */ input_mlu_ptr,
        /* beta           */ beta,
        /* y_desc         */ output_desc,
        /* y              */ output_mlu_ptr,
        /* workspace      */ workspace,
        /* workspace_size */ workspace_size);
    return;
}
#endif
#endif // JIT

} // jittor
