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
#include "cnnl_pool_backward_op.h"

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

CnnlPoolBackwardOp::CnnlPoolBackwardOp(Var* grad, Var *x, Var *index, int kernel_size, int stride, int padding, bool ceil_mode, bool count_include_pad, string xformat, string op)
    : grad(grad), x(x), index(index), kernel_size(kernel_size), stride(stride), padding(padding), ceil_mode(ceil_mode), count_include_pad(count_include_pad),
      xformat(move(xformat)), op(move(op)) {
    dx = create_output(nullptr, ns_float32);
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
}

void CnnlPoolBackwardOp::infer_shape() {
    ASSERTop(x->shape.size(),==,4);
    int xn, xc, xh, xw;
    get_shape(x, "abcd", xformat, xn, xc, xh, xw);
    set_shape(dx, "abcd", xformat, xn, xc, xh, xw);
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

void CnnlPoolBackwardOp::jit_prepare(JK& jk) {
    jk << _CS("[Txd:") << x->dtype();
    jk << _CS("][Tx:") << short_type(x);
    jk << _CS("][XFORMAT:") << xformat;
    jk << ']';
}


#else // JIT
#ifdef JIT_cpu
#pragma clang diagnostic ignored "-Wtautological-compare"

void CnnlPoolBackwardOp::jit_run() {
    int ni, ci, hi, wi, gn, gc, gh, gw;
    get_shape(x, "abcd", xformat, ni, ci, hi, wi);
    get_shape(grad, "abcd", xformat, gn, gc, gh, gw);
    // LOGw << ni << ci << hi << wi << gn << gc << gh << gw;

    cnnlTensorDescriptor_t input_desc = nullptr;
    cnnlTensorDescriptor_t index_desc = nullptr;
    cnnlTensorDescriptor_t grad_desc = nullptr;
    cnnlTensorDescriptor_t output_desc = nullptr;
    cnnlCreateTensorDescriptor(&input_desc);
    cnnlCreateTensorDescriptor(&index_desc);
    cnnlCreateTensorDescriptor(&grad_desc);
    cnnlCreateTensorDescriptor(&output_desc);
    int input_dim[4] = {ni, ci, hi, wi};
    int grad_dim[4] = {gn, gc, gh, gw};
    cnnlSetTensorDescriptor(input_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, input_dim);
    cnnlSetTensorDescriptor(index_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_INT32, 4, grad_dim);
    cnnlSetTensorDescriptor(grad_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, grad_dim);
    //cnnlSetTensorDescriptor(output_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, input_dim);
    cnnlSetTensorDescriptor(output_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, input_dim);

    cnnlPoolingMode_t mode;
    void* index_ptr = nullptr;
    if (op == ns_maximum) {
        mode = CNNL_POOLING_MAX;
        index_ptr = (int32*)index->mem_ptr;
    }
    else mode =  CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    //else mode =  CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

    cnnlPoolingDescriptor_t pooling_desc = nullptr;
    cnnlCreatePoolingDescriptor(&pooling_desc);
    cnnlSetPooling2dDescriptor(pooling_desc, mode, CNNL_PROPAGATE_NAN, kernel_size, kernel_size, padding, padding, padding, padding, stride, stride);

    const void * alpha = nullptr;
    const void * beta = nullptr;
    cnnlPoolingBackward(
    /* handle       */ mlu_handle,
    /* pooling_desc */ pooling_desc,
    /* alpha        */ alpha,
    /* y_desc       */ index_desc,
    /* y            */ index->mem_ptr,
    /* diff_y_desc  */ grad_desc,
    /* diff_y       */ grad->mem_ptr,
    /* x_desc       */ input_desc,
    /* x            */ nullptr,
    /* beta         */ beta,
    /* diff_x_desc  */ output_desc,
    /* diff_x       */ dx->mem_ptr);
}
#endif
#endif // JIT

} // jittor
