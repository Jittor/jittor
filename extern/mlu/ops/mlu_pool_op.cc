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
#include "mlu_pool_op.h"

#include <string.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include "mlu_warper.h"
#include "/data/zwy/jittor/src/profiler/simple_profiler.h"

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

MluPoolOp::MluPoolOp(Var* x, int kernel_size, int stride, int padding, int dilation, int pool_mode_row, bool ceil_mode, bool count_include_pad, string xformat, string yformat, string op)
    : x(x), kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation), pool_mode_row(pool_mode_row), ceil_mode(ceil_mode), count_include_pad(count_include_pad),
      xformat(move(xformat)), yformat(move(yformat)), op(move(op)) {
    y = create_output(nullptr, ns_float32);
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    if (!this->yformat.size())
        this->yformat = this->xformat;
}

void MluPoolOp::infer_shape() {
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

void MluPoolOp::jit_prepare(JK& jk) {
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

void MluPoolOp::jit_run() {
    const int dimNum = 4;

    int ni, ci, hi, wi, no, co, ho, wo;
    get_shape(x, "abcd", xformat, ni, ci, hi, wi);
    get_shape(y, "abcd", yformat, no, co, ho, wo);

    float* input_mlu_ptr = (float*)x->mem_ptr;
    float *output_mlu_ptr = (float*)y->mem_ptr;

    int input_shape[] = {ni, ci, hi, wi};
    int output_shape[] = {no, co, ho, wo};

    cnmlTensor_t input_tensor = NULL;
    cnmlTensor_t output_tensor = NULL;
    cnmlCreateTensor_V2(&input_tensor, CNML_TENSOR);
    cnmlSetTensorShape_V2(input_tensor, dimNum, input_shape, NULL);
    cnmlCreateTensor_V2(&output_tensor, CNML_TENSOR);
    cnmlSetTensorShape_V2(output_tensor, dimNum, output_shape, NULL);

    cnmlSetTensorDataType(input_tensor, CNML_DATA_FLOAT32);
    cnmlSetTensorDataType(output_tensor, CNML_DATA_FLOAT32);

    cnmlPoolOpParam_t pool_param;

    if (op == ns_maximum) {
        cnmlCreatePoolOpParam(&pool_param, kernel_size, kernel_size, stride, stride, 2 * padding, 2 * padding, dilation, dilation, CNML_POOL_MAX, CNML_POOL_KFULL, !count_include_pad);
    }
    else {
        cnmlCreatePoolOpParam(&pool_param, kernel_size, kernel_size, stride, stride, 2 * padding, 2 * padding, dilation, dilation, CNML_POOL_AVG, CNML_POOL_KFULL, !count_include_pad);
    }
    
    cnmlBaseOp_t pool_op;
    cnmlCreatePoolOp(&pool_op, pool_param, input_tensor, output_tensor);
    cnmlSetOperationComputingLayout(pool_op, CNML_NCHW);
    cnmlCompileBaseOp_V2(pool_op);
    cnmlComputePoolOpForward_V4(pool_op, NULL, input_mlu_ptr, NULL, output_mlu_ptr, mlu_queue, NULL);
    JT_MLU_CHECK(cnrtSyncQueue(mlu_queue));
    return;
}
#endif
#endif // JIT

} // jittor
