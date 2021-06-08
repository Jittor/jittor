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
#include "cnnl_mlu_conv_op.h"

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

CnnlMluConvOp::CnnlMluConvOp(Var* x, Var* w, int strideh, int stridew, int paddingh, int paddingw, int dilationh, int dilationw, int groups, string xformat, string wformat, string yformat)
    : x(x), w(w), strideh(strideh), stridew(stridew), paddingh(paddingh), paddingw(paddingw), dilationh(dilationh), dilationw(dilationw), groups(groups),
      xformat(move(xformat)), wformat(move(wformat)), yformat(move(yformat)) {
    y = create_output(nullptr, ns_float32);
    // LOGir << "shit";
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    if (!this->yformat.size())
        this->yformat = this->xformat;
}

void CnnlMluConvOp::infer_shape() {
    ASSERTop(x->shape.size(),==,4);
    ASSERTop(w->shape.size(),==,4);
    int xn, xc, xh, xw, wh, ww, wci, wco, yn, yc, yh, yw;
    get_shape(x, "abcd", xformat, xn, xh, xw, xc);
    get_shape(w, "oihw", wformat, wco, wh, ww, wci);
    // LOGw << xn << xc << xh << xw;
    // LOGw << wco << wci << wh << ww;
    ASSERTop(wci * groups,==,xc);
    yn = xn, yc = wco;
    yh = (xh+paddingh*2-wh*dilationh+dilationh-1)/strideh+1;
    yw = (xw+paddingw*2-ww*dilationw+dilationw-1)/stridew+1;
    set_shape(y, "abcd", yformat, yn, yh, yw, yc);
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

void CnnlMluConvOp::jit_prepare(JK& jk) {
    jk << _CS("[Txd:") << x->dtype();
    jk << _CS("][Tyd:") << y->dtype();
    jk << _CS("][Twd:") << w->dtype();
    jk << _CS("][Tx:") << short_type(x);
    jk << _CS("][Tw:") << short_type(w);
    jk << _CS("][Ty:") << short_type(y);
    jk << _CS("][XFORMAT:") << xformat;
    jk << _CS("][WFORMAT:") << wformat;
    jk << _CS("][YFORMAT:") << yformat;
    jk << ']';
}

unordered_map<string, CnnlMluConv_t> cnnl_mlu_conv_cache;

#else // JIT
#ifdef JIT_cpu
#pragma clang diagnostic ignored "-Wtautological-compare"

extern unordered_map<string, CnnlMluConv_t> cnnl_mlu_conv_cache;

void CnnlMluConvOp::jit_run() {
    // cnrtSyncQueue(mlu_queue);
    // auto start = std::chrono::high_resolution_clock::now();
    int8_t* input_mlu_ptr = (int8_t*)x->mem_ptr;
    int8_t* filter_mlu_ptr = (int8_t*)w->mem_ptr;
    float *output_mlu_ptr = (float*)y->mem_ptr;

    int ni, ci, hi, wi, no, co, cw, ho, wo, kh, kw;
    get_shape(x, "abcd", xformat, ni, hi, wi, ci);
    get_shape(w, "oihw", wformat, co, kh, kw, cw);
    get_shape(y, "abcd", yformat, no, ho, wo, co);

    // cnnlHandle_t handle = nullptr;
    cnnlTensorDescriptor_t input_desc = nullptr;
    cnnlTensorDescriptor_t weight_desc = nullptr;
    cnnlTensorDescriptor_t output_desc = nullptr;
    cnnlConvolutionDescriptor_t conv_desc = nullptr;
    cnnlConvolutionForwardAlgo_t algo = CNNL_CONVOLUTION_FWD_ALGO_DIRECT;
    void *workspace = nullptr;
    size_t workspace_size = 0;
    // cnnlCreate(&handle);
    // cnnlSetQueue(handle, mlu_queue);

    cnnlCreateTensorDescriptor(&input_desc);
    cnnlCreateTensorDescriptor(&weight_desc);
    cnnlCreateTensorDescriptor(&output_desc);

    int input_dim[4] = {ni, hi, wi, ci};
    int weight_dim[4] = {co, kh, kw, cw};
    int output_dim[4] = {no, ho, wo, co};
    cnnlSetTensorDescriptor(input_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_INT8, 4, input_dim);
    cnnlSetTensorDescriptor(weight_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_INT8, 4, weight_dim);
    cnnlSetTensorDescriptor(output_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, output_dim);

    // int input_dim[4] = {ni, ci, hi, wi};
    // int weight_dim[4] = {co, cw, kh, kw};
    // int output_dim[4] = {no, co, ho, wo};
    // LOGw << ni << ci << hi << wi;
    // LOGw << co << cw << kh << kw;
    // LOGw << no << co << ho << wo;
    // LOGw << x->shape;
    // LOGw << w->shape;
    // LOGw << y->shape;
    // cnnlSetTensorDescriptor(input_desc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT8, 4, input_dim);
    // cnnlSetTensorDescriptor(weight_desc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_INT8, 4, weight_dim);
    // cnnlSetTensorDescriptor(output_desc, CNNL_LAYOUT_NCHW, CNNL_DTYPE_FLOAT, 4, output_dim);

    int pad[4] = {paddingh, paddingh, paddingw, paddingw};
    int stride[2] = {strideh, stridew};
    int dilation[2] = {dilationh, dilationw};
    cnnlCreateConvolutionDescriptor(&conv_desc);
    cnnlSetConvolutionDescriptor(conv_desc, 4, pad, stride, dilation, groups, CNNL_DTYPE_FLOAT);

    cnnlConvolutionForward(mlu_handle, conv_desc, algo, nullptr, input_desc, input_mlu_ptr, weight_desc, filter_mlu_ptr, nullptr, nullptr, workspace, workspace_size, nullptr, output_desc, output_mlu_ptr);

    // cnrtSyncQueue(mlu_queue);
    // auto finish = std::chrono::high_resolution_clock::now();
    // auto total_ns =  (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count() / 1000.;
    // (void) total_ns;
    // LOGw << total_ns << " ms";
    return;
    // int input_count = ni * hi * wi * ci;
    // int8_t* input_cpu_ptr = (int8_t *)malloc(input_count * sizeof(int8_t));
    // cnrtMemcpy(input_cpu_ptr, input_mlu_ptr, input_count * siz eof(int8_t),
    //           CNRT_MEM_TRANS_DIR_DEV2HOST);
    // LOGw << (int)input_cpu_ptr[0];
    // LOGw << (int)input_cpu_ptr[1];
    // LOGw << (int)input_cpu_ptr[2];
    // LOGw << (int)input_cpu_ptr[3];

    // int filter_count = co * kh * kw * cw;
    // int8_t* filter_cpu_ptr = (int8_t *)malloc(filter_count * sizeof(int8_t));
    // cnrtMemcpy(filter_cpu_ptr, filter_mlu_ptr, filter_count * sizeof(int8_t),
    //           CNRT_MEM_TRANS_DIR_DEV2HOST);
    // LOGw << (int)filter_cpu_ptr[0];
    // LOGw << (int)filter_cpu_ptr[1];
    // LOGw << (int)filter_cpu_ptr[2];
    // LOGw << (int)filter_cpu_ptr[3];

    // LOGw << input_desc;
    // LOGw << weight_desc;
    // LOGw << output_desc;
    // LOGw << conv_desc;
    // LOGw << algo;
    // LOGw << workspace;
    // LOGw << workspace_size;
}
#endif
#endif // JIT

} // jittor
