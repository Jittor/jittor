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
#include "ops/op_register.h"
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

static auto make_transpose = get_op_info("transpose")
    .get_constructor<VarPtr, Var*, NanoVector>();
static auto make_cnnl_mlu_conv_backward_x = get_op_info("cnnl_conv_backward_x")
    .get_constructor<VarPtr, Var*, Var*, Var*, Var*, int, int, int, int, int, int, int, string, string, string>();
// static auto make_cnnl_mlu_conv_backward_w = get_op_info("cnnl_mlu_conv_backward_w")
//     .get_constructor<VarPtr, ......>();

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
    // dx = create_output(nullptr, ns_int8);
    if (!this->yformat.size())
        this->yformat = this->xformat;
}

VarPtr CnnlMluConvOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    // int8_t* input_mlu_ptr = (int8_t*)x->mem_ptr;
    // int8_t* filter_mlu_ptr = (int8_t*)w->mem_ptr;
    // float *output_mlu_ptr = (float*)y->mem_ptr;
    // LOGw << "CnnlMluConvOp::grad" << v_index;
    if (v_index == 0) {
        // LOGw << "Grad to x";
        return make_cnnl_mlu_conv_backward_x(out, dout, w, x, strideh, stridew, paddingh, paddingw, dilationh, dilationw, groups, xformat, wformat, yformat);
    }
    else
        return nullptr;
    // else
    //     return make_cnnl_mlu_conv_backward_w(out, dout, w, ....);
    // if (v_index == 0) {
    //     LOGw << "grad to x";
    //     LOGw << w->shape;

    //     int ni, ci, hi, wi, no, co, cw, ho, wo, kh, kw;
    //     get_shape(x, "abcd", xformat, ni, hi, wi, ci);
    //     get_shape(w, "oihw", wformat, co, kh, kw, cw);
    //     get_shape(y, "abcd", yformat, no, ho, wo, co);

    //     int input_dim[4] = {ni, hi, wi, ci};
    //     int weight_dim[4] = {co, kh, kw, cw};
    //     int output_dim[4] = {no, ho, wo, co};

    //     cnnlTensorDescriptor_t input_desc = nullptr;
    //     cnnlTensorDescriptor_t weight_desc = nullptr;
    //     cnnlTensorDescriptor_t output_desc = nullptr;
    //     cnnlTensorDescriptor_t grad_input_desc = nullptr;
    //     cnnlTensorDescriptor_t grad_desc = nullptr;
    //     cnnlSetTensorDescriptor(input_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_INT8, 4, input_dim);
    //     cnnlSetTensorDescriptor(weight_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_INT8, 4, weight_dim);
    //     cnnlSetTensorDescriptor(output_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, output_dim);
    //     cnnlSetTensorDescriptor(grad_input_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_INT8, 4, input_dim);
    //     cnnlSetTensorDescriptor(grad_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_INT8, 4, output_dim);

    //     cnnlConvolutionBwdDataPreference_t pre_t = CNNL_CONVOLUTION_BWD_DATA_FASTEST;
    //     cnnlConvolutionBwdDataAlgo_t algo_t;
    //     cnnlConvolutionDescriptor_t conv_desc;
    //     void *workspace = nullptr;
    //     size_t workspace_size = 0;
    //     cnnlHandle_t mlu_handle = nullptr;
    //     cnnlCreate(&mlu_handle);
    //     cnnlSetQueue(mlu_handle, mlu_queue);

    //     int pad[4] = {paddingh, paddingh, paddingw, paddingw};
    //     int stride[2] = {strideh, stridew};
    //     int dilation[2] = {dilationh, dilationw};
    //     cnnlSetConvolutionDescriptor(conv_desc, 4, pad, stride, dilation, groups,CNNL_DTYPE_FLOAT);

    //     cnnlGetConvolutionBackwardDataAlgorithm(mlu_handle, weight_desc, grad_desc, conv_desc, grad_input_desc, pre_t, &algo_t);

    //     cnnlGetConvolutionBackwardDataWorkspaceSize(mlu_handle, weight_desc, grad_desc, conv_desc, grad_input_desc, algo_t, &workspace_size);

    //     const void * alpha = nullptr;
    //     const void * beta = nullptr;

    //     cnnlConvolutionBackwardData(
    //     /* mlu_handle         */ mlu_handle,
    //     /* alpha          */ alpha,
    //     /* weight_desc    */ weight_desc,
    //     /* weight         */ filter_mlu_ptr,
    //     /* diff_y_desc    */ grad_desc,
    //     /* diff_y         */ dout->mem_ptr,
    //     /* conv_desc      */ conv_desc,
    //     /* algo           */ algo_t,
    //     /* workspace      */ workspace,
    //     /* workspace_size */ workspace_size,
    //     /* beta           */ beta,
    //     /* diff_x_desc    */ grad_input_desc,
    //     /* diff_x         */ dw->mem_ptr);
    //     return dx;
    //     // LOGw << dout->dtype();
    //     // auto dout_t = make_transpose(dout, {0,2,1,3});

    //     // void* dout_ptr = dout->mem_ptr;
    //     // float* cache_ptr = (float *)malloc(dout->size);
    //     // cnrtMemcpy(cache_ptr, dout_ptr, dout->size, CNRT_MEM_TRANS_DIR_DEV2HOST);
    //     // LOGw << (float)cache_ptr[0];
    //     // LOGw << (float)cache_ptr[2];
    //     // LOGw << (float)cache_ptr[3];
    //     // LOGw << dout->size;
    // }
    // else if (v_index == 1) {
    //     LOGw << "grad to w";

    //     /*
    //     int ni, ci, hi, wi, no, co, cw, ho, wo, kh, kw;
    //     get_shape(x, "abcd", xformat, ni, hi, wi, ci);
    //     get_shape(w, "oihw", wformat, co, kh, kw, cw);
    //     get_shape(y, "abcd", yformat, no, ho, wo, co);

    //     int input_dim[4] = {ni, hi, wi, ci};
    //     int weight_dim[4] = {co, kh, kw, cw};
    //     int output_dim[4] = {no, ho, wo, co};

    //     cnnlTensorDescriptor_t input_desc = nullptr;
    //     cnnlTensorDescriptor_t weight_desc = nullptr;
    //     cnnlTensorDescriptor_t output_desc = nullptr;
    //     cnnlTensorDescriptor_t grad_weight_desc = nullptr;
    //     cnnlTensorDescriptor_t grad_desc = nullptr;
    //     cnnlSetTensorDescriptor(input_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_INT8, 4, input_dim);
    //     cnnlSetTensorDescriptor(weight_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_INT8, 4, weight_dim);
    //     cnnlSetTensorDescriptor(output_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, output_dim);
    //     cnnlSetTensorDescriptor(grad_weight_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_INT8, 4, weight_dim);
    //     cnnlSetTensorDescriptor(grad_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_INT8, 4, output_dim);

    //     cnnlConvolutionBwdDataAlgo_t algo_t;
    //     cnnlConvolutionDescriptor_t conv_desc;
    //     void *workspace = nullptr;
    //     size_t workspace_size = 0;
    //     cnnlCreate(&mlu_handle);
    //     cnnlSetQueue(mlu_handle, mlu_queue);

    //     int pad[4] = {paddingh, paddingh, paddingw, paddingw};
    //     int stride[2] = {strideh, stridew};
    //     int dilation[2] = {dilationh, dilationw};
    //     cnnlSetConvolutionDescriptor(conv_desc, 4, pad, stride, dilation, groups, CNNL_DTYPE_FLOAT);

    //     cnnlGetConvolutionBackwardFilterWorkspaceSize(mlu_handle, input_desc, grad_desc, grad_weight_desc, conv_desc, algo_t, &workspace_size)

    //     const void * alpha = nullptr;
    //     const void * beta = nullptr;

    //     */

    //     // cnnlConvolutionBackwardFilter(
    //     // /* mlu_handle         */ mlu_handle,
    //     // /* alpha          */ alpha,
    //     // /* x_desc         */ input_desc,
    //     // /* x              */ input_mlu_ptr,
    //     // /* diff_y_desc    */ grad_desc,
    //     // /* diff_y         */ dout->mem_ptr,
    //     // /* conv_desc      */ conv_desc,
    //     // /* algo           */ algo_t,
    //     // /* workspace      */ workspace_ptr,
    //     // /* workspace_size */ workspace_size,
    //     // /* beta           */ beta,
    //     // /* diff_w_desc    */ grad_weight_desc,
    //     // /* diff_w         */ grad_weight_ptr);
    //     // LOGw << dout->dtype();
    //     // auto dout_t = make_transpose(dout, {0,2,1,3});

    //     // void* dout_ptr = dout->mem_ptr;
    //     // float* cache_ptr = (float *)malloc(dout->size);
    //     // cnrtMemcpy(cache_ptr, dout_ptr, dout->size, CNRT_MEM_TRANS_DIR_DEV2HOST);
    //     // LOGw << (float)cache_ptr[0];
    //     // LOGw << (float)cache_ptr[2];
    //     // LOGw << (float)cache_ptr[3];
    //     // LOGw << dout->size;
    // }
    // return v;
}

void CnnlMluConvOp::infer_shape() {
    ASSERTop(x->shape.size(),==,4);
    ASSERTop(w->shape.size(),==,4);
    int xn, xc, xh, xw, wh, ww, wci, wco, yn, yc, yh, yw;
    get_shape(x, "abcd", xformat, xn, xh, xw, xc);
    get_shape(w, "oihw", wformat, wco, wh, ww, wci);
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
    int8_t* input_mlu_ptr = (int8_t*)x->mem_ptr;
    int8_t* filter_mlu_ptr = (int8_t*)w->mem_ptr;
    float *output_mlu_ptr = (float*)y->mem_ptr;

    int ni, ci, hi, wi, no, co, cw, ho, wo, kh, kw;
    get_shape(x, "abcd", xformat, ni, hi, wi, ci);
    get_shape(w, "oihw", wformat, co, kh, kw, cw);
    get_shape(y, "abcd", yformat, no, ho, wo, co);

    // cnnlHandle_t mlu_handle = nullptr;
    cnnlTensorDescriptor_t input_desc = nullptr;
    cnnlTensorDescriptor_t weight_desc = nullptr;
    cnnlTensorDescriptor_t output_desc = nullptr;
    cnnlConvolutionDescriptor_t conv_desc = nullptr;
    cnnlConvolutionForwardAlgo_t algo = CNNL_CONVOLUTION_FWD_ALGO_DIRECT;
    void *workspace = nullptr;
    size_t workspace_size = 0;
    cnnlCreate(&mlu_handle);
    cnnlSetQueue(mlu_handle, mlu_queue);

    cnnlCreateTensorDescriptor(&input_desc);
    cnnlCreateTensorDescriptor(&weight_desc);
    cnnlCreateTensorDescriptor(&output_desc);

    int input_dim[4] = {ni, hi, wi, ci};
    int weight_dim[4] = {co, kh, kw, cw};
    int output_dim[4] = {no, ho, wo, co};

    cnnlSetTensorDescriptor(input_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_INT8, 4, input_dim);
    cnnlSetTensorDescriptor(weight_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_INT8, 4, weight_dim);
    cnnlSetTensorDescriptor(output_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, output_dim);

    int pad[4] = {paddingh, paddingh, paddingw, paddingw};
    int stride[2] = {strideh, stridew};
    int dilation[2] = {dilationh, dilationw};
    cnnlCreateConvolutionDescriptor(&conv_desc);
    cnnlSetConvolutionDescriptor(conv_desc, 4, pad, stride, dilation, groups, CNNL_DTYPE_FLOAT);

    cnnlConvolutionForward(mlu_handle, conv_desc, algo, nullptr, input_desc, input_mlu_ptr, weight_desc, filter_mlu_ptr, nullptr, nullptr, workspace, workspace_size, nullptr, output_desc, output_mlu_ptr);
}
#endif
#endif // JIT

} // jittor
