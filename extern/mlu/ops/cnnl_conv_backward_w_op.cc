// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Wenyang Zhou <576825820@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <random>

#include "var.h"
#include "mlu_warper.h"
#include "cnnl_conv_backward_w_op.h"

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

CnnlConvBackwardWOp::CnnlConvBackwardWOp(Var* out, Var* dout, Var* w, Var* x, int strideh, int stridew, int paddingh, int paddingw, int dilationh, int dilationw, int groups, string xformat, string wformat, string yformat) 
        : out(out), dout(dout), w(w), x(x), strideh(strideh), stridew(stridew), paddingh(paddingh), paddingw(paddingw), dilationh(dilationh), dilationw(dilationw), groups(groups),
      xformat(move(xformat)), wformat(move(wformat)), yformat(move(yformat)) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    dw = create_output(nullptr, ns_float32);
}

void CnnlConvBackwardWOp::infer_shape() {
    int xn, xh, xw, xc;
    get_shape(w, "abcd", xformat, xn, xh, xw, xc);
    set_shape(dw, "abcd", xformat, xn, xh, xw, xc);
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

void CnnlConvBackwardWOp::jit_prepare(JK& jk) {
    jk << _CS("[Twd:") << w->dtype();
    jk << _CS("][Tdwd:") << dw->dtype();
    jk << _CS("][Tdx:") << short_type(dw);
    jk << _CS("][Tw:") << short_type(w);
    jk << _CS("][XFORMAT:") << xformat;
    jk << _CS("][WFORMAT:") << wformat;
    jk << _CS("][YFORMAT:") << yformat;
    jk << ']';
}

#else // JIT
#ifdef JIT_cpu
void CnnlConvBackwardWOp::jit_run() {
    int ni, ci, hi, wi, no, co, cw, ho, wo, kh, kw;
    get_shape(x, "abcd", xformat, ni, hi, wi, ci);
    get_shape(w, "oihw", wformat, co, kh, kw, cw);
    get_shape(out, "abcd", yformat, no, ho, wo, co);

    int input_dim[4] = {ni, hi, wi, ci};
    int weight_dim[4] = {co, kh, kw, cw};
    int output_dim[4] = {no, ho, wo, co};

    cnnlTensorDescriptor_t input_desc = nullptr;
    cnnlTensorDescriptor_t weight_desc = nullptr;
    cnnlTensorDescriptor_t grad_weight_desc = nullptr;
    cnnlTensorDescriptor_t grad_desc = nullptr;
    cnnlCreateTensorDescriptor(&input_desc);
    cnnlCreateTensorDescriptor(&weight_desc);
    cnnlCreateTensorDescriptor(&grad_weight_desc);
    cnnlCreateTensorDescriptor(&grad_desc);
    cnnlSetTensorDescriptor(input_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_INT8, 4, input_dim);
    cnnlSetTensorDescriptor(weight_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_INT8, 4, weight_dim);
    cnnlSetTensorDescriptor(grad_weight_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_FLOAT, 4, weight_dim);
    cnnlSetTensorDescriptor(grad_desc, CNNL_LAYOUT_NHWC, CNNL_DTYPE_INT8, 4, output_dim);

    cnnlConvolutionBwdFilterPreference_t pre_t = CNNL_CONVOLUTION_BWD_FILTER_FASTEST;
    cnnlConvolutionBwdFilterAlgo_t algo_t;
    cnnlConvolutionDescriptor_t conv_desc = nullptr;
    void *workspace = nullptr;
    size_t workspace_size = 0;
    cnnlHandle_t handle = nullptr;
    cnnlCreate(&handle);
    cnnlSetQueue(handle, mlu_queue);

    int pad[4] = {paddingh, paddingh, paddingw, paddingw};
    int stride[2] = {strideh, stridew};
    int dilation[2] = {dilationh, dilationw};
    cnnlCreateConvolutionDescriptor(&conv_desc);
    cnnlSetConvolutionDescriptor(conv_desc, 4, pad, stride, dilation, groups, CNNL_DTYPE_FLOAT);

    cnnlGetConvolutionBackwardFilterAlgorithm(handle, conv_desc, input_desc, grad_desc, grad_weight_desc, pre_t, &algo_t);

    cnnlGetConvolutionBackwardFilterWorkspaceSize(handle, input_desc, grad_desc, grad_weight_desc, conv_desc, algo_t, &workspace_size);

    if (workspace_size != 0) {
        cnrtMalloc(&workspace, workspace_size);
        cnrtMemset(workspace, 0, workspace_size);
    }

    const void * alpha = nullptr;
    const void * beta = nullptr;
    cnnlConvolutionBackwardFilter(
    /* handle         */ handle,
    /* alpha          */ alpha,
    /* weight_desc    */ input_desc,
    /* weight         */ x->mem_ptr,
    /* diff_y_desc    */ grad_desc,
    /* diff_y         */ dout->mem_ptr,
    /* conv_desc      */ conv_desc,
    /* algo           */ algo_t,
    /* workspace      */ workspace,
    /* workspace_size */ workspace_size,
    /* beta           */ beta,
    /* diff_x_desc    */ grad_weight_desc,
    /* diff_x         */ dw->mem_ptr);
}
#endif
#endif // JIT

} // jittor