// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Wenyang Zhou <576825820@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cnnl.h>
#include <cnml.h>
#include <cnrt.h>
#include "op.h"

namespace jittor {

struct CnnlMluConvOp : Op {
    Var* x, * w, * y;
    int strideh, stridew, paddingh, paddingw, dilationh, dilationw, groups;
    string xformat, wformat, yformat;
    
    CnnlMluConvOp(Var* x, Var* w, int strideh, int stridew, int paddingh, int paddingw, int dilationh=1, int dilationw=1, int groups=1, string xformat="abcd", string wformat="oihw", string yformat="");

    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    
    const char* name() const override { return "cnnl_mlu_conv"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

struct CnnlMluConv_t {
    cnnlConvolutionDescriptor_t conv_desc_cache;
    cnnlTensorDescriptor_t input_desc_cache;
    cnnlTensorDescriptor_t weight_desc_cache;
    cnnlTensorDescriptor_t output_desc_cache;
};

} // jittor