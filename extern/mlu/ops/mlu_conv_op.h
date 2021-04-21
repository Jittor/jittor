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
#include "op.h"
#include "./cnml.h"
#include "./cnrt.h"

namespace jittor {

struct MluConvOp : Op {
    Var* x, * w, * y;
    int strideh, stridew, paddingh, paddingw, dilationh, dilationw, groups;
    string xformat, wformat, yformat;
    /* MklConvOp: xformat abcd represents nchw */
    MluConvOp(Var* x, Var* w, int strideh, int stridew, int paddingh, int paddingw, int dilationh=1, int dilationw=1, int groups=1, string xformat="abcd", string wformat="oihw", string yformat="");
    
    const char* name() const override { return "mlu_conv"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

struct MluConv_t {
    cnmlBaseOp_t conv_op_cache;
    cnmlConvOpParam_t conv_param_cache;
    cnmlTensor_t input_tensor_cache;
    cnmlTensor_t filter_tensor_cache;
    cnmlTensor_t output_tensor_cache;
    int8_t* filter_cpu_ptr_cache;
};

} // jittor