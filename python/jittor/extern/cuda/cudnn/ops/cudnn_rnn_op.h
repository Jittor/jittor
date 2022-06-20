// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: 
//      Zheng-Ning Liu <lzhengning@gmail.com>
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"

namespace jittor {

struct CudnnRnnOp : Op {
    Var* x, * hx, * cx, * y, * hy, * cy;
    Var* w;
    Var* reservation;
    string mode;
    int input_size, hidden_size, num_layers, proj_size;
    int seq_length, batch_size;
    float dropout;
    bool bias, bidirectional, is_train;

    // @attrs(multiple_outputs)
    CudnnRnnOp(Var* x, Var* hx, Var* cx, Var* w, string mode, int input_size, int hidden_size, int num_layers, int proj_size, double dropout, bool batch_first, bool bias, bool bidirectional);
    // @attrs(multiple_outputs)
    CudnnRnnOp(Var* x, Var* hx, Var* w, string mode, int input_size, int hidden_size, int num_layers, int proj_size, double dropout, bool batch_first, bool bias, bool bidirectional);

    void init_rnn();

    const char* name() const override { return "cudnn_rnn"; }
    void grads(Var** douts, VarPtr* dins) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
