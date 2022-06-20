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

struct CudnnRnnBackwardXOp : Op {
    Var* x, * hx, * cx;
    Var* y, * dy, * dhy, * dcy;
    Var* w;
    Var* dx, * dhx, * dcx, * dw;
    Var* reservation;
    string mode;
    int input_size, hidden_size, num_layers, proj_size, batch_size;
    int seq_length;
    float dropout;
    bool bias, bidirectional;
    
    // @attrs(multiple_outputs)
    CudnnRnnBackwardXOp(Var* x, Var* hx, Var* cx, Var* y, Var* dy, Var* dhy, Var* dcy, Var* w, Var* reservation, string mode, int input_size, int hidden_size, int num_layers, int proj_size, double dropout, bool bias, bool bidirectional);

    // @attrs(multiple_outputs)
    CudnnRnnBackwardXOp(Var* x, Var* hx, Var* y, Var* dy, Var* dhy, Var* w, Var* reservation, string mode, int input_size, int hidden_size, int num_layers, int proj_size, double dropout, bool bias, bool bidirectional);

    void init_rnn();

    const char* name() const override { return "cudnn_rnn_backward_x"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
