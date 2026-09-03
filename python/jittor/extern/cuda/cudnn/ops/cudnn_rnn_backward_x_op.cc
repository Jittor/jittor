// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//      Zheng-Ning Liu <lzhengning@gmail.com>
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "cudnn_rnn_descriptor.h"
#include "cudnn_descriptor.h"
#include "cudnn_rnn_backward_x_op.h"
#include "cudnn_wrapper.h"
#include "executor.h"
#include "ops/op_register.h"

namespace jittor {

#pragma GCC diagnostic ignored "-Wunused-variable"

#ifndef JIT

CudnnRnnBackwardXOp::CudnnRnnBackwardXOp(Var *x, Var* hx, Var* cx, Var* y, Var* dy, Var* dhy, Var* dcy, Var* w, Var* reservation, 
    string mode, int input_size, int hidden_size, int num_layers, int proj_size, 
    double dropout, bool bias, bool bidirectional) 
    : x(x), hx(hx), cx(cx), y(y), dy(dy), dhy(dhy), dcy(dcy), w(w), reservation(reservation),
        mode(mode), input_size(input_size), hidden_size(hidden_size), num_layers(num_layers),
        proj_size(proj_size), dropout(dropout), bias(bias), bidirectional(bidirectional) {
    
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);

    ASSERTop(mode,==,"lstm");
    ASSERTop(proj_size,==,0);
    init_rnn();
}

CudnnRnnBackwardXOp::CudnnRnnBackwardXOp(Var* x, Var* hx, Var* y, Var* dy, Var* dhy, Var* w, Var* reservation, 
    string mode, int input_size, int hidden_size, int num_layers, int proj_size, 
    double dropout, bool bias, bool bidirectional) 
    : x(x), hx(hx), cx(nullptr), y(y), dy(dy), dhy(dhy), dcy(nullptr), w(w), reservation(reservation),
        mode(mode), input_size(input_size), hidden_size(hidden_size), num_layers(num_layers),
        proj_size(proj_size), dropout(dropout), bias(bias), bidirectional(bidirectional) {

    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);

    ASSERTop(mode,!=,"lstm");
    ASSERTop(proj_size,==,0);
    init_rnn();
}

void CudnnRnnBackwardXOp::init_rnn() {
    // Gradients carry the dtype of what they are gradients of. These were
    // hardcoded fp32 while jit_run writes them through ptr<Tx>(), so on a
    // half RNN cuDNN wrote half values into buffers laid out for floats.
    dx = create_output(nullptr, x->ns);
    dhx = create_output(nullptr, hx->ns);
    
    if (mode == "lstm")
        dcx = create_output(nullptr, cx->ns);
    else
        dcx = nullptr;

    dw = create_output(nullptr, dtype_infer(x->ns, y->ns));

    seq_length = y->shape[0];
    batch_size = y->shape[1];
}

void CudnnRnnBackwardXOp::infer_shape() {
    dx->set_shape(NanoVector(seq_length, batch_size, input_size));

    int num_directions = 1 + bidirectional;
    if (proj_size > 0)
        dhx->set_shape(NanoVector(num_layers * num_directions, batch_size, proj_size));
    else
        dhx->set_shape(NanoVector(num_layers * num_directions, batch_size, hidden_size));

    if (dcx)
        dcx->set_shape(NanoVector(num_layers * num_directions, batch_size, hidden_size));

    dw->set_shape(w->shape);
}

void CudnnRnnBackwardXOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << hx->dtype();
    jk << "«Ty:" << y->dtype();
    jk << "«Tw:" << w->dtype();
}

#else // JIT
#ifdef JIT_cuda

template <typename T_ELEM> __inline__  cudnnDataType_t getDataType();
// template <> __inline__ cudnnDataType_t getDataType<half1>() { return CUDNN_DATA_HALF;   }
// template <> __inline__ cudnnDataType_t getDataType<float>() { return CUDNN_DATA_FLOAT;  }

void CudnnRnnBackwardXOp::jit_run() {
    int num_directions = 1 + bidirectional;
    int hidden_dims[3] = {num_layers * num_directions, batch_size, hidden_size};
    int hidden_strides[3] = {hidden_dims[1] * hidden_dims[2], hidden_dims[2], 1};

    // Two sequence-data descriptors where v6 took four arrays of `seq_length`
    // tensor descriptors: x and dx share one, y and dy share the other,
    // because they have the same shape and layout and always did.
    CudnnRnnSeqLengths seq(batch_size, seq_length);
    CudnnRnnDataDescriptor xDesc, yDesc;
    xDesc.set(getDataType<Ty>(), seq_length, batch_size, input_size, seq.host_data());
    yDesc.set(getDataType<Ty>(), seq_length, batch_size,
        hidden_size * num_directions, seq.host_data());

    CudnnTensorDescriptor hDesc, cDesc;
    checkCudaErrors(cudnnSetTensorNdDescriptor(hDesc, getDataType<Tx>(), 3, hidden_dims, hidden_strides));
    checkCudaErrors(cudnnSetTensorNdDescriptor(cDesc, getDataType<Tx>(), 3, hidden_dims, hidden_strides));

    RnnDescriptor rnn_desc(cudnn_handle, mode, input_size, hidden_size,
        num_layers, dropout, bidirectional, getDataType<Tx>());

    // The backward reads back what the forward wrote, so it has to size its
    // spaces for the mode the forward ran in -- training, always, or there
    // would be no reserve to read.
    size_t work_space_size = 0, reserve_space_size = 0;
    rnn_desc.temp_space_sizes(CUDNN_FWD_MODE_TRAINING, xDesc,
        &work_space_size, &reserve_space_size);
    CudnnWorkspace work_space(work_space_size);

    checkCudaErrors(cudnnRNNBackwardData_v8(
        cudnn_handle, rnn_desc.desc,
        seq.dev(),
        yDesc, y->ptr<Ty>(), dy->ptr<Ty>(),
        xDesc, dx->ptr<Tx>(),
        hDesc, hx->ptr<Tx>(), dhy->ptr<Ty>(), dhx->ptr<Tx>(),
        cDesc,
        mode == "lstm" ? cx->ptr<Tx>() : nullptr,
        mode == "lstm" ? dcy->ptr<Ty>() : nullptr,
        mode == "lstm" ? dcx->ptr<Tx>() : nullptr,
        w->size, w->ptr<Tw>(),
        work_space.size, work_space.ptr,
        reservation->size, reservation->ptr<Tx>()
    ));

    // CUDNN_WGRAD_MODE_ADD accumulates into dw, so dw starts at zero. (v6's
    // cudnnRNNBackwardWeights had the same contract and the same memset.)
    checkCudaErrors(cudaMemset(dw->ptr<Tw>(), 0, dw->size));

    // Must follow BackwardData: they share the reserve space, and cuDNN
    // documents this order.
    checkCudaErrors(cudnnRNNBackwardWeights_v8(
        cudnn_handle, rnn_desc.desc,
        CUDNN_WGRAD_MODE_ADD,
        seq.dev(),
        xDesc, x->ptr<Tx>(),
        hDesc, hx->ptr<Tx>(),
        yDesc, y->ptr<Ty>(),
        dw->size, dw->ptr<Tw>(),
        work_space.size, work_space.ptr,
        reservation->size, reservation->ptr<Tx>()
    ));

}

#endif
#endif // JIT
}
