// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: 
//      Zheng-Ning Liu <lzhengning@gmail.com>
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "cudnn_rnn_descriptor.h"
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
    
    flags.set(NodeFlags::_cuda, 1);
    flags.set(NodeFlags::_cpu, 0);

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

    flags.set(NodeFlags::_cuda, 1);
    flags.set(NodeFlags::_cpu, 0);

    ASSERTop(mode,!=,"lstm");
    ASSERTop(proj_size,==,0);
    init_rnn();
}

void CudnnRnnBackwardXOp::init_rnn() {
    dx = create_output(nullptr, ns_float32);
    dhx = create_output(nullptr, ns_float32);
    
    if (mode == "lstm")
        dcx = create_output(nullptr, ns_float32);
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
template <> __inline__ cudnnDataType_t getDataType<half1>() { return CUDNN_DATA_HALF;   }
template <> __inline__ cudnnDataType_t getDataType<float>() { return CUDNN_DATA_FLOAT;  }

void CudnnRnnBackwardXOp::jit_run() {
    int num_directions = 1 + bidirectional;

    int in_dims[3] = {batch_size, input_size, 1};
    int out_dims[3] = {batch_size, hidden_size * num_directions, 1};
    int in_strides[3] = {in_dims[1] * in_dims[2], in_dims[2], 1};
    int out_strides[3] = {out_dims[1] * out_dims[2], out_dims[2], 1};
    int hidden_dims[3] = {num_layers * num_directions, batch_size, hidden_size};
    int hidden_strides[3] = {hidden_dims[1] * hidden_dims[2], hidden_dims[2], 1};

    vector<cudnnTensorDescriptor_t> xDesc(seq_length), dxDesc(seq_length);
    vector<cudnnTensorDescriptor_t> yDesc(seq_length), dyDesc(seq_length);

    for (int i = 0; i < seq_length; ++i) {
        checkCudaErrors(cudnnCreateTensorDescriptor(&xDesc[i]));
        checkCudaErrors(cudnnCreateTensorDescriptor(&dxDesc[i]));
        checkCudaErrors(cudnnCreateTensorDescriptor(&yDesc[i]));
        checkCudaErrors(cudnnCreateTensorDescriptor(&dyDesc[i]));
        checkCudaErrors(cudnnSetTensorNdDescriptor(xDesc[i], getDataType<Ty>(), 3, in_dims, in_strides));
        checkCudaErrors(cudnnSetTensorNdDescriptor(dxDesc[i], getDataType<Ty>(), 3, in_dims, in_strides));
        checkCudaErrors(cudnnSetTensorNdDescriptor(yDesc[i], getDataType<Ty>(), 3, out_dims, out_strides));
        checkCudaErrors(cudnnSetTensorNdDescriptor(dyDesc[i], getDataType<Ty>(), 3, out_dims, out_strides));
    }

    cudnnTensorDescriptor_t dhyDesc, dcyDesc;
    cudnnTensorDescriptor_t hxDesc, cxDesc, dhxDesc, dcxDesc;
    checkCudaErrors(cudnnCreateTensorDescriptor(&hxDesc));
    checkCudaErrors(cudnnCreateTensorDescriptor(&cxDesc));
    checkCudaErrors(cudnnCreateTensorDescriptor(&dhxDesc));
    checkCudaErrors(cudnnCreateTensorDescriptor(&dcxDesc));
    checkCudaErrors(cudnnCreateTensorDescriptor(&dhyDesc));
    checkCudaErrors(cudnnCreateTensorDescriptor(&dcyDesc));
    checkCudaErrors(cudnnSetTensorNdDescriptor(hxDesc, getDataType<Tx>(), 3, hidden_dims, hidden_strides));
    checkCudaErrors(cudnnSetTensorNdDescriptor(cxDesc, getDataType<Tx>(), 3, hidden_dims, hidden_strides));
    checkCudaErrors(cudnnSetTensorNdDescriptor(dhxDesc, getDataType<Tx>(), 3, hidden_dims, hidden_strides));
    checkCudaErrors(cudnnSetTensorNdDescriptor(dcxDesc, getDataType<Tx>(), 3, hidden_dims, hidden_strides));
    checkCudaErrors(cudnnSetTensorNdDescriptor(dhyDesc, getDataType<Tx>(), 3, hidden_dims, hidden_strides));
    checkCudaErrors(cudnnSetTensorNdDescriptor(dcyDesc, getDataType<Tx>(), 3, hidden_dims, hidden_strides));

    RnnWeightDescriptor w_desc(w->size);
    RnnDescriptor rnn_desc(cudnn_handle, mode, hidden_size, num_layers, dropout, bidirectional);

    void *work_space;
    size_t work_space_size = rnn_desc.work_space_size(dxDesc.data(), seq_length);
    size_t work_space_allocation;
    if (work_space_size > 0)
        work_space = exe.temp_allocator->alloc(work_space_size, work_space_allocation);

    size_t reserveSpaceSize = reservation->size;

    checkCudaErrors(cudnnRNNBackwardData(
        cudnn_handle, rnn_desc.desc,
        seq_length,
        yDesc.data(), y->ptr<Ty>(),
        dyDesc.data(), dy->ptr<Ty>(),
        dhyDesc, dhy->ptr<Ty>(),
        dcyDesc, mode == "lstm" ? dcy->ptr<Ty>(): nullptr,
        w_desc.desc, w->ptr<Tw>(),
        hxDesc, hx->ptr<Tx>(),
        cxDesc, mode == "lstm" ? cx->ptr<Tx>() : nullptr,
        dxDesc.data(), dx->ptr<Tx>(),
        dhxDesc, dhx->ptr<Tx>(),
        dcxDesc, mode == "lstm" ? dcx->ptr<Tx>() : nullptr,
        work_space, work_space_size,
        reservation->ptr<Tx>(), reservation->size
    ));

    checkCudaErrors(cudaMemset(dw->ptr<Tw>(), 0, dw->size));

    checkCudaErrors(cudnnRNNBackwardWeights(
        cudnn_handle, rnn_desc.desc,
        seq_length,
        xDesc.data(), x->ptr<Tx>(),
        hxDesc, hx->ptr<Tx>(),
        yDesc.data(), y->ptr<Ty>(),
        work_space, work_space_size,
        w_desc.desc, dw->ptr<Tw>(),
        reservation->ptr<Tx>(), reservation->size
    ));

    for (int i = 0; i < seq_length; ++i) {
        checkCudaErrors(cudnnDestroyTensorDescriptor(xDesc[i]));
        checkCudaErrors(cudnnDestroyTensorDescriptor(dxDesc[i]));
        checkCudaErrors(cudnnDestroyTensorDescriptor(yDesc[i]));
        checkCudaErrors(cudnnDestroyTensorDescriptor(dyDesc[i]));
    }

    checkCudaErrors(cudnnDestroyTensorDescriptor(dhyDesc));
    checkCudaErrors(cudnnDestroyTensorDescriptor(dcyDesc));
    checkCudaErrors(cudnnDestroyTensorDescriptor(hxDesc));
    checkCudaErrors(cudnnDestroyTensorDescriptor(cxDesc));
    checkCudaErrors(cudnnDestroyTensorDescriptor(dhxDesc));
    checkCudaErrors(cudnnDestroyTensorDescriptor(dcxDesc));

    if (work_space)
        exe.temp_allocator->free(work_space, work_space_size, work_space_allocation);
}

#endif
#endif // JIT
}
