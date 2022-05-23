// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: 
//      Zheng-Ning Liu <lzhengning@gmail.com>
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "cudnn_rnn_descriptor.h"
#include "cudnn_rnn_op.h"
#include "cudnn_wrapper.h"
#include "executor.h"
#include "ops/op_register.h"

using namespace std;

namespace jittor {

#pragma GCC diagnostic ignored "-Wunused-variable"

#ifndef JIT

CudnnRnnOp::CudnnRnnOp(Var* x, Var* hx, Var* cx, Var* w, 
        string mode, int input_size, int hidden_size, int num_layers, int proj_size, 
        double dropout, bool bias, bool bidirectional, bool is_train)
    : x(x), hx(hx), cx(cx), w(w), mode(mode), input_size(input_size), hidden_size(hidden_size), 
        num_layers(num_layers), proj_size(proj_size), dropout(dropout), bias(bias), 
        bidirectional(bidirectional), is_train(is_train) {
    flags.set(NodeFlags::_cuda, 1);
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_grads, 1);

    ASSERTop(mode,==,"lstm");
    ASSERTop(proj_size,==,0);
    init_rnn();
}

CudnnRnnOp::CudnnRnnOp(Var* x, Var* hx, Var* w, 
        string mode, int input_size, int hidden_size, int num_layers, int proj_size, 
        double dropout, bool bias, bool bidirectional, bool is_train)
    : x(x), hx(hx), cx(nullptr), w(w), mode(mode), input_size(input_size), hidden_size(hidden_size), 
        num_layers(num_layers), proj_size(proj_size), dropout(dropout), bias(bias), 
        bidirectional(bidirectional), is_train(is_train) {
    flags.set(NodeFlags::_cuda, 1);
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_grads, 1);

    ASSERTop(mode,!=,"lstm");
    ASSERTop(proj_size,==,0);
    init_rnn();
}

void CudnnRnnOp::init_rnn() {
    y = create_output(nullptr, dtype_infer(x->ns, w->ns));
    hy = create_output(nullptr, dtype_infer(x->ns, w->ns));
    if (mode == "lstm")
        cy = create_output(nullptr, dtype_infer(x->ns, w->ns));
    else
        cy = nullptr;

    if (is_train)
        reservation = create_output(nullptr, ns_float32);
    else
        reservation = nullptr;

    seq_length = x->shape[0];
    batch_size = x->shape[1];
}

void CudnnRnnOp::infer_shape() {
    ASSERTop(x->shape.size(),==,3);
    ASSERTop(x->shape[2],==,input_size);
    
    int num_directions = 1 + bidirectional;

    y->set_shape(NanoVector(seq_length, batch_size, hidden_size * num_directions));

    if (proj_size > 0)
        hy->set_shape(NanoVector(num_layers * num_directions, batch_size, proj_size));
    else
        hy->set_shape(NanoVector(num_layers * num_directions, batch_size, hidden_size));

    if (cy)
        cy->set_shape(NanoVector(num_layers * num_directions, batch_size, hidden_size));

    if (reservation) {
        int in_dims[3] = {batch_size, input_size, 1};
        int in_strides[3] = {in_dims[1] * in_dims[2], in_dims[2], 1};

        vector<cudnnTensorDescriptor_t> xDesc(seq_length);
        RnnDescriptor rnn_desc(cudnn_handle, mode, hidden_size, num_layers, dropout, bidirectional);
        for (int i = 0; i < seq_length; ++i) {
            checkCudaErrors(cudnnCreateTensorDescriptor(&xDesc[i]));
            checkCudaErrors(cudnnSetTensorNdDescriptor(xDesc[i], CUDNN_DATA_FLOAT, 3, in_dims, in_strides));
        }
        reservation->set_shape(rnn_desc.reserve_space_size(xDesc.data(), seq_length));
    }
}

void CudnnRnnOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
    jk << "«Ty:" << y->dtype();
    jk << "«Tw:" << w->dtype();
}

static auto make_backwardx_with_cx = get_op_info("cudnn_rnn_backward_x")
    .get_constructor<vector<VarPtr>, Var*, Var*, Var*, Var*, Var*, Var*, Var*, Var*, Var*, string, int, int, int, int, double, bool, bool>();
static auto make_backwardx_without_cx = get_op_info("cudnn_rnn_backward_x")
    .get_constructor<vector<VarPtr>, Var*, Var*, Var*, Var*, Var*, Var*, Var*, string, int, int, int, int, double, bool, bool>();
static auto make_number = get_op_info("number")
    .get_constructor<VarPtr, float, Var*>();

void CudnnRnnOp::grads(Var** dout, VarPtr* dins) {
    VarPtr dy = dout[0];
    VarPtr dhy = dout[1];
    VarPtr dcy = cx ? dout[2] : nullptr;
    if (!dy.ptr) dy = make_number(0.0, y);
    if (!dhy.ptr) dhy = make_number(0.0, hy);
    if (!dcy.ptr && cx) dcy = make_number(0.0, cy);


    vector<VarPtr> dInput;
    if (cx)
        dInput = make_backwardx_with_cx(x, hx, cx, y, dy, dhy, dcy, w, reservation, mode, input_size, hidden_size, num_layers, proj_size, dropout, bias, bidirectional);
    else
        dInput = make_backwardx_without_cx(x, hx, y, dy, dhy, w, reservation, mode, input_size, hidden_size, num_layers, proj_size, dropout, bias, bidirectional);

    for (int i = 0; i < 3 + (cx != nullptr); ++i)
        dins[i] = move(dInput[i]);
}

#else // JIT
#ifdef JIT_cuda

#pragma clang diagnostic ignored "-Wtautological-compare"

template <typename T_ELEM> __inline__  cudnnDataType_t getDataType();
template <> __inline__ cudnnDataType_t getDataType<half1>() { return CUDNN_DATA_HALF;   }
template <> __inline__ cudnnDataType_t getDataType<float>() { return CUDNN_DATA_FLOAT;  }

void CudnnRnnOp::jit_run() {
    int num_directions = bidirectional + 1;
    int num_linear_layers = rnn_string_to_num_linear_layers(mode);

    int in_dims[3] = {batch_size, input_size, 1};
    int out_dims[3] = {batch_size, hidden_size * num_directions, 1};
    int in_strides[3] = {in_dims[1] * in_dims[2], in_dims[2], 1};
    int out_strides[3] = {out_dims[1] * out_dims[2], out_dims[2], 1};
    int hidden_dims[3] = {num_layers * num_directions, batch_size, hidden_size};
    int hidden_strides[3] = {hidden_dims[1] * hidden_dims[2], hidden_dims[2], 1};

    vector<cudnnTensorDescriptor_t> xDesc(seq_length);
    vector<cudnnTensorDescriptor_t> yDesc(seq_length);
    cudnnTensorDescriptor_t hxDesc, cxDesc;
    cudnnTensorDescriptor_t hyDesc, cyDesc;

    for (int i = 0; i < seq_length; ++i) {
        checkCudaErrors(cudnnCreateTensorDescriptor(&xDesc[i]));
        checkCudaErrors(cudnnCreateTensorDescriptor(&yDesc[i]));
        checkCudaErrors(cudnnSetTensorNdDescriptor(xDesc[i], getDataType<Tx>(), 3, in_dims, in_strides));
        checkCudaErrors(cudnnSetTensorNdDescriptor(yDesc[i], getDataType<Ty>(), 3, out_dims, out_strides));
    }

    checkCudaErrors(cudnnCreateTensorDescriptor(&hxDesc));
    checkCudaErrors(cudnnCreateTensorDescriptor(&cxDesc));
    checkCudaErrors(cudnnCreateTensorDescriptor(&hyDesc));
    checkCudaErrors(cudnnCreateTensorDescriptor(&cyDesc));

    checkCudaErrors(cudnnSetTensorNdDescriptor(hxDesc, getDataType<Tx>(), 3, hidden_dims, hidden_strides));
    checkCudaErrors(cudnnSetTensorNdDescriptor(cxDesc, getDataType<Tx>(), 3, hidden_dims, hidden_strides));

    checkCudaErrors(cudnnSetTensorNdDescriptor(hyDesc, getDataType<Tx>(), 3, hidden_dims, hidden_strides));
    checkCudaErrors(cudnnSetTensorNdDescriptor(cyDesc, getDataType<Tx>(), 3, hidden_dims, hidden_strides));

    RnnDescriptor rnn_desc(cudnn_handle, mode, hidden_size, num_layers, dropout, bidirectional);

    void *work_space;
    size_t work_space_size = rnn_desc.work_space_size(xDesc.data(), seq_length);
    size_t work_space_allocation;
    if (work_space_size > 0)
        work_space = exe.temp_allocator->alloc(work_space_size, work_space_allocation);

    RnnWeightDescriptor w_desc(w->size);

    if (is_train) {
        checkCudaErrors(cudnnRNNForwardTraining(
            cudnn_handle, rnn_desc.desc,
            seq_length,
            xDesc.data(), x->ptr<Tx>(),
            hxDesc, hx->ptr<Tx>(),
            cxDesc, mode == "lstm" ? cx->ptr<Tx>() : nullptr,
            w_desc.desc, w->ptr<Tw>(),
            yDesc.data(), y->ptr<Ty>(),
            hyDesc, hy->ptr<Ty>(),
            cyDesc, mode == "lstm" ? cy->ptr<Ty>() : nullptr,
            work_space, work_space_size,
            reservation->ptr<Tx>(), reservation->size
        ));
    } else {
        checkCudaErrors(cudnnRNNForwardInference(
            cudnn_handle, rnn_desc.desc,
            seq_length,
            xDesc.data(), x->ptr<Tx>(),
            hxDesc, hx->ptr<Tx>(),
            cxDesc, mode == "lstm" ? cx->ptr<Tx>() : nullptr,
            w_desc.desc, w->ptr<Tw>(),
            yDesc.data(), y->ptr<Ty>(),
            hyDesc, hy->ptr<Ty>(),
            cyDesc, mode == "lstm" ? cy->ptr<Ty>() : nullptr,
            work_space, work_space_size
        ));
    }

    for (int i = 0; i < seq_length; i++) {
        checkCudaErrors(cudnnDestroyTensorDescriptor(xDesc[i]));
        checkCudaErrors(cudnnDestroyTensorDescriptor(yDesc[i]));
    }

    checkCudaErrors(cudnnDestroyTensorDescriptor(hxDesc));
    checkCudaErrors(cudnnDestroyTensorDescriptor(cxDesc));
    checkCudaErrors(cudnnDestroyTensorDescriptor(hyDesc));
    checkCudaErrors(cudnnDestroyTensorDescriptor(cyDesc));

    if (work_space)
        exe.temp_allocator->free(work_space, work_space_size, work_space_allocation);
}

#endif
#endif // JIT

} // jittor

