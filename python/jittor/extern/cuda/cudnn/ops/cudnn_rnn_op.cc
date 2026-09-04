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
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_grads, 1);

    USER_CHECKop(mode,==,"lstm");
    ASSERTop(proj_size,==,0);
    init_rnn();
}

CudnnRnnOp::CudnnRnnOp(Var* x, Var* hx, Var* w, 
        string mode, int input_size, int hidden_size, int num_layers, int proj_size, 
        double dropout, bool bias, bool bidirectional, bool is_train)
    : x(x), hx(hx), cx(nullptr), w(w), mode(mode), input_size(input_size), hidden_size(hidden_size), 
        num_layers(num_layers), proj_size(proj_size), dropout(dropout), bias(bias), 
        bidirectional(bidirectional), is_train(is_train) {
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    set_flag(OpFlags::_grads, 1);

    USER_CHECKop(mode,!=,"lstm");
    ASSERTop(proj_size,==,0);
    init_rnn();
}

void CudnnRnnOp::init_rnn() {
    // cuDNN describes the whole RNN with a single data type, so x, w and y
    // have to agree. Saying so here turns a mixed-dtype call into a readable
    // error at construction instead of CUDNN_STATUS_BAD_PARAM from inside
    // cudnnRNNForward*, which names neither operand.
    USER_CHECK(x->ns == w->ns) << "cudnn_rnn needs input and weight of the same dtype,"
        << "got" << x->ns << "and" << w->ns;
    cudnn_rnn_dtype(x->ns);   // refuse an unsupported dtype here, by name

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
        #ifdef IS_CUDA
        // Was: seq_length cudnnCreateTensorDescriptor here and no Destroy
        // anywhere, i.e. seq_length leaked descriptors per shape inference,
        // which for a training RNN is per step. The query is cached per
        // configuration now and cleans up after itself; shape inference still
        // reaches cuDNN, because the reserve-space size is cuDNN's to know.
        reservation->set_shape(cudnn_rnn_reserve_space_size(
            mode, input_size, hidden_size, num_layers, dropout, bidirectional,
            seq_length, batch_size, cudnn_rnn_dtype(x->dtype())));
        #endif
    }
}

void CudnnRnnOp::jit_prepare(JK& jk) {
    jk << "«Tx:" << x->dtype();
    jk << "«Ty:" << y->dtype();
    jk << "«Tw:" << w->dtype();
}

static auto make_backwardx_with_cx = op_constructor<vector<VarPtr>, Var*, Var*, Var*, Var*, Var*, Var*, Var*, Var*, Var*, string, int, int, int, int, double, bool, bool>("cudnn_rnn_backward_x");
static auto make_backwardx_without_cx = op_constructor<vector<VarPtr>, Var*, Var*, Var*, Var*, Var*, Var*, Var*, string, int, int, int, int, double, bool, bool>("cudnn_rnn_backward_x");
static auto make_number = op_constructor<VarPtr, float, Var*>("number");

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
// template <> __inline__ cudnnDataType_t getDataType<half1>() { return CUDNN_DATA_HALF;   }
// template <> __inline__ cudnnDataType_t getDataType<float>() { return CUDNN_DATA_FLOAT;  }

void CudnnRnnOp::jit_run() {
    cudnn_bind_stream();
    int num_directions = bidirectional + 1;
    int hidden_dims[3] = {num_layers * num_directions, batch_size, hidden_size};
    int hidden_strides[3] = {hidden_dims[1] * hidden_dims[2], hidden_dims[2], 1};

    // One sequence-data descriptor each for x and y, where the v6 API took an
    // array of `seq_length` tensor descriptors per operand -- 2*seq_length
    // objects created and destroyed per call, all describing the same shape.
    CudnnRnnSeqLengths seq(batch_size, seq_length);
    CudnnRnnDataDescriptor xDesc, yDesc;
    xDesc.set(getDataType<Tx>(), seq_length, batch_size, input_size, seq.host_data());
    yDesc.set(getDataType<Ty>(), seq_length, batch_size,
        hidden_size * num_directions, seq.host_data());

    // v8 describes h and c with one tensor descriptor each, not one per end:
    // the same descriptor covers hx and hy.
    CudnnTensorDescriptor hDesc, cDesc;
    checkCudaErrors(cudnnSetTensorNdDescriptor(hDesc, getDataType<Tx>(), 3, hidden_dims, hidden_strides));
    checkCudaErrors(cudnnSetTensorNdDescriptor(cDesc, getDataType<Tx>(), 3, hidden_dims, hidden_strides));

    RnnDescriptor rnn_desc(cudnn_handle, mode, input_size, hidden_size,
        num_layers, dropout, bidirectional, getDataType<Tx>());

    cudnnForwardMode_t fwd_mode = is_train
        ? CUDNN_FWD_MODE_TRAINING : CUDNN_FWD_MODE_INFERENCE;
    size_t work_space_size = 0, reserve_space_size = 0;
    rnn_desc.temp_space_sizes(fwd_mode, xDesc, &work_space_size, &reserve_space_size);
    CudnnWorkspace work_space(work_space_size);

    // Inference asks for no reserve at all, and cuDNN wants the pair to agree:
    // a size of zero must come with a null pointer. `reservation` is still
    // allocated (shape inference sizes it for training either way), it just is
    // not handed over.
    void *reserve = is_train ? (void *)reservation->ptr<Tx>() : nullptr;
    size_t reserve_size = is_train ? reservation->size : 0;

    checkCudaErrors(cudnnRNNForward(
        cudnn_handle, rnn_desc.desc,
        fwd_mode,
        seq.dev(),
        xDesc, x->ptr<Tx>(),
        yDesc, y->ptr<Ty>(),
        hDesc, hx->ptr<Tx>(), hy->ptr<Ty>(),
        cDesc,
        mode == "lstm" ? cx->ptr<Tx>() : nullptr,
        mode == "lstm" ? cy->ptr<Ty>() : nullptr,
        w->size, w->ptr<Tw>(),
        work_space.size, work_space.ptr,
        reserve_size, reserve
    ));

}

#endif
#endif // JIT

} // jittor
