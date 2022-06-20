// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: 
//      Zheng-Ning Liu <lzhengning@gmail.com>
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"
#include "cudnn_wrapper.h"
#include "executor.h"
#include "init.h"


namespace jittor {

static inline cudnnRNNMode_t rnn_string_to_rnn_mode(string mode) {
    if (mode == "relu")
        return CUDNN_RNN_RELU;
    if (mode == "tanh")
        return CUDNN_RNN_TANH;
    if (mode == "lstm")
        return CUDNN_LSTM;
    ASSERT(mode == "gru") << "rnn mode must be relu, tanh, lstm, or gru, but got " << mode;
    return CUDNN_GRU;
}

static inline int rnn_string_to_num_linear_layers(string mode) {
    if (mode == "relu")
        return 2;
    if (mode == "tanh")
        return 2;
    if (mode == "lstm")
        return 8;
    ASSERT(mode == "gru") << "mode must be relu, tanh, lstm, or gru, but got " << mode;
    return 6;
}

/** A wrapper for CUDNN dropout descriptor
 */
struct DropoutDescriptor {
    cudnnDropoutDescriptor_t desc;
    size_t stateSize, stateAllocation;
    float dropout;
    void *stateSpace;

    DropoutDescriptor(cudnnHandle_t handle, float dropout) 
        : dropout(dropout), stateSpace(nullptr) {
        checkCudaErrors(cudnnCreateDropoutDescriptor(&desc));
        if (dropout > 0) {
            checkCudaErrors(cudnnDropoutGetStatesSize(handle, &stateSize));
            stateSpace = exe.temp_allocator->alloc(stateSize, stateAllocation);
            checkCudaErrors(cudnnSetDropoutDescriptor(
                desc,
                cudnn_handle,
                dropout,
                stateSpace,
                stateSize,
                get_seed()
            ));
        } else {
            checkCudaErrors(cudnnSetDropoutDescriptor(
                desc, handle, 0, nullptr, 0, 0
            ));
        }
    }
    ~DropoutDescriptor() {
        checkCudaErrors(cudnnDestroyDropoutDescriptor(desc));
        if (stateSpace)
            exe.temp_allocator->free(stateSpace, stateSize, stateAllocation);
    }
};

/** A wrapper for CUDNN RNN descriptor
 */
struct RnnDescriptor {
    cudnnHandle_t handle;
    cudnnRNNDescriptor_t desc;
    DropoutDescriptor dropoutDesc;
    
    RnnDescriptor(cudnnHandle_t handle, string mode, int hidden_size, int num_layers, 
        float dropout, bool bidirectional) : handle(handle), dropoutDesc(handle, dropout) {
        checkCudaErrors(cudnnCreateRNNDescriptor(&desc));
        checkCudaErrors(cudnnSetRNNDescriptor_v6(
            handle,
            desc,
            hidden_size,
            num_layers,
            dropoutDesc.desc,
            CUDNN_LINEAR_INPUT,
            bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
            rnn_string_to_rnn_mode(mode),
            CUDNN_RNN_ALGO_STANDARD,
            CUDNN_DATA_FLOAT
        ));
    }

    ~RnnDescriptor() {
        checkCudaErrors(cudnnDestroyRNNDescriptor(desc));
    }

    size_t weight_space_size(const cudnnTensorDescriptor_t &xDesc) {
        size_t size;
        checkCudaErrors(cudnnGetRNNParamsSize(
            handle, desc, xDesc, &size, CUDNN_DATA_FLOAT
        ));
        return size;
    }

    size_t work_space_size(const cudnnTensorDescriptor_t *xDesc, int seq_length) {
        size_t size;
        checkCudaErrors(cudnnGetRNNWorkspaceSize(
            handle, desc, seq_length, xDesc, &size
        ));
        return size;
    }

    size_t reserve_space_size(const cudnnTensorDescriptor_t *xDesc, int seq_length) {
        size_t size;
        checkCudaErrors(cudnnGetRNNTrainingReserveSize(
            handle, desc, seq_length, xDesc, &size
        ));
        return size;
    }
};

/** 
 */
struct RnnWeightDescriptor {
    cudnnFilterDescriptor_t desc;
    size_t size;
    RnnWeightDescriptor(size_t size) : size(size) {
        int dimW[3] = {(int) (size / sizeof(float)), 1, 1};
        checkCudaErrors(cudnnCreateFilterDescriptor(&desc));
        checkCudaErrors(cudnnSetFilterNdDescriptor(desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));
    }
    ~RnnWeightDescriptor() {
        cudnnDestroyFilterDescriptor(desc);
    }
};

/** 
    Returns offsets of RNN linear parameters in a flatten array.

    Returns
    =======
        list: [total size, param #1 offset, param #2 offset, ...]

    TODO: support cudnn rnn-v8; support proj_size
 */
// @pyjt(cudnn_rnn_weight_offset)
vector<int32_t> cudnn_rnn_weight_offset(string mode, int input_size, int hidden_size, int num_layers, int proj_size, bool bias, bool bidirectional);

} // jittor