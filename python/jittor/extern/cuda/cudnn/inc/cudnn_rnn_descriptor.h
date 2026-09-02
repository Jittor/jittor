// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//      Zheng-Ning Liu <lzhengning@gmail.com>
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "op.h"
#include "cudnn_wrapper.h"
#include "cudnn_descriptor.h"
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

/** cuDNN's RNN data type for a jittor dtype.

    cudnnSetRNNDescriptor_v6 takes CUDNN_DATA_FLOAT, _DOUBLE or _HALF and
    nothing else -- bfloat16 has no v6 RNN at any cuDNN version -- so refuse
    here, naming the dtype, rather than letting the descriptor come back
    CUDNN_STATUS_NOT_SUPPORTED with no operand named. Every RNN path (the op,
    shape inference, the weight-offset query) goes through this. */
static inline cudnnDataType_t cudnn_rnn_dtype(NanoString dtype) {
    ASSERT(dtype == ns_float32 || dtype == ns_float64 || dtype == ns_float16)
        << "cudnn rnn supports float16, float32 and float64, got" << dtype;
    return cudnn_dtype(dtype);
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
 *
 * The state buffer is cuDNN's RNG state for dropout, and cuDNN advances it in
 * place on every call that uses it.  So this object has to outlive the call:
 * built per call -- which is what a member of a `jit_run` local amounts to --
 * it was re-seeded from the global seed every single time, and every step of
 * training drew the *same* dropout mask sequence.  Nothing about that is
 * visible from Python; the loss curve just quietly belongs to a model that was
 * regularized far less than it asked for.  Get one of these from
 * `cudnn_rnn_dropout_descriptor()`, which keeps them across calls.
 */
struct DropoutDescriptor {
    cudnnDropoutDescriptor_t desc;
    size_t stateSize, stateAllocation;
    float dropout;
    void *stateSpace;
    // Freed through the allocator that served it: this outlives the call, and
    // `exe.allocator` is not necessarily the same object by then.
    Allocator *stateAllocator;

    DropoutDescriptor(cudnnHandle_t handle, float dropout, int seed)
        : dropout(dropout), stateSpace(nullptr), stateAllocator(nullptr) {
        checkCudaErrors(cudnnCreateDropoutDescriptor(&desc));
        if (dropout > 0) {
            checkCudaErrors(cudnnDropoutGetStatesSize(handle, &stateSize));
            stateAllocator = exe.allocator;
            stateSpace = stateAllocator->alloc(stateSize, stateAllocation);
            checkCudaErrors(cudnnSetDropoutDescriptor(
                desc,
                handle,
                dropout,
                stateSpace,
                stateSize,
                seed
            ));
        } else {
            checkCudaErrors(cudnnSetDropoutDescriptor(
                desc, handle, 0, nullptr, 0, 0
            ));
        }
    }
    ~DropoutDescriptor() {
        // Destructors are noexcept: checkCudaErrors here (it LOGf's, i.e.
        // throws) turned any failing descriptor destroy -- including one
        // during the unwinding of an earlier cuDNN error -- into terminate.
        peekCudaErrorsAlways(cudnnDestroyDropoutDescriptor(desc));
        if (stateSpace)
            stateAllocator->free(stateSpace, stateSize, stateAllocation);
    }

    DropoutDescriptor(const DropoutDescriptor&) = delete;
    DropoutDescriptor& operator=(const DropoutDescriptor&) = delete;
};

/** The dropout state for `dropout`, created on first use and reused after
    that, so consecutive RNN calls continue the mask sequence instead of
    restarting it.  `jt.set_seed()` drops the cache, so seeding still
    reproduces a run exactly. */
cudnnDropoutDescriptor_t cudnn_rnn_dropout_descriptor(cudnnHandle_t handle, float dropout);

/** A wrapper for CUDNN RNN descriptor
 */
struct RnnDescriptor {
    cudnnHandle_t handle;
    cudnnRNNDescriptor_t desc;
    // The RNN's own data type. It used to be hardcoded CUDNN_DATA_FLOAT while
    // the tensor descriptors around it were built from getDataType<Tx>(), so a
    // half or bfloat16 RNN described its tensors as one type and itself as
    // another -- cuDNN rejected the call with BAD_PARAM -- and the weight
    // space was sized as if the weights were fp32.
    cudnnDataType_t dataType;

    RnnDescriptor(cudnnHandle_t handle, string mode, int hidden_size, int num_layers, 
        float dropout, bool bidirectional, cudnnDataType_t dataType)
        : handle(handle), dataType(dataType) {
        checkCudaErrors(cudnnCreateRNNDescriptor(&desc));
        checkCudaErrors(cudnnSetRNNDescriptor_v6(
            handle,
            desc,
            hidden_size,
            num_layers,
            cudnn_rnn_dropout_descriptor(handle, dropout),
            CUDNN_LINEAR_INPUT,
            bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
            rnn_string_to_rnn_mode(mode),
            CUDNN_RNN_ALGO_STANDARD,
            dataType
        ));
        if (dataType != CUDNN_DATA_FLOAT && dataType != CUDNN_DATA_DOUBLE)
            // A half/bfloat16 RNN accumulates in fp32 unless the math type
            // says otherwise; without this cuDNN refuses the reduced-precision
            // descriptor on several architectures.
            checkCudaErrors(cudnnSetRNNMatrixMathType(desc, CUDNN_TENSOR_OP_MATH));
    }

    RnnDescriptor(const RnnDescriptor&) = delete;
    RnnDescriptor& operator=(const RnnDescriptor&) = delete;

    ~RnnDescriptor() {
        // Same reason as ~DropoutDescriptor: report, never raise.
        peekCudaErrorsAlways(cudnnDestroyRNNDescriptor(desc));
    }

    size_t weight_space_size(const cudnnTensorDescriptor_t &xDesc) {
        size_t size;
        checkCudaErrors(cudnnGetRNNParamsSize(
            handle, desc, xDesc, &size, dataType
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
    // Ownership comes from CudnnFilterDescriptor; this type only adds the
    // shape. `desc` stays a public member so call sites read unchanged.
    CudnnFilterDescriptor filter;
    cudnnFilterDescriptor_t desc;
    size_t size;
    // `size` is in bytes; the filter's extent is in elements, so it depends on
    // the weight dtype. Both were fixed at fp32, which described a half weight
    // buffer as twice as many fp32 values as it holds.
    RnnWeightDescriptor(size_t size, cudnnDataType_t dataType, int elem_size)
        : desc(filter.desc), size(size) {
        int dimW[3] = {(int) (size / elem_size), 1, 1};
        checkCudaErrors(cudnnSetFilterNdDescriptor(desc, dataType, CUDNN_TENSOR_NCHW, 3, dimW));
    }

    RnnWeightDescriptor(const RnnWeightDescriptor&) = delete;
    RnnWeightDescriptor& operator=(const RnnWeightDescriptor&) = delete;
};

/** Training reserve-space size for an RNN of this shape, cached per
    configuration. Shape inference needs the number and cuDNN will only give it
    up through descriptors, so the call lives here where it happens once rather
    than in infer_shape where it happened -- and leaked seq_length descriptors
    -- on every step. */
size_t cudnn_rnn_reserve_space_size(string mode, int input_size, int hidden_size,
    int num_layers, float dropout, bool bidirectional,
    int seq_length, int batch_size, cudnnDataType_t dtype);

/** Drops the cached dropout states, so the next RNN call rebuilds them from
    the current seed. Registered as a set_seed callback. */
void cudnn_rnn_dropout_reset();

/** 
    Returns offsets of RNN linear parameters in a flatten array.

    Returns
    =======
        list: [total size, param #1 offset, param #2 offset, ...]

    TODO: support cudnn rnn-v8; support proj_size
 */
// @pyjt(cudnn_rnn_weight_offset)
vector<int32_t> cudnn_rnn_weight_offset(string mode, int input_size, int hidden_size, int num_layers, int proj_size, bool bias, bool bidirectional, string dtype="float32");

} // jittor