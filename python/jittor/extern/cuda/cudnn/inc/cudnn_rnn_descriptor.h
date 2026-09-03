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
    USER_CHECK(dtype == ns_float32 || dtype == ns_float64 || dtype == ns_float16)
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

/** Math type for an RNN of this data type.

    float32 follows `float32_matmul_precision` like every other cuDNN op
    (misc/float32_precision.h). It used to follow nothing: the descriptor set
    a math type only for reduced precision, so an fp32 RNN got cuDNN's default,
    and cuDNN's default for RNN on Ampere and later *allows tf32*. An fp32
    LSTM therefore ran at tf32 whatever `cuda_allow_cudnn_tf32` said, and no
    setting could turn it off.

    It is worth being precise about the size of that: against a float64
    reference on a 5-step LSTM, jittor's own CPU recurrence lands 1.2e-07 out
    and the cuDNN path landed 2.3e-04 out -- three orders of magnitude, on the
    weight gradients, silently, at the default setting. With CUDNN_FMA_MATH the
    cuDNN path lands 1.1e-07 out, i.e. on top of the CPU one.

    float16 keeps CUDNN_TENSOR_OP_MATH unconditionally: reduced-precision
    operands have nothing to lose to a tensor core, and cuDNN refuses the
    half descriptor without it on several architectures.
 */
static inline cudnnMathType_t rnn_math_type(cudnnDataType_t dataType) {
    if (dataType == CUDNN_DATA_DOUBLE)
        return CUDNN_DEFAULT_MATH;
    if (dataType != CUDNN_DATA_FLOAT)
        return CUDNN_TENSOR_OP_MATH;
    if (float32_cudnn_tier() != F32_HIGHEST)
        return CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
#if CUDNN_VERSION >= 8000
    return CUDNN_FMA_MATH;
#else
    return CUDNN_DEFAULT_MATH;
#endif
}

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

    RnnDescriptor(cudnnHandle_t handle, string mode, int input_size, int hidden_size,
        int num_layers, float dropout, bool bidirectional, cudnnDataType_t dataType)
        : handle(handle), dataType(dataType) {
        checkCudaErrors(cudnnCreateRNNDescriptor(&desc));
        checkCudaErrors(cudnnSetRNNDescriptor_v8(
            desc,
            CUDNN_RNN_ALGO_STANDARD,
            rnn_string_to_rnn_mode(mode),
            // The v6 descriptor had no bias mode and always laid out two bias
            // vectors per linear layer. The flat weight jittor builds from
            // cudnn_rnn_weight_offset() is that layout, so saying anything
            // else here would silently move every weight.
            CUDNN_RNN_DOUBLE_BIAS,
            bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
            CUDNN_LINEAR_INPUT,
            dataType,
            // v6 took one type and used it as both; keep that.
            dataType,
            // v6 needed a second call (cudnnSetRNNMatrixMathType, gone in
            // cuDNN 9); v8 takes it here.
            rnn_math_type(dataType),
            input_size,
            hidden_size,
            // projSize == hiddenSize is "no projection". jittor asserts
            // proj_size == 0 at construction, so there is never one.
            hidden_size,
            num_layers,
            cudnn_rnn_dropout_descriptor(handle, dropout),
            // Every sequence in the batch is the full length; nothing is
            // padded, so cuDNN need not look for padding.
            CUDNN_RNN_PADDED_IO_DISABLED
        ));
    }

    RnnDescriptor(const RnnDescriptor&) = delete;
    RnnDescriptor& operator=(const RnnDescriptor&) = delete;

    ~RnnDescriptor() {
        // Same reason as ~DropoutDescriptor: report, never raise.
        peekCudaErrorsAlways(cudnnDestroyRNNDescriptor(desc));
    }

    /// Bytes of weight space. v6 needed an x descriptor to answer this; v8
    /// knows the shape from the descriptor itself.
    size_t weight_space_size() {
        size_t size;
        checkCudaErrors(cudnnGetRNNWeightSpaceSize(handle, desc, &size));
        return size;
    }

    /** Workspace and reserve space, which v8 reports together.

        The reserve is what the backward pass reads back, and it is non-zero
        only for CUDNN_FWD_MODE_TRAINING -- v6 had a separate entry point per
        question and a separate one per mode, so the two could disagree about
        which mode they were sizing for. */
    void temp_space_sizes(cudnnForwardMode_t fwd_mode,
            cudnnRNNDataDescriptor_t xDesc, size_t *work, size_t *reserve) {
        checkCudaErrors(cudnnGetRNNTempSpaceSizes(
            handle, desc, fwd_mode, xDesc, work, reserve));
    }
};

/** cuDNN's v8 sequence-data descriptor.

    It replaces the array of `seq_length` tensor descriptors that every RNN
    call used to build and destroy -- 2*seq_length of them in the forward,
    4*seq_length in the backward, for a batch whose sequences are all the same
    length and whose layout never varied.
 */
struct CudnnRnnDataDescriptor {
    cudnnRNNDataDescriptor_t desc = nullptr;

    CudnnRnnDataDescriptor() {
        checkCudaErrors(cudnnCreateRNNDataDescriptor(&desc));
    }
    ~CudnnRnnDataDescriptor() {
        // Destructors report and never raise (6.B17).
        if (desc) peekCudaErrorsAlways(cudnnDestroyRNNDataDescriptor(desc));
    }
    CudnnRnnDataDescriptor(const CudnnRnnDataDescriptor&) = delete;
    CudnnRnnDataDescriptor& operator=(const CudnnRnnDataDescriptor&) = delete;
    operator cudnnRNNDataDescriptor_t() const { return desc; }

    void set(cudnnDataType_t dtype, int seq_length, int batch_size,
             int vector_size, const int *seq_lengths) {
        // SEQ_MAJOR_UNPACKED is [seq, batch, vector] contiguous, which is
        // exactly what the per-timestep descriptors described: dims
        // {batch, vector, 1} with strides {vector, 1, 1}, one per step.
        checkCudaErrors(cudnnSetRNNDataDescriptor(
            desc, dtype, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
            seq_length, batch_size, vector_size, seq_lengths, nullptr));
    }
};

/** The per-sequence lengths, which v8 wants in two memories at once.

    `cudnnSetRNNDataDescriptor` reads them from the host; `cudnnRNNForward`
    and `cudnnRNNBackwardData_v8` read them from the device. jittor pads
    nothing -- every sequence in the batch is the full length -- so both are
    `batch_size` copies of one number.
 */
struct CudnnRnnSeqLengths {
    vector<int32_t> host;
    CudnnWorkspace device;

    CudnnRnnSeqLengths(int batch_size, int seq_length)
        : host(batch_size, seq_length),
          device(sizeof(int32_t) * (size_t)batch_size) {
        // Async on the null stream, which is the stream every cuDNN call
        // below runs on, so the copy is ordered before them. The source is
        // pageable, so the runtime has staged it by the time this returns and
        // `host` may die whenever.
        checkCudaErrors(cudaMemcpyAsync(device.ptr, host.data(),
            sizeof(int32_t) * (size_t)batch_size, cudaMemcpyHostToDevice, 0));
    }

    const int* host_data() const { return host.data(); }
    const int32_t* dev() const { return (const int32_t*)device.ptr; }
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
