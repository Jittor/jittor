// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//      Zheng-Ning Liu <lzhengning@gmail.com>
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <memory>
#include <unordered_map>

#include "cudnn_rnn_descriptor.h"
#include "cudnn_descriptor.h"
#include "misc/cuda_flags.h"

namespace jittor {

// ---------------------------------------------------------------- dropout
//
// One dropout state per dropout probability, kept for the life of the seed.
//
// cuDNN advances the state in place, so reusing it is what makes step N+1's
// mask differ from step N's. Rebuilding it per call (the old behaviour: it was
// a member of a descriptor local to jit_run) re-seeded it from get_seed()
// every time, which made every step identical -- silently, since nothing in
// the outputs looks wrong.
static std::unordered_map<float, std::unique_ptr<DropoutDescriptor>> rnn_dropout_cache;
// The seed the cached states were built from. jt.set_seed() must still
// reproduce a run, so a (re-)seed throws the advanced states away.
static int rnn_dropout_cache_seed = 0;
static bool rnn_dropout_cache_seeded = false;

void cudnn_rnn_dropout_reset() {
    rnn_dropout_cache.clear();
    rnn_dropout_cache_seeded = false;
}

cudnnDropoutDescriptor_t cudnn_rnn_dropout_descriptor(cudnnHandle_t handle, float dropout) {
    int seed = get_seed();
    if (!rnn_dropout_cache_seeded || seed != rnn_dropout_cache_seed) {
        // Covers the seed being changed without going through set_seed's
        // callback (and makes the invariant local rather than remote).
        rnn_dropout_cache.clear();
        rnn_dropout_cache_seed = seed;
        rnn_dropout_cache_seeded = true;
    }
    auto iter = rnn_dropout_cache.find(dropout);
    if (iter == rnn_dropout_cache.end())
        iter = rnn_dropout_cache.emplace(dropout,
            std::unique_ptr<DropoutDescriptor>(
                new DropoutDescriptor(handle, dropout, seed))).first;
    return iter->second->desc;
}

struct cudnn_rnn_dropout_initer {
    cudnn_rnn_dropout_initer() {
        if (!get_device_count()) return;
        add_set_seed_callback([](int) { cudnn_rnn_dropout_reset(); });
    }
} cudnn_rnn_dropout_init;

// ------------------------------------------------------------ reserve space
//
// cuDNN will only report the training reserve space through descriptors, and
// the answer is needed during shape inference. It used to be re-derived on
// every inference -- i.e. every training step -- from seq_length freshly
// created tensor descriptors that were never destroyed: seq_length leaked
// descriptors per step. Cached per configuration instead, and the descriptors
// that the one query does need are destroyed.
struct RnnReserveKey {
    int mode, input_size, hidden_size, num_layers, seq_length, batch_size;
    int bidirectional, has_dropout, dtype;
    bool operator==(const RnnReserveKey& o) const {
        return mode == o.mode && input_size == o.input_size
            && hidden_size == o.hidden_size && num_layers == o.num_layers
            && seq_length == o.seq_length && batch_size == o.batch_size
            && bidirectional == o.bidirectional && has_dropout == o.has_dropout
            && dtype == o.dtype;
    }
};

struct RnnReserveKeyHash {
    size_t operator()(const RnnReserveKey& k) const {
        size_t h = 1469598103934665603ull;
        for (int v : {k.mode, k.input_size, k.hidden_size, k.num_layers,
                      k.seq_length, k.batch_size, k.bidirectional,
                      k.has_dropout, k.dtype})
            h = (h ^ (size_t)(uint32)v) * 1099511628211ull;
        return h;
    }
};

static std::unordered_map<RnnReserveKey, size_t, RnnReserveKeyHash> rnn_reserve_cache;

size_t cudnn_rnn_reserve_space_size(string mode, int input_size, int hidden_size,
        int num_layers, float dropout, bool bidirectional,
        int seq_length, int batch_size, cudnnDataType_t dtype) {
    RnnReserveKey key{(int)rnn_string_to_rnn_mode(mode), input_size, hidden_size,
        num_layers, seq_length, batch_size, (int)bidirectional,
        dropout > 0 ? 1 : 0, (int)dtype};
    auto iter = rnn_reserve_cache.find(key);
    if (iter != rnn_reserve_cache.end()) return iter->second;

    int in_dims[3] = {batch_size, input_size, 1};
    int in_strides[3] = {in_dims[1] * in_dims[2], in_dims[2], 1};

    // Owned: the query throws on failure and the descriptors go either way.
    CudnnTensorDescriptorArray xDesc(seq_length);
    for (int i = 0; i < seq_length; ++i)
        checkCudaErrors(cudnnSetTensorNdDescriptor(xDesc[i], dtype, 3, in_dims, in_strides));
    RnnDescriptor rnn_desc(cudnn_handle, mode, hidden_size, num_layers, dropout, bidirectional, dtype);
    size_t size = rnn_desc.reserve_space_size(xDesc.data(), seq_length);

    rnn_reserve_cache[key] = size;
    // Readable proof that this is a cache miss and not a per-step query; see
    // tests/backends/cuda/test_cudnn_rnn_dropout.py.
    LOGvvv << "cudnn_rnn reserve space query: seq_length=" >> seq_length
        << "batch_size=" >> batch_size << "size=" >> size;
    return size;
}

vector<int32_t> cudnn_rnn_weight_offset(string mode, int input_size, int hidden_size, int num_layers, int proj_size, bool bias, bool bidirectional, string dtype) {
    // The offsets and the flat size are in *elements* of the weight dtype, so
    // they depend on it. This used to be fp32 throughout, which is why the
    // caller could only ever build an fp32 flat weight.
    NanoString ns(dtype.c_str());
    cudnnDataType_t data_type = cudnn_rnn_dtype(ns);
    int elem_size = ns.dsize();

    // A pseudo mini-batch for fetching weight space size.
    int dimX[] = {1, input_size, 1};
    int strideX[] = {input_size, 1, 1};
    CudnnTensorDescriptor xDesc;
    checkCudaErrors(cudnnSetTensorNdDescriptor(xDesc, data_type, 3, dimX, strideX));

    RnnDescriptor rnn_desc(cudnn_handle, mode, hidden_size, num_layers, 0, bidirectional, data_type);
    int weightSpaceSize = rnn_desc.weight_space_size(xDesc);
    RnnWeightDescriptor w_desc(weightSpaceSize, data_type, elem_size);
    
    vector<int> weight_offsets;
    weight_offsets.push_back(weightSpaceSize / elem_size);    

    int num_directions = bidirectional + 1;
    int num_linear_layers = rnn_string_to_num_linear_layers(mode);
    
    for (int layer = 0; layer < num_layers * num_directions; layer++) {
        for (int linLayerID = 0; linLayerID < num_linear_layers; linLayerID++) {
            // Owned per iteration; they used to leak two filter descriptors
            // per linear layer per query.
            CudnnFilterDescriptor linLayerMatDesc;
            CudnnFilterDescriptor linLayerBiasDesc;
            char *linLayerMat = nullptr;
            char *linLayerBias = nullptr;

            checkCudaErrors(cudnnGetRNNLinLayerMatrixParams(
                cudnn_handle, rnn_desc.desc,
                layer, 
                xDesc, 
                w_desc.desc, 
                nullptr,
                linLayerID,
                linLayerMatDesc, 
                (void **) &linLayerMat
            ));
            // cuDNN hands back a byte address into a buffer based at 0; the
            // caller indexes the flat weight by element.
            weight_offsets.push_back((int)((linLayerMat - (char *) nullptr) / elem_size));

            if (bias) {
                checkCudaErrors(cudnnGetRNNLinLayerBiasParams(
                    cudnn_handle, rnn_desc.desc,
                    layer, 
                    xDesc, 
                    w_desc.desc, 
                    nullptr,
                    linLayerID,
                    linLayerBiasDesc, 
                    (void **) &linLayerBias
                ));
                weight_offsets.push_back((int)((linLayerBias - (char *) nullptr) / elem_size));
            }

        }
    }

    return weight_offsets;
}


} // jittor
