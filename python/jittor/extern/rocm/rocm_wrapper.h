// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Zheng-Ning Liu <lzhengning@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
#include <rocprim/device/device_segmented_reduce.hpp>

namespace jittor {

struct RocprimArgMax
{
    template<
        class Key,
        class Value
    >
    __host__ __device__ inline
    constexpr rocprim::key_value_pair<Key, Value>
    operator()(const rocprim::key_value_pair<Key, Value>& a,
               const rocprim::key_value_pair<Key, Value>& b) const
    {
        return ((b.value > a.value) || ((a.value == b.value) && (b.key > a.key))) ? b : a;
    }
};

struct RocprimArgMin
{
    template<
        class Key,
        class Value
    >
    __host__ __device__ inline
    constexpr rocprim::key_value_pair<Key, Value>
    operator()(const rocprim::key_value_pair<Key, Value>& a,
               const rocprim::key_value_pair<Key, Value>& b) const
    {
        return ((b.value < a.value) || ((a.value == b.value) && (b.key < a.key))) ? b : a;
    }
};

template<typename Key, typename Value>
static __global__ void global_index_to_segment_index(rocprim::key_value_pair<Key, Value>* d_out, Key* offsetsp, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tnum = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += tnum) {
        d_out[i].key -= offsetsp[i];
    }
}

template<
        typename InputIteratorT,
        typename OutputIteratorT,
        typename OffsetIteratorT
    >
static hipError_t rocprim_argmax(void * d_temp_storage,
                                 size_t& temp_storage_bytes,
                                 InputIteratorT d_in,
                                 OutputIteratorT d_out,
                                 int num_segments,
                                 OffsetIteratorT d_begin_offsets,
                                 OffsetIteratorT d_end_offsets,
                                 hipStream_t stream = 0,
                                 bool debug_synchronous = false) {
    using OffsetT = int;                                                                                                                     
    using T = typename std::iterator_traits<InputIteratorT>::value_type;
    using O = typename std::iterator_traits<OutputIteratorT>::value_type;
    using OutputTupleT = typename std::conditional<                                                     
                             std::is_same<O, void>::value,
                             rocprim::key_value_pair<OffsetT, T>,
                             O
                         >::type;

    using OutputValueT = typename OutputTupleT::Value;
    using IteratorT = rocprim::arg_index_iterator<InputIteratorT, OffsetT, OutputValueT>;

    IteratorT d_indexed_in(d_in);
    const OutputTupleT init(1, std::numeric_limits<T>::lowest());

    auto ret = rocprim::segmented_reduce(d_temp_storage, 
                                         temp_storage_bytes,                                          
                                         d_indexed_in, 
                                         d_out,
                                         num_segments, 
                                         d_begin_offsets, 
                                         d_end_offsets,
                                         RocprimArgMax(),
                                         init,
                                         stream,
                                         debug_synchronous);                                                                               
    if (d_temp_storage != NULL) {
        global_index_to_segment_index<<<max(1, num_segments / 1024), 1024>>>(d_out, d_begin_offsets, num_segments);
    }

    return ret;
}


template<
        typename InputIteratorT,
        typename OutputIteratorT,
        typename OffsetIteratorT
    >
static hipError_t rocprim_argmin(void * d_temp_storage,
                                 size_t& temp_storage_bytes,
                                 InputIteratorT d_in,
                                 OutputIteratorT d_out,
                                 int num_segments,
                                 OffsetIteratorT d_begin_offsets,
                                 OffsetIteratorT d_end_offsets,
                                 hipStream_t stream = 0,
                                 bool debug_synchronous = false) {
    using OffsetT = int;                                                                                                                     
    using T = typename std::iterator_traits<InputIteratorT>::value_type;
    using O = typename std::iterator_traits<OutputIteratorT>::value_type;
    using OutputTupleT = typename std::conditional<                                                     
                             std::is_same<O, void>::value,
                             rocprim::key_value_pair<OffsetT, T>,
                             O
                         >::type;

    using OutputValueT = typename OutputTupleT::Value;
    using IteratorT = rocprim::arg_index_iterator<InputIteratorT, OffsetT, OutputValueT>;

    IteratorT d_indexed_in(d_in);
    const OutputTupleT init(1, std::numeric_limits<T>::max());

    auto ret = rocprim::segmented_reduce(d_temp_storage, 
                                         temp_storage_bytes,                                          
                                         d_indexed_in, 
                                         d_out,
                                         num_segments, 
                                         d_begin_offsets, 
                                         d_end_offsets,
                                         RocprimArgMin(),
                                         init,
                                         stream,
                                         debug_synchronous);                                                                               
    if (d_temp_storage != NULL) {
        global_index_to_segment_index<<<max(1, num_segments / 1024), 1024>>>(d_out, d_begin_offsets, num_segments);
    }

    return ret;
}

}
