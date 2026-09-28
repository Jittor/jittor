// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/var.h"
#include "runtime/init.h"
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "curand_random_op.h"
#include "curand_wrapper.h"
#include "runtime/cuda_streams.h"
#include "runtime/device.h"
#include "stream_compat.h"
#include "utils/philox.h"

namespace jittor {

#ifndef JIT
CurandRandomOp::CurandRandomOp(NanoVector shape, NanoString dtype, NanoString type) {
    set_flag(OpFlags::_cuda, 1);
    // The CUDA RNG kernel generates float and double only. jt.random() lowers
    // float16 and bfloat16 to a float32 draw followed by a cast.
    USER_CHECK(dtype == ns_float32 || dtype == ns_float64)
        << "curand_random supports float32 and float64 only, got" << dtype
        << "\n  Draw float32 and cast if another dtype is needed.";
    output = create_output(shape, dtype);
    this->type = type;
    USER_CHECK(type == ns_normal || type == ns_uniform);
}

void CurandRandomOp::jit_prepare(JK& jk) {
    jk << "«T:" << output->dtype();
    jk << "«R:" << type;
}

#else // JIT
#ifdef JIT_cpu
void CurandRandomOp::jit_run() {
}
#else // JIT_cuda

__device__ inline uint32_t jittor_philox_word(
        uint64 seed, uint64 word_index) {
    auto value = philox4x32_10(seed, word_index >> 2);
    switch (word_index & 3) {
    case 0: return value.x0;
    case 1: return value.x1;
    case 2: return value.x2;
    default: return value.x3;
    }
}

__global__ void jittor_philox_random_kernel(
        T* output, index_t num, uint64 seed, uint64 word_offset,
        uint64 work_items) {
    for (uint64 item = uint64(blockIdx.x) * blockDim.x + threadIdx.x;
         item < work_items; item += uint64(blockDim.x) * gridDim.x) {
        @if(@strcmp(@R,uniform)==0,
            @if(@strcmp(@T,float32)==0,
                // One Philox block supplies four adjacent float outputs. This
                // preserves split-vs-single continuation even when a previous
                // draw ended in the middle of a block.
                uint64 block_word = (word_offset >> 2) + item;
                auto value = philox4x32_10(seed, block_word);
                uint32_t words[4];
                words[0] = value.x0;
                words[1] = value.x1;
                words[2] = value.x2;
                words[3] = value.x3;
                #pragma unroll
                for (int lane = 0; lane < 4; ++lane) {
                    uint64 global_word = (block_word << 2) + uint64(lane);
                    if (global_word >= word_offset) {
                        uint64 out_index = global_word - word_offset;
                        if (out_index < uint64(num))
                            output[out_index] = philox_uniform_float(words[lane]);
                    }
                }
            ,
                uint64 first = word_offset + item * 2;
                output[item] = philox_uniform_double(
                    jittor_philox_word(seed, first),
                    jittor_philox_word(seed, first + 1));
            )
        ,
            @if(@strcmp(@T,float32)==0,
                auto value = philox4x32_10(seed, (word_offset >> 2) + item);
                float u0 = philox_uniform_float(value.x0);
                float u1 = philox_uniform_float(value.x1);
                float u2 = philox_uniform_float(value.x2);
                float u3 = philox_uniform_float(value.x3);
                float radius0 = sqrtf(-2.0f * logf(u0));
                float angle0 = 6.2831853071795864769f * u1;
                float radius1 = sqrtf(-2.0f * logf(u2));
                float angle1 = 6.2831853071795864769f * u3;
                uint64 base = item * 4;
                if (base < uint64(num)) output[base] = radius0 * cosf(angle0);
                if (base + 1 < uint64(num)) output[base + 1] = radius0 * sinf(angle0);
                if (base + 2 < uint64(num)) output[base + 2] = radius1 * cosf(angle1);
                if (base + 3 < uint64(num)) output[base + 3] = radius1 * sinf(angle1);
            ,
                auto value = philox4x32_10(seed, (word_offset >> 2) + item);
                double u0 = philox_uniform_double(value.x0, value.x1);
                double u1 = philox_uniform_double(value.x2, value.x3);
                double radius = sqrt(-2.0 * log(u0));
                double angle = 6.283185307179586476925286766559 * u1;
                uint64 base = item * 2;
                if (base < uint64(num)) output[base] = radius * cos(angle);
                if (base + 1 < uint64(num)) output[base + 1] = radius * sin(angle);
            )
        )
    }
}

void CurandRandomOp::jit_run() {
    auto* __restrict__ x = output->ptr<T>();
    index_t num = output->num;
    if (num == 0) return;
    uint64 words;
    uint64 alignment;
    uint64 work_items;
    @if(@strcmp(@R,uniform)==0,
        words = uint64(num) * @if(@strcmp(@T,float32)==0,1,2);
        alignment = 1;
        work_items = @if(@strcmp(@T,float32)==0,
            ((uint64(num) + 3) >> 2), uint64(num));
    ,
        alignment = 4;
        work_items = @if(@strcmp(@T,float32)==0,
            ((uint64(num) + 3) >> 2), ((uint64(num) + 1) >> 1));
        words = work_items * 4;
    )
    uint64 seed, word_offset;
    curand_reserve_philox_words(words, alignment, seed, word_offset);
    @if(@strcmp(@R,uniform)==0,
        @if(@strcmp(@T,float32)==0,
            work_items = ((word_offset & 3) + uint64(num) + 3) >> 2;
        ,)
    ,)
    const int threads = 256;
    int blocks = int(std::min<uint64>((work_items + threads - 1) / threads, 4096));
    jittor_philox_random_kernel<<<blocks, threads, 0,
        cuda_compute_stream(current_device())>>>(
            x, num, seed, word_offset, work_items);
    checkCudaErrors(cudaGetLastError());
}
#endif // JIT_cpu
#endif // JIT

} // jittor
