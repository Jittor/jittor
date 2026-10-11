// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <random>

#include "core/var.h"
#include "runtime/init.h"
#include <cuda_runtime.h>
#include <curand.h>
#ifdef JIT_cuda
#include <curand_kernel.h>
#endif
#include "helper_cuda.h"
#include "curand_random_op.h"
#include "curand_wrapper.h"
#include "core/executor.h"
#include "runtime/device.h"
#include "runtime/random_generator.h"

namespace jittor {

#ifndef JIT
CurandRandomOp::CurandRandomOp(NanoVector shape, NanoString dtype, NanoString type, RandomGenerator* generator) {
    set_flag(OpFlags::_cuda, 1);
    // curand generates float and double only. Anything else used to expand to
    // curandGenerate*Double against a pointer of the wrong type and fail deep
    // inside nvcc; say so here instead. jt.random() already lowers float16 and
    // bfloat16 to a float32 draw plus a cast.
    USER_CHECK(dtype == ns_float32 || dtype == ns_float64)
        << "curand_random supports float32 and float64 only, got" << dtype
        << "\n  Draw float32 and cast if another dtype is needed.";
    output = create_output(shape, dtype);
    this->type = type;
    USER_CHECK(type == ns_normal || type == ns_uniform);
    if (generator) {
        // Raw 32-bit values per output value: a double takes two. Normal
        // values (Box-Muller) and doubles are produced in pairs, and an odd
        // count still consumes the whole last pair. Must match
        // philox_random_kernel below. Reserving here makes the stream
        // position a property of issue order, so get_state() between two
        // draws covers the first even while it has not executed yet.
        int64 num = output->num;
        int64 per_value = dtype == ns_float64 ? 2 : 1;
        bool pairs = type == ns_normal || dtype == ns_float64;
        int64 values = pairs ? num + (num & 1) : num;
        int64 raw;
        // Unreachable for any allocatable output (more than 2^62 values).
        ASSERT(!__builtin_mul_overflow(values, per_value, &raw));
        from_generator = true;
        philox_seed = (uint64)generator->state->seed_;
        philox_offset = (uint64)generator->state->reserve_cuda(raw);
    }
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
// One Philox4x32-10 stream per generator, keyed by the seed. Unit u (one
// float32 uniform value, or a pair otherwise) reads raw positions
// offset + u*RAW onward; each thread initialises once and walks
// kPhiloxUnitsPerThread consecutive units, so the result depends only on
// (seed, offset, num), never on the launch shape. RAW is what the curand
// device functions really consume on Philox: curand_uniform 1, curand_normal2
// 2, and curand_uniform2_double / curand_normal2_double 4 (one curand4).
// curand_uniform_double is avoided: it also takes a curand4 but uses only
// half of it, for a single double.
static constexpr int kPhiloxUnitsPerThread = 4;

template <class T, bool NORMAL>
__global__ static void philox_random_kernel(T* __restrict__ x, index_t num,
        unsigned long long seed, unsigned long long offset) {
    constexpr bool dbl = sizeof(T) == 8;
    constexpr bool PAIR = NORMAL || dbl;
    constexpr unsigned long long RAW = dbl ? 4 : (NORMAL ? 2 : 1);
    const index_t units = PAIR ? (num + 1) / 2 : num;
    const index_t first = ((index_t)blockIdx.x * blockDim.x + threadIdx.x) * kPhiloxUnitsPerThread;
    if (first >= units) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, 0, offset + (unsigned long long)first * RAW, &state);
    for (int k = 0; k < kPhiloxUnitsPerThread; ++k) {
        const index_t u = first + k;
        if (u >= units) return;
        if (dbl) {
            double2 v = NORMAL ? curand_normal2_double(&state) : curand_uniform2_double(&state);
            x[2*u] = (T)v.x;
            if (2*u+1 < num) x[2*u+1] = (T)v.y;
        } else if (NORMAL) {
            float2 v = curand_normal2(&state);
            x[2*u] = (T)v.x;
            if (2*u+1 < num) x[2*u+1] = (T)v.y;
        } else {
            x[u] = (T)curand_uniform(&state);
        }
    }
}

void CurandRandomOp::jit_run() {
    @define(TT,@if(@strcmp(@T,float32)==0,,Double))

    auto* __restrict__ x = output->ptr<T>();
    index_t num = output->num;
    if (num == 0) return;
    if (from_generator) {
        const index_t units = @if(@strcmp(@R,uniform)==0,@if(@strcmp(@T,float32)==0,num,(num + 1) / 2),(num + 1) / 2);
        const index_t threads = (units + kPhiloxUnitsPerThread - 1) / kPhiloxUnitsPerThread;
        const int block = 256;
        const index_t grid = (threads + block - 1) / block;
        philox_random_kernel<T, @if(@strcmp(@R,uniform)==0,false,true)>
            <<<(unsigned)grid, block>>>(x, num, philox_seed, philox_offset);
        checkCudaErrors(cudaGetLastError());
        return;
    }
    // Draws without a jt.Generator share the global curand host generator.
    curandGenerator_t local_gen = curand_bind_stream();
    // curandGenerateUniform has no parity requirement; curandGenerateNormal
    // wants an even count for pseudorandom generators. The old code rounded
    // the count up for both and wrote one element past the end of the output
    // -- it only stayed out of trouble because the allocator happened to leave
    // slack -- and consumed one extra value from the generator, so two uniform
    // draws of odd length no longer continued the stream that a single draw of
    // the combined length produces.
    //
    // Uniform now asks for exactly num. Normal fills the even prefix in place
    // and takes the last element from a two-element scratch buffer, so nothing
    // is written outside the output. An odd-length normal draw still consumes
    // num+1 values; that is inherent to the even-count requirement.
    // Count what this draw costs the generator, so a checkpoint has a position
    // to save. Measured against CURAND_RNG_PSEUDO_DEFAULT: a uniform draw of n
    // costs n and a normal draw of n costs n/2, whatever the precision, and a
    // mixed history costs the sum. The odd-length normal below draws num-1 and
    // then 2, so it costs (num-1)/2 + 1. (Draws from a jt.Generator returned
    // above: they keep their own Philox position.)
    //
    // NOTE: this whole @if(...)/@for(...) block is jittor's own JIT
    // templating syntax, whose argument splitter does not skip `//`
    // comments -- a comma inside a comment nested in here is parsed as a
    // macro-argument separator ("if wrong arguments" / args.size() checks
    // failing at JIT-compile time, not a normal C++ diagnostic). Comments
    // for this block live above it, comma-free, for exactly that reason.
    @if(@strcmp(@R,uniform)==0,
        curand_advance(current_device(), num);
        checkCudaErrors(curandGenerateUniform@TT (local_gen, x, num));
    ,
        index_t normal_cost = num/2;
        if (num & 1) normal_cost = (num-1)/2 + 1;
        curand_advance(current_device(), normal_cost);
        if (num & 1) {
            if (num > 1)
                checkCudaErrors(curandGenerateNormal@TT (local_gen, x, num-1, 0, 1));
            size_t tail_allocation;
            T* tail = (T*)runtime_executor().temp_allocator->alloc(2*sizeof(T), tail_allocation);
            checkCudaErrors(curandGenerateNormal@TT (local_gen, tail, 2, 0, 1));
            checkCudaErrors(cudaMemcpyAsync(x+num-1, tail, sizeof(T),
                cudaMemcpyDeviceToDevice, cudaStreamPerThread));
            runtime_executor().temp_allocator->free(tail, 2*sizeof(T), tail_allocation);
        } else {
            checkCudaErrors(curandGenerateNormal@TT (local_gen, x, num, 0, 1));
        }
    )
}
#endif // JIT_cpu
#endif // JIT

} // jittor
