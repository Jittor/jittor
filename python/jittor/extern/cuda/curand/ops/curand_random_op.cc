// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <random>

#include "var.h"
#include "init.h"
#include <cuda_runtime.h>
#include <curand.h>
#include "helper_cuda.h"
#include "curand_random_op.h"
#include "curand_wrapper.h"
#include "executor.h"

namespace jittor {

#ifndef JIT
CurandRandomOp::CurandRandomOp(NanoVector shape, NanoString dtype, NanoString type) {
    set_flag(OpFlags::_cuda, 1);
    // curand generates float and double only. Anything else used to expand to
    // curandGenerate*Double against a pointer of the wrong type and fail deep
    // inside nvcc; say so here instead. jt.random() already lowers float16 and
    // bfloat16 to a float32 draw plus a cast.
    ASSERT(dtype == ns_float32 || dtype == ns_float64)
        << "curand_random supports float32 and float64 only, got" << dtype
        << "\n  Draw float32 and cast if another dtype is needed.";
    output = create_output(shape, dtype);
    this->type = type;
    ASSERT(type == ns_normal || type == ns_uniform);
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
void CurandRandomOp::jit_run() {
    @define(TT,@if(@strcmp(@T,float32)==0,,Double))

    auto* __restrict__ x = output->ptr<T>();
    index_t num = output->num;
    if (num == 0) return;
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
    @if(@strcmp(@R,uniform)==0,
        checkCudaErrors(curandGenerateUniform@TT (gen, x, num));
    ,
        if (num & 1) {
            if (num > 1)
                checkCudaErrors(curandGenerateNormal@TT (gen, x, num-1, 0, 1));
            size_t tail_allocation;
            T* tail = (T*)exe.temp_allocator->alloc(2*sizeof(T), tail_allocation);
            checkCudaErrors(curandGenerateNormal@TT (gen, tail, 2, 0, 1));
            checkCudaErrors(cudaMemcpyAsync(x+num-1, tail, sizeof(T),
                cudaMemcpyDeviceToDevice, 0));
            exe.temp_allocator->free(tail, 2*sizeof(T), tail_allocation);
        } else {
            checkCudaErrors(curandGenerateNormal@TT (gen, x, num, 0, 1));
        }
    )
}
#endif // JIT_cpu
#endif // JIT

} // jittor