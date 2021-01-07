// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
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
#include "curand_warper.h"

namespace jittor {

#ifndef JIT
CurandRandomOp::CurandRandomOp(NanoVector shape, NanoString dtype, NanoString type) {
    flags.set(NodeFlags::_cuda, 1);
    output = create_output(shape, dtype);
    this->type = type;
    ASSERT(type == ns_normal || type == ns_uniform);
}

void CurandRandomOp::jit_prepare(JK& jk) {
    jk << _CS("[T:") << output->dtype();
    jk << _CS("][R:") << type << ']';
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
    // curand doesn't support even number, we add 1 when it is even
    // because allocator will make odd chunks, so this wouldn't cause
    // segmentation fault
    num += num&1;
    @if(@strcmp(@R,uniform)==0,
        checkCudaErrors(curandGenerateUniform@TT (gen, x, num));,
        checkCudaErrors(curandGenerateNormal@TT (gen, x, num, 0, 1));
    )
}
#endif // JIT_cpu
#endif // JIT

} // jittor