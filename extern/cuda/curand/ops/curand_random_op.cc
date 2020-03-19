// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <random>

#include "var.h"
#include "init.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <helper_cuda.h>
#include "curand_random_op.h"
#include "curand_warper.h"

namespace jittor {

#ifndef JIT
CurandRandomOp::CurandRandomOp(NanoVector shape, NanoString dtype) {
    flags.set(NodeFlags::_cuda, 1);
    output = create_output(shape, dtype);
}

void CurandRandomOp::jit_prepare() {
    add_jit_define("T", output->dtype());
}

#else // JIT
#ifdef JIT_cpu
void CurandRandomOp::jit_run() {
}
#else // JIT_cuda
void CurandRandomOp::jit_run() {
    auto* __restrict__ x = output->ptr<T>();
    index_t num = output->num;
    if (sizeof(T) == 4) {
        checkCudaErrors( curandGenerateUniform(gen, (float*)x, num) );
    } else {
        checkCudaErrors( curandGenerateUniformDouble(gen, (float64*)x, num) );
    }
}
#endif // JIT_cpu
#endif // JIT

} // jittor