// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <random>

#include "var.h"
#include "init.h"
#include "ops/random_op.h"
#include "misc/cuda_flags.h"
#include "ops/op_register.h"

namespace jittor {

#ifndef JIT
RandomOp::RandomOp(NanoVector shape, NanoString dtype) {
    // auto curand_random = get_op_info("curand_random")
    // .get_constructor<NanoVector, NanoString>();
    // output = curand_random(shape, dtype);
    #ifdef HAS_CUDA
    if (use_cuda) {
        static VarPtr(*curand_random)(NanoVector, NanoString) = nullptr;
        if (!curand_random && has_op("curand_random")) {
            curand_random = get_op_info("curand_random")
                .get_constructor<VarPtr, NanoVector, NanoString>();
        }
        if (curand_random) {
            auto var = curand_random(shape, dtype);
            forward(var);
            return;
        }
    }
    #endif
    output = create_output(shape, dtype);
}

void RandomOp::jit_prepare() {
    add_jit_define("T", output->dtype());
}

#else // JIT
#ifdef JIT_cpu
void RandomOp::jit_run() {
    auto* generator = get_random_engine();
    std::uniform_real_distribution<T> distribution(0.0,1.0);
    auto* __restrict__ x = output->ptr<T>();
    index_t num = output->num;
    for (index_t i=0; i<num; i++)
        x[i] = distribution(*generator);
}
#else // JIT_cuda
void RandomOp::jit_run() {
    // cuda device code
}
#endif // JIT_cpu
#endif // JIT

} // jittor