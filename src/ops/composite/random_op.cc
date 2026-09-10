// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <random>

#include "core/var.h"
#include "runtime/init.h"
#include "ops/composite/random_op.h"
#include "runtime/device.h"
#include "ops/op_register.h"
#include "ops/composite/op_capability.h"

namespace jittor {

#ifndef JIT
RandomOp::RandomOp(NanoVector shape, NanoString dtype, NanoString type) {
    #ifdef HAS_ACCELERATOR
    const auto backend = construction_target_backend();
    if (backend != BackendId::Cpu) {
        auto accelerated_random = find_op_capability<VarPtr, NanoVector, NanoString, NanoString>(
            backend, OpCapability::Random, shape, dtype, type);
        if (accelerated_random) {
            auto var = accelerated_random(shape, dtype, type);
            forward(var);
            return;
        }
    }
    #endif
    output = create_output(shape, dtype);
    this->type = type;
    USER_CHECK(type == ns_normal || type == ns_uniform);
    #ifdef HAS_ACCELERATOR
    // No capability op, but the backend may run `random` itself -- ACL does,
    // through its own launcher. Without this flag the executor treats the op
    // as CPU-only and reports a backend fallback for every weight
    // initialisation, dropout mask and sampled tensor.
    if (backend != BackendId::Cpu && backend_runs_op_natively(backend, "random"))
        set_flag(OpFlags::_cuda);
    #endif
}

void RandomOp::jit_prepare(JK& jk) {
    jk << "«T:" << output->dtype();
    jk << "«R:" << type;
}

#else // JIT
#ifdef JIT_cpu
void RandomOp::jit_run() {
    auto* generator = get_random_engine();
    @if(@strcmp(@R,uniform)==0,
        std::uniform_real_distribution<T> distribution(0.0,1.0);,
        std::normal_distribution<T> distribution(0.0,1.0);
    )
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
