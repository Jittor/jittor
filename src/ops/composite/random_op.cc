// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <random>
#include <limits>

#include "core/var.h"
#include "runtime/init.h"
#include "ops/composite/random_op.h"
#include "runtime/device.h"
#include "ops/op_register.h"
#include "ops/composite/op_capability.h"

namespace jittor {

#ifndef JIT
RandomOp::RandomOp(NanoVector shape, NanoString dtype, NanoString type, RandomGenerator* generator) {
    #ifdef HAS_ACCELERATOR
    const auto backend = construction_target_backend();
    if (backend != BackendId::Cpu) {
        auto accelerated_random = find_op_capability<VarPtr, NanoVector, NanoString, NanoString, RandomGenerator*>(
            backend, OpCapability::Random, shape, dtype, type, generator);
        if (accelerated_random) {
            auto var = accelerated_random(shape, dtype, type, generator);
            forward(var);
            return;
        }
    }
    #endif
    output = create_output(shape, dtype);
    this->type = type;
    if (generator) this->gen = generator->state;
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

// Separate overload/capability preserves the floating constructor contract.
RandomOp::RandomOp(NanoVector shape, int64 low, int64 high, bool high_none,
        RandomGenerator* generator) {
    #ifdef HAS_ACCELERATOR
    const auto backend = construction_target_backend();
    if (backend != BackendId::Cpu) {
        USER_CHECK(backend == BackendId::Cuda)
            << "random_int64 supports CPU and CUDA only";
        auto integer_random = find_op_capability<VarPtr, NanoVector, int64, int64, bool, RandomGenerator*>(
            backend, OpCapability::RandomInteger, shape, low, high, high_none, generator);
        USER_CHECK(integer_random) << "random_int64 CUDA provider is unavailable";
        forward(integer_random(shape, low, high, high_none, generator));
        return;
    }
    #endif
    USER_CHECK(high_none || low < high) << "random_int64 requires low < high";
    USER_CHECK(!generator || (generator->state && generator->state->device_id < 0))
        << "random_int64 CPU output requires a CPU Generator";
    const int64 maximum = std::numeric_limits<int64>::max();
    int64 num = 1;
    for (uint i=0; i<shape.size(); ++i) {
        USER_CHECK(shape[i] >= 0) << "random_int64 negative dimension";
        int64 next;
        USER_CHECK(!__builtin_mul_overflow(num, int64(shape[i]), &next))
            << "random_int64 element count overflows int64";
        num = next;
    }
    USER_CHECK(num <= maximum / int64(sizeof(int64)))
        << "random_int64 byte count overflows int64";
    // high-1 is representable because a bounded request has low < high.
    // std::uniform_int_distribution is integer-native, including full int64;
    // no float generation/cast, Python PRNG, or CUDA fallback is involved.
    integer_random = true;
    integer_low = low;
    integer_high = high_none ? maximum : high - 1;
    type = ns_uniform;
    if (generator) gen = generator->state;
    set_flag(OpFlags::_cpu, 1);
    set_flag(OpFlags::_cuda, 0);
    output = create_output(shape, ns_int64);
    USER_CHECK(output->num == num) << "random_int64 shape count mismatch";
}

void RandomOp::jit_prepare(JK& jk) {
    jk << "«T:" << output->dtype();
    jk << "«R:" << type;
    jk << "«I:" << integer_random;
}

#else // JIT
#ifdef JIT_cpu
void RandomOp::jit_run() {
    auto* generator = gen ? &gen->cpu_engine : get_random_engine();
    @if(I,
        std::uniform_int_distribution<int64> distribution(integer_low, integer_high);,
        @if(@strcmp(@R,uniform)==0,
            std::uniform_real_distribution<T> distribution(0.0,1.0);,
            std::normal_distribution<T> distribution(0.0,1.0);
        )
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
