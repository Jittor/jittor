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
#include "runtime/rematerialization_scope.h"
#include "runtime/rng_state.h"

namespace jittor {

#ifndef JIT
RandomOp::RandomOp(NanoVector shape, NanoString dtype, NanoString type,
                   RandomGenerator* generator, bool torch_normal) : torch_normal(torch_normal) {
    const auto backend = construction_target_backend();
    if (torch_normal)
        USER_CHECK(backend == BackendId::Cpu && type == ns_normal
                   && (dtype == ns_float32 || dtype == ns_float64))
            << "Torch CPU normal generation requires a CPU float32 or float64 output";
    #ifdef HAS_ACCELERATOR
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
    else rematerialization_preserve_rng = rematerialization_preserves_rng();
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
    if (torch_normal) {
        if (!output->num) return;
        auto& owner = runtime_rng_state();
        auto& normal = gen ? gen->cpu_torch_normal : owner.host_torch_normal;
        const uint64 seed = gen ? uint64(gen->seed_) : owner.host_seed;
        if (rematerialization_preserve_rng && rematerialization_torch_rng) {
            auto replay = *rematerialization_torch_rng;
            fill_torch_cpu_normal(replay, output->mem_ptr, output->num, output->dtype() == ns_float64);
        } else {
            if (!normal) normal = std::make_shared<TorchCpuNormal>(seed);
            if (rematerialization_preserve_rng)
                rematerialization_torch_rng = std::make_shared<TorchCpuNormal>(*normal);
            fill_torch_cpu_normal(*normal, output->mem_ptr, output->num, output->dtype() == ns_float64);
        }
        return;
    }
    auto* generator = gen ? &gen->cpu_engine : get_random_engine();
    std::default_random_engine replay_engine;
    if (rematerialization_preserve_rng) {
        if (rematerialization_rng) {
            replay_engine = *rematerialization_rng;
            generator = &replay_engine;
        } else rematerialization_rng = std::make_shared<std::default_random_engine>(*generator);
    }
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
