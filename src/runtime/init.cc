// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "runtime/device.h"
#include "runtime/backend.h"
#include <random>

#include <csignal>
#include "runtime/init.h"
#include "runtime/async_exec.h"
#include "runtime/rng_state.h"
#include "runtime/executor_entry.h"
#include "ops/op_register.h"
#include "ops/composite/op_registration.h"
#include "ops/composite/tape_op.h"
#include "core/fused_op.h"
#include "core/var.h"
#include "core/op.h"
#include "core/executor.h"
#include "runtime/float32_precision.h"

namespace jittor {

DEFINE_FLAG(vector<int>, cuda_archs, {}, "Cuda arch");
// How precisely a float32 product is accumulated, on the same three-name
// scale torch uses, shared by matmul and convolution. See
// runtime/float32_precision.h for the full mapping and for why the three flags
// below are now overrides on top of it rather than four separate encodings.

// Two policies, one per library, with torch's defaults: matmul at `highest`
// (torch.backends.cuda.matmul.allow_tf32 = False) and cuDNN -- convolution and
// RNN -- at `high` (torch.backends.cudnn.allow_tf32 = True).
//
// They used to be one flag that wrote both, defaulting to `highest`, so a
// native float32 convolution ran at CUDNN_FMA_MATH where torch and Jittor
// 1.3.10 both let cuDNN use tf32 on Ampere and later: a ResNet-50 training
// step at batch 64 took 128 ms against 1.3.10's 95, all of it in the
// convolution backward. Separate flags are also what lets each default stand:
// a scope over one would otherwise restore both on exit and overwrite the
// other's default.
DEFINE_FLAG_WITH_SETTER(string, float32_matmul_precision, "highest",
    "Accumulate precision for float32 matmul: highest (float32), "
    "high (tf32), medium (bfloat16). float16/bfloat16 inputs always "
    "accumulate in float32. Convolution has its own, float32_cudnn_precision.");

void setter_float32_matmul_precision(const string& old_value, const string& value) {
    int tier = parse_float32_precision_tier(value);
    // Throwing here rolls the flag back to `old_value` (see DEFINE_FLAG_WITH_SETTER),
    // so a typo leaves the previous policy in force instead of a half-applied one.
    if (tier < 0)
        LOGf << "float32_matmul_precision must be one of highest, high, medium; got"
            << '"' >> value >> '"';
    runtime_jit_policy().float32_matmul_precision_tier = tier;
}

DEFINE_FLAG_WITH_SETTER(string, float32_cudnn_precision, "high",
    "Accumulate precision for float32 cuDNN convolution and RNN: highest "
    "(float32, CUDNN_FMA_MATH), high (tf32, the default, as torch's "
    "cudnn.allow_tf32), medium (bfloat16). float16/bfloat16 inputs always "
    "accumulate in float32.");

void setter_float32_cudnn_precision(const string& old_value, const string& value) {
    int tier = parse_float32_precision_tier(value);
    if (tier < 0)
        LOGf << "float32_cudnn_precision must be one of highest, high, medium; got"
            << '"' >> value >> '"';
    runtime_jit_policy().float32_cudnn_precision_tier = tier;
}

// Deprecated: each raises the tier for the domain it names. Kept because they
// are what `torch.backends.*` maps onto and what existing code sets; prefer
// float32_matmul_precision, which covers both domains at once.
DEFINE_FLAG(int, use_tensorcore, 0,
    "Deprecated, use float32_matmul_precision and float32_cudnn_precision. "
    "Raises the float32 accumulate tier for matmul and convolution: "
    "1=high(tf32), 2 and 3=medium(bfloat16).");
DEFINE_FLAG(int, cuda_allow_tf32, 0,
    "Deprecated, use float32_matmul_precision. Raises the float32 matmul "
    "accumulate tier to high (tf32).");
DEFINE_FLAG(int, cuda_allow_cudnn_tf32, 0,
    "Deprecated, use float32_cudnn_precision. Raises the float32 cuDNN "
    "convolution accumulate tier to high (tf32).");
// Ascend ACL equivalent of torch.npu.matmul.allow_hf32: permits FP32
// matmul/bmm/conv to run aclnn with ALLOW_FP32_DOWN_PRECISION (fp16 split
// cube). Required on Ascend310P whose cube has no FP32 KEEP_DTYPE; on chips
// that support full FP32 it still lowers precision, so it stays opt-in.
// Defined here (src/) rather than backends/acl/ because the flags binding
// generator only scans the core source tree.
DEFINE_FLAG(int, acl_allow_hf32, 0,
    "Ascend ACL only. Allows FP32 matmul/bmm/convolution to run with "
    "ALLOW_FP32_DOWN_PRECISION (reduced precision, ~1e-3). Required on "
    "Ascend310P whose cube unit cannot run FP32 KEEP_DTYPE.");

vector<set_seed_callback> callbacks;

// fron fetch_op.cc
EXTERN_LIB list<VarPtr> fetcher;
EXTERN_LIB list<VarPtr> fetcher_to_free;
EXTERN_LIB vector<void(*)()> take_cleanup_callbacks();
EXTERN_LIB volatile sig_atomic_t exited;

void cleanup() {
    exited = true;
    fetcher_to_free.clear();
    fetcher.clear();
    // Walk a private copy, taken under the registration lock, rather than the
    // live vector: a callback may register another one (`get_resources` does,
    // on whichever thread first touches a side stream), and `push_back` then
    // reallocates storage the walk is still holding iterators into. Taking the
    // whole list also makes a second `cleanup()` a no-op instead of a second
    // round of stream destruction.
    for (auto cb : take_cleanup_callbacks())
        cb();
}

static void init_device_architectures() {
#ifdef HAS_ACCELERATOR
    if (cuda_archs.size()) return;
    const auto& backend = backend_ops(accelerator_backend_id());
    if (!backend.architectures) return;
    cuda_archs = backend.architectures();
    if (cuda_archs.size()) LOGi << "Found device architectures:" << cuda_archs;
#endif
}

void init() {
    // init default_random_engine
    set_seed(time(0));
    // init fused op
    OpDef fused("fused", "", "");
    Codegen fused_codegen;
    fused_codegen.fragment = [](Op* op, JK& key) {
        static_cast<FusedOp*>(op)->prepare_fused_key(key);
    };
    fused_codegen.prepare = fused_codegen.fragment;
    fused_codegen.optimize = [](Op*, string&) {};
    Kernel fused_kernel;
    fused_kernel.jit = [](Op* op, JK& key) {
        static_cast<FusedOp*>(op)->execute_fused_prepared(key);
    };
    fused.codegen = fused_codegen;
    fused.implementations.emplace(BackendId::Cpu, OpImplementation{fused_kernel, fused_codegen});
    fused.implementations.emplace(accelerator_backend_id(), OpImplementation{fused_kernel, fused_codegen});
    op_registe(fused);
    register_op_definition<Tapes>({"tapes", "", ""});
    init_device_architectures();
    LOGv << "sizeof(Node)" << sizeof(Node);
    LOGv << "sizeof(Var)" << sizeof(Var);
    LOGv << "sizeof(Op)" << sizeof(Op);
}

static void apply_global_seed(uint64 seed) {
    auto& rng = runtime_rng_state();
    rng.global_seed = seed;
    rng.host_seed = seed;
    rng.host_engine.seed(seed);
    rng.acl_streams.clear();
    // Legacy callbacks retain their ABI. Seed-aware backends read the full
    // value from the shared owner; callbacks such as dropout reset ignore it.
    for (auto cb : callbacks)
        cb(static_cast<int>(seed));
}

void set_seed(int seed) {
    async_exec_wait();
    apply_global_seed(static_cast<uint64>(seed));
}

EXTERN_LIB void sync_all(bool device_sync);

void set_seed64(uint64 seed) {
    async_exec_wait();
    ExecutorEntryScope lock;
    // A lazily issued draw belongs to the stream before this reseed.
    sync_all(true);
    apply_global_seed(seed);
}

int get_seed() {
    return static_cast<int>(runtime_rng_state().host_seed);
}

void add_set_seed_callback(set_seed_callback callback) {
    callbacks.push_back(callback);
    callback(get_seed());
}

std::default_random_engine* get_random_engine() {
    return &runtime_rng_state().host_engine;
}

#ifdef HAS_ACCELERATOR
bool no_device_error_when_free = 0;
#endif

void jt_init_subprocess() {
    #ifdef HAS_ACCELERATOR
    runtime_device_state().use_cuda = 0;
    runtime_executor().last_is_cuda = false;
    no_device_error_when_free = 1;
    #endif
    callbacks.clear();
}

}
