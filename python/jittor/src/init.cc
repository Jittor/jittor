// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "misc/cuda_flags.h"
#endif
#include <random>

#include <csignal>
#include "init.h"
#include "ops/op_register.h"
#include "var.h"
#include "op.h"
#include "executor.h"
#include "misc/float32_precision.h"

namespace jittor {

DEFINE_FLAG(vector<int>, cuda_archs, {}, "Cuda arch");
// How precisely a float32 product is accumulated, on the same three-name
// scale torch uses, shared by matmul and convolution. See
// misc/float32_precision.h for the full mapping and for why the three flags
// below are now overrides on top of it rather than four separate encodings.
int float32_matmul_precision_tier = F32_HIGHEST;

DEFINE_FLAG_WITH_SETTER(string, float32_matmul_precision, "highest",
    "Accumulate precision for float32 matmul and convolution: "
    "highest (float32), high (tf32), medium (bfloat16). "
    "float16/bfloat16 inputs always accumulate in float32.");

void setter_float32_matmul_precision(const string& old_value, const string& value) {
    int tier = parse_float32_precision_tier(value);
    // Throwing here rolls the flag back to `old_value` (see DEFINE_FLAG_WITH_SETTER),
    // so a typo leaves the previous policy in force instead of a half-applied one.
    if (tier < 0)
        LOGf << "float32_matmul_precision must be one of highest, high, medium; got"
            << '"' >> value >> '"';
    float32_matmul_precision_tier = tier;
}

// Deprecated: each raises the tier for the domain it names. Kept because they
// are what `torch.backends.*` maps onto and what existing code sets; prefer
// float32_matmul_precision, which covers both domains at once.
DEFINE_FLAG(int, use_tensorcore, 0,
    "Deprecated, use float32_matmul_precision. Raises the float32 accumulate "
    "tier for matmul and convolution: 1=high(tf32), 2 and 3=medium(bfloat16).");
DEFINE_FLAG(int, cuda_allow_tf32, 0,
    "Deprecated, use float32_matmul_precision. Raises the float32 matmul "
    "accumulate tier to high (tf32).");
DEFINE_FLAG(int, cuda_allow_cudnn_tf32, 0,
    "Deprecated, use float32_matmul_precision. Raises the float32 cuDNN "
    "convolution accumulate tier to high (tf32).");

unique_ptr<std::default_random_engine> eng;

vector<set_seed_callback> callbacks;
int current_seed;
int64 current_offset;

// fron fetch_op.cc
EXTERN_LIB list<VarPtr> fetcher;
EXTERN_LIB list<VarPtr> fetcher_to_free;
EXTERN_LIB vector<void(*)()> cleanup_callback;
EXTERN_LIB volatile sig_atomic_t exited;

void cleanup() {
    exited = true;
    fetcher_to_free.clear();
    fetcher.clear();
    for (auto cb : cleanup_callback)
        cb();
    cleanup_callback.clear();
}

static void init_cuda_devices() {
#ifdef IS_CUDA
    if (cuda_archs.size()) return;
    int count=0;
    cudaGetDeviceCount(&count);
    for (int i=0; i<count; i++) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        int number = devProp.major * 10 + devProp.minor;
        int found = 0;
        for (auto v : cuda_archs)
            if (v==number) {
                found = 1;
                break;
            }
        if (!found) cuda_archs.push_back(number);
    }
    LOGi << "Found cuda archs:" << cuda_archs;
#endif
}

void init() {
    // init default_random_engine
    set_seed(time(0));
    // init fused op
    op_registe({"fused","",""});
    init_cuda_devices();
    LOGv << "sizeof(Node)" << sizeof(Node);
    LOGv << "sizeof(Var)" << sizeof(Var);
    LOGv << "sizeof(Op)" << sizeof(Op);
}

void set_seed(int seed) {
    current_seed = seed;
    current_offset = 0;
    eng.reset(new std::default_random_engine(seed));
    for (auto cb : callbacks)
        cb(seed);
}

int get_seed() {
    return current_seed;
}

void add_set_seed_callback(set_seed_callback callback) {
    callbacks.push_back(callback);
    callback(current_seed);
}

std::default_random_engine* get_random_engine() { return eng.get(); }

#ifdef HAS_CUDA
bool no_cuda_error_when_free = 0;
#endif

void jt_init_subprocess() {
    #ifdef HAS_CUDA
    use_cuda = 0;
    exe.last_is_cuda = false;
    no_cuda_error_when_free = 1;
    #endif
    callbacks.clear();
}

}
