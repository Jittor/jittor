// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>
#endif
#include <random>

#include "init.h"
#include "ops/op_register.h"
#include "var.h"
#include "op.h"

namespace jittor {

DEFINE_FLAG(vector<int>, cuda_archs, {}, "Cuda arch");

unique_ptr<std::default_random_engine> eng;

vector<set_seed_callback> callbacks;
int current_seed;

// fron fetch_op.cc
extern list<VarPtr> fetcher;
extern list<VarPtr> fetcher_to_free;

void cleanup() {
    fetcher_to_free.clear();
    fetcher.clear();
}

static void init_cuda_devices() {
#ifdef HAS_CUDA
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
    eng.reset(new std::default_random_engine(seed));
    for (auto cb : callbacks)
        cb(seed);
}

void add_set_seed_callback(set_seed_callback callback) {
    callbacks.push_back(callback);
    callback(current_seed);
}

std::default_random_engine* get_random_engine() { return eng.get(); }

}