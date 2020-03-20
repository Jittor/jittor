// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <random>

#include "init.h"
#include "ops/op_register.h"

namespace jittor {

unique_ptr<std::default_random_engine> eng;

vector<set_seed_callback> callbacks;
int current_seed;

void init() {
    // init default_random_engine
    set_seed(time(0));
    // init fused op
    op_registe({"fused","",""});
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