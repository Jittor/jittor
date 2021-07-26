// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <random>
#include "common.h"

namespace jittor {

typedef void (*set_seed_callback)(int);

void init();

/**
Sets the seed of jittor random number generator. Also see @jittor.set_global_seed.

----------------

* [in] seed: a python number.

 */
// @pyjt(set_seed, seed)
void set_seed(int seed);

/**
Returns the seed of jittor random number generator.
 */
// @pyjt(get_seed)
int get_seed();

void add_set_seed_callback(set_seed_callback callback);

extern
std::default_random_engine* get_random_engine();

// things need to be clean before python exit
// @pyjt(cleanup)
void cleanup();

// @pyjt(jt_init_subprocess)
void jt_init_subprocess();

} // jittor
