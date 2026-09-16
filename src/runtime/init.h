// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <random>
#include "core/common.h"

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

// CPU-only seeding with complete metadata, without accelerator callbacks.
// @pyjt(set_cpu_seed)
void set_cpu_seed(uint64 seed);
// @pyjt(get_cpu_initial_seed)
uint64 get_cpu_initial_seed();

// Versioned snapshots of the CPU engine, independent of accelerator streams.
// @pyjt(get_cpu_rng_state)
string get_cpu_rng_state();

// @pyjt(set_cpu_rng_state)
void set_cpu_rng_state(const string& state);

// @pyjt(get_cpu_num_threads)
int get_cpu_num_threads();

// @pyjt(set_cpu_num_threads, threads)
void set_cpu_num_threads(int threads);

void add_set_seed_callback(set_seed_callback callback);

extern
std::default_random_engine* get_random_engine();

// things need to be clean before python exit
// @pyjt(cleanup)
void cleanup();

// @pyjt(jt_init_subprocess)
void jt_init_subprocess();

} // jittor
