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

// Full-width frontend entry; the legacy native int API retains its signature.
// @pyjt(_set_seed64)
void set_seed64(uint64 seed);

/**
Returns the seed of jittor random number generator.
 */
// @pyjt(get_seed)
int get_seed();

/** Opaque same-runtime CPU engine snapshot. Flushes pending eager graphs;
 * does not capture kept/replay graphs or any accelerator/Python/NumPy RNG.
 */
// @pyjt(get_cpu_rng_state)
string get_cpu_rng_state();

/** Restore a validated complete CPU engine snapshot, never seed callbacks.
 * Invalid/foreign/legacy seed-only states are rejected before flushing graphs.
 */
// @pyjt(set_cpu_rng_state)
void set_cpu_rng_state(const string& state);

// @pyjt(get_cpu_rng_seed)
int get_cpu_rng_seed();

void add_set_seed_callback(set_seed_callback callback);

extern
std::default_random_engine* get_random_engine();

// things need to be clean before python exit
// @pyjt(cleanup)
void cleanup();

// @pyjt(jt_init_subprocess)
void jt_init_subprocess();

} // jittor
