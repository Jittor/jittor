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

// Optional native device RNG owner, independent of CPU and of BackendOps ABI.
struct DeviceRngOps {
    string (*get_state)(int) = nullptr;
    void (*validate_state)(const string&) = nullptr;
    void (*set_state)(int, const string&) = nullptr;
    void (*manual_seed)(int, uint64) = nullptr;
    uint64 (*initial_seed)(int) = nullptr;
};
EXTERN_LIB void register_device_rng_backend(const string& backend, const DeviceRngOps& ops);
// Snapshots flush lazy work before observing or replacing a device stream.
// @pyjt(get_device_rng_state)
string get_device_rng_state(const string& backend, int device);
// @pyjt(validate_device_rng_state)
void validate_device_rng_state(const string& backend, const string& state);
// @pyjt(set_device_rng_state)
void set_device_rng_state(const string& backend, int device, const string& state);
// @pyjt(set_device_seed)
void set_device_seed(const string& backend, int device, uint64 seed);
// @pyjt(get_device_initial_seed)
uint64 get_device_initial_seed(const string& backend, int device);

void add_set_seed_callback(set_seed_callback callback);

extern
std::default_random_engine* get_random_engine();

// things need to be clean before python exit
// @pyjt(cleanup)
void cleanup();

/**
    Wait for the queued batch and for the accelerator to finish everything
    launched so far, raising the first error either hit. `Var.sync()` and
    `jt.sync_all()` only hand the work to the device, so a kernel that faults
    after the last read would otherwise go unreported and the process would
    exit 0. Nothing is waited for on a process that never used an
    accelerator. Called once, at exit.
 */
// @pyjt(_wait_for_devices)
void wait_for_devices();

// @pyjt(jt_init_subprocess)
void jt_init_subprocess();

} // jittor
