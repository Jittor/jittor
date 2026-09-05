// ***************************************************************
// Copyright (c) 2023 Jittor.
// All Rights Reserved. 
// Maintainers:
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#define OMPI_SKIP_MPICXX
#include <common.h>
#include <mpi.h>
#include "var_holder.h"
#include "type/nano_string.h"

extern void throw_mpi_error(int result, 
    char const *const func, const char *const file, int const line);

static inline void mpi_check(int result, 
    char const *const func, const char *const file, int const line) {
    if (result != MPI_SUCCESS) {
        throw_mpi_error(result, func, file, line);
    }
}

#define MPI_CHECK(val) mpi_check((val), #val, __FILE__, __LINE__)

// Reporting-only variant for destructor and teardown paths. throw_mpi_error
// LOGf's, i.e. throws, and a throw out of a noexcept destructor terminates the
// process -- which is how an MPI_Finalize that merely came too late turned
// into an abort that hid whatever the rank was actually failing on.
extern void report_mpi_error(int result,
    char const *const func, const char *const file, int const line);

static inline void mpi_peek(int result,
    char const *const func, const char *const file, int const line) {
    if (result != MPI_SUCCESS) {
        report_mpi_error(result, func, file, line);
    }
}

#define MPI_PEEK(val) mpi_peek((val), #val, __FILE__, __LINE__)

namespace jittor {

EXTERN_LIB int mpi_world_size;
EXTERN_LIB int mpi_world_rank;
EXTERN_LIB int mpi_local_size;
EXTERN_LIB int mpi_local_rank;
EXTERN_LIB bool inside_mpi;
EXTERN_LIB bool mpi_enabled;
EXTERN_LIB bool use_device_mpi;

// Finalizes MPI, reporting a failure instead of raising. Idempotent.
void mpi_shutdown();

/**
Map a jittor dtype to the MPI datatype used to send it, and to the MPI
reduction operator that implements `add` for it.

These are the only mapping tables for MPI; the operator files and the
`var_*` helpers below all go through them. See `misc/collective_dtype.h`
for why the per-operator copies were removed. Both raise (LOGf) on a dtype
MPI cannot carry, instead of expanding to nothing.
*/
MPI_Datatype mpi_dtype(NanoString dtype);
MPI_Op mpi_add_op(NanoString dtype);

/**
Return number of MPI nodes.
*/
// @pyjt(world_size)
int _mpi_world_size();

/**
Return global ID of this MPI node.
*/
// @pyjt(world_rank)
int _mpi_world_rank();

/**
Return local ID of this MPI node.
*/
// @pyjt(local_rank)
int _mpi_local_rank();

/**
 Set MPI state, enable or disable, if disabled, all mpi operators
 have no affect.
*/
// @pyjt(set_state)
inline void _mpi_set_state(bool enable) { mpi_enabled = enable; }

/**
 Get MPI state, enable or disable.
*/
// @pyjt(get_state)
inline int _mpi_get_state() { return mpi_enabled; }

struct ArrayArgs;

/**

Use jt.Module.mpi_param_broadcast(root=0) to broadcast all moudule parameters of this module in [root] MPI node to all MPI nodes.

This operation has no gradient, and the input parameter type is numpy array.
*/
// @pyjt(broadcast)
void _mpi_broadcast(ArrayArgs&& args, int root);

// @pyjt(var_broadcast)
void var_broadcast(VarHolder* x, int root=0);

// @pyjt(var_reduce)
void var_reduce(VarHolder* x, int root=0);

// @pyjt(var_all_reduce)
void var_all_reduce(VarHolder* x);

// @pyjt(mpi_barrier)
void mpi_barrier();

} // jittor
