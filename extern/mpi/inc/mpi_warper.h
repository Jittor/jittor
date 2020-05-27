// ***************************************************************
// Copyright (c) 2020 Jittor.
// Authors:
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#define OMPI_SKIP_MPICXX
#include <mpi.h>

extern void throw_mpi_error(int result, 
    char const *const func, const char *const file, int const line);

static inline void mpi_check(int result, 
    char const *const func, const char *const file, int const line) {
    if (result != MPI_SUCCESS) {
        throw_mpi_error(result, func, file, line);
    }
}

#define MPI_CHECK(val) mpi_check((val), #val, __FILE__, __LINE__)

namespace jittor {

extern int mpi_world_size;
extern int mpi_world_rank;
extern int mpi_local_rank;
extern bool inside_mpi;

// @pyjt(world_size)
int _mpi_world_size();

// @pyjt(world_rank)
int _mpi_world_rank();

// @pyjt(local_rank)
int _mpi_local_rank();

struct ArrayArgs;

// @pyjt(broadcast)
void _mpi_broadcast(ArrayArgs&& args, int i);

} // jittor
