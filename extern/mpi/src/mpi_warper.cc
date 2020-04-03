// ***************************************************************
// Copyright (c) 2020 Jittor.
// Authors:
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "mpi_warper.h"
#include "common.h"

char jt_mpi_err_buffer[MPI_MAX_ERROR_STRING];

void throw_mpi_error(int result, 
    char const *const func, const char *const file, int const line) {
    int resultlen;
    MPI_Error_string(result, jt_mpi_err_buffer, &resultlen);
    fprintf(stderr, "MPI error at %s:%d code=%d(%s) \"%s\" \n", 
        file, line,
        static_cast<unsigned int>(result), jt_mpi_err_buffer, func);
    throw std::runtime_error("MPI error");
}

namespace jittor {


int mpi_world_size;
int mpi_world_rank;

int _mpi_world_size() {
    return mpi_world_size;
}

int _mpi_world_rank() {
    return mpi_world_rank;
}


struct mpi_initer {

mpi_initer() {
    MPI_CHECK(MPI_Init(NULL, NULL));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank));
}

~mpi_initer() {
    MPI_CHECK(MPI_Finalize());
}

};

static mpi_initer mpi_init;

} // jittor