// ***************************************************************
// Copyright (c) 2020 Jittor.
// Authors:
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>

#include "mpi_warper.h"
#include "common.h"
#include "ops/array_op.h"

char jt_mpi_err_buffer[MPI_MAX_ERROR_STRING];

void throw_mpi_error(int result, 
    char const *const func, const char *const file, int const line) {
    int resultlen;
    MPI_Error_string(result, jt_mpi_err_buffer, &resultlen);
    LOGf << "MPI error at " >> file >> ":" >> line << "code="
        >> result >> '(' >> jt_mpi_err_buffer >> ')' << func;
}

namespace jittor {


int mpi_world_size = 1;
int mpi_world_rank = 0;
int mpi_local_rank = 0;
bool inside_mpi = false;
bool mpi_enabled = false;

int _mpi_world_size() {
    return mpi_enabled ? mpi_world_size : 1;
}

int _mpi_world_rank() {
    return mpi_enabled ? mpi_world_rank : 0;
}

int _mpi_local_rank() {
    return mpi_enabled ? mpi_local_rank : 0;
}

void _mpi_broadcast(ArrayArgs&& args, int root) {
    if (!mpi_enabled) return;
    int64 size = args.dtype.dsize();
    for (auto j : args.shape)
        size *= j;
    MPI_CHECK(MPI_Bcast((void *)args.ptr, size, MPI_BYTE, root, MPI_COMM_WORLD));
}

static uint64_t getHostHash(const char* string) {
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++){
        result = ((result << 5) + result) + string[c];
    }
    return result;
}


static void getHostName(char* hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i=0; i< maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

struct mpi_initer {

mpi_initer() {
    inside_mpi = !!getenv("OMPI_COMM_WORLD_SIZE");
    if (!inside_mpi) return;
    mpi_enabled = true;
    LOGvv << "MPI init...";
    MPI_CHECK(MPI_Init(NULL, NULL));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank));

    //calculating localRank based on hostname which is used in selecting a GPU
    uint64_t hostHashs[mpi_world_rank];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[mpi_world_rank] = getHostHash(hostname);
    MPI_CHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    mpi_local_rank = 0;
    for (int p=0; p<mpi_world_size; p++) {
        if (p == mpi_world_rank) break;
        if (hostHashs[p] == hostHashs[mpi_world_rank]) mpi_local_rank++;
    }
    LOGv << "MPI init finished: local" << mpi_local_rank
        << "global" << mpi_world_rank
        << "size" << mpi_world_size;
}

~mpi_initer() {
    if (!inside_mpi) return;
    MPI_CHECK(MPI_Finalize());
}

};

static mpi_initer mpi_init;

} // jittor