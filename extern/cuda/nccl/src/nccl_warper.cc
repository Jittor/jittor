// ***************************************************************
// Copyright (c) 2020 Jittor.
// Authors:
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "nccl_warper.h"

#ifdef HAS_CUDA
#include "event_queue.h"
#endif

const char *_cudaGetErrorEnum(ncclResult_t error) {
    return ncclGetErrorString(error);
}

namespace jittor {

ncclComm_t comm;
ncclUniqueId id;


struct nccl_initer {

nccl_initer() {
    if (mpi_world_rank == 0)
        checkCudaErrors(ncclGetUniqueId(&id));
    MPI_CHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    checkCudaErrors(cudaSetDevice(mpi_local_rank));
    #ifdef HAS_CUDA
    event_queue.run_sync([]() {
        checkCudaErrors(cudaSetDevice(mpi_local_rank));
    });
    #endif
    checkCudaErrors(ncclCommInitRank(&comm, mpi_world_size, id, mpi_world_rank));
}

~nccl_initer() {
    checkCudaErrors(ncclCommDestroy(comm));
}

};

static nccl_initer nccl_init;

} // jittor