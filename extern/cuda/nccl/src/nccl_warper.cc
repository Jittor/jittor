// ***************************************************************
// Copyright (c) 2020 Jittor.
// Authors:
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "nccl_warper.h"
#include "event_queue.h"

const char *_cudaGetErrorEnum(ncclResult_t error) {
    return ncclGetErrorString(error);
}

namespace jittor {

ncclComm_t comm;
ncclUniqueId id;


struct nccl_initer {

nccl_initer() {
    if (!get_device_count()) return;
    if (mpi_world_rank == 0)
        checkCudaErrors(ncclGetUniqueId(&id));
    MPI_CHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    LOGv << "NCCL init in device" << mpi_local_rank;
    checkCudaErrors(cudaSetDevice(mpi_local_rank));
    event_queue.run_sync([]() {
        checkCudaErrors(cudaSetDevice(mpi_local_rank));
    });
    checkCudaErrors(ncclCommInitRank(&comm, mpi_world_size, id, mpi_world_rank));
}

~nccl_initer() {
    if (!get_device_count()) return;
    checkCudaErrors(ncclCommDestroy(comm));
}

};

static nccl_initer nccl_init;

} // jittor