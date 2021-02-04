// ***************************************************************
// Copyright (c) 2021 Jittor.
// All Rights Reserved. 
// Maintainers:
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "misc/cuda_flags.h"
#include "nccl_warper.h"
#include "event_queue.h"

const char *_cudaGetErrorEnum(ncclResult_t error) {
    return ncclGetErrorString(error);
}

namespace jittor {

ncclComm_t comm;
ncclUniqueId id;
int nccl_device_id = 0;


struct nccl_initer {

nccl_initer() {
    int device_count = get_device_count();
    if (!device_count) return;
    if (!inside_mpi) return;
    if (mpi_local_rank >= device_count)
        LOGf << "mpi_local_rank(">>mpi_local_rank>>") is larger than device_count("
            >>device_count>>")";
    nccl_device_id = mpi_local_rank;
    LOGv << "NCCL init in device" << nccl_device_id << "local_rank" << mpi_local_rank;
    checkCudaErrors(cudaSetDevice(nccl_device_id));
    event_queue.run_sync([]() {
        checkCudaErrors(cudaSetDevice(nccl_device_id));
    });
    if (mpi_world_rank == 0)
        checkCudaErrors(ncclGetUniqueId(&id));
    MPI_CHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    checkCudaErrors(ncclCommInitRank(&comm, mpi_world_size, id, mpi_world_rank));
}

~nccl_initer() {
    if (!get_device_count()) return;
    if (!inside_mpi) return;
    checkCudaErrors(ncclCommDestroy(comm));
}

};

static nccl_initer nccl_init;

} // jittor