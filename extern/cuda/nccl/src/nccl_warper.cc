// ***************************************************************
// Copyright (c) 2020 Jittor.
// Authors:
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
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
cudaStream_t all_reduce_s;
std::map<void*, ncclComm_t> comms;
std::vector<ncclUniqueId> ids;

ncclComm_t get_comm(void* s) {
    if (!comms.count(s)) {
        ids.push_back(ncclUniqueId());
        if (mpi_world_rank == 0)
            checkCudaErrors(ncclGetUniqueId(&ids.back()));
        MPI_CHECK(MPI_Bcast((void *)&ids.back(), sizeof(ids.back()), MPI_BYTE, 0, MPI_COMM_WORLD));
        comms[s] = ncclComm_t();
        checkCudaErrors(ncclCommInitRank(&comms[s], mpi_world_size, ids.back(), mpi_world_rank));
    }
    return comms[s];
}

struct nccl_initer {

nccl_initer() {
    int device_count = get_device_count();
    if (!device_count) return;
    if (!inside_mpi) return;
    if (mpi_world_rank == 0)
        checkCudaErrors(ncclGetUniqueId(&id));
    MPI_CHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    if (mpi_local_rank >= device_count)
        LOGf << "mpi_local_rank(">>mpi_local_rank>>") is larger than device_count("
            >>device_count>>")";
    nccl_device_id = mpi_local_rank;
    LOGv << "NCCL init in device" << nccl_device_id << "local_rank" << mpi_local_rank;
    checkCudaErrors(cudaSetDevice(nccl_device_id));
    event_queue.run_sync([]() {
        checkCudaErrors(cudaSetDevice(nccl_device_id));
    });
    checkCudaErrors(cudaStreamCreateWithFlags(&all_reduce_s, cudaStreamNonBlocking));
    checkCudaErrors(ncclCommInitRank(&comm, mpi_world_size, id, mpi_world_rank));
}

~nccl_initer() {
    if (!get_device_count()) return;
    if (!inside_mpi) return;
    checkCudaErrors(ncclCommDestroy(comm));
    for (auto c : comms) {
        checkCudaErrors(ncclCommDestroy(c.second));
    }
}

};

static nccl_initer nccl_init;

} // jittor