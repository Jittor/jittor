// ***************************************************************
// Copyright (c) 2023 Jittor.
// All Rights Reserved. 
// Maintainers:
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "misc/cuda_flags.h"
#include "nccl_wrapper.h"
#include "event_queue.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

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
    // ---- MPI-free env/file rendezvous (torchrun-style; NO mpirun) ----
    // The launcher sets JT_NCCL_WORLD_SIZE/RANK/LOCAL_RANK/ROOTINFO_FILE per rank.
    // rank 0 writes ncclGetUniqueId to the file; others poll+read it; then all call
    // ncclCommInitRank. Mirrors the Ascend HCCL env path so NVIDIA multi-card DDP
    // works without MPI too (constraint: both backends supported).
    if (const char* env_ws = getenv("JT_NCCL_WORLD_SIZE")) {
        int world_size = std::atoi(env_ws);
        const char* env_r = getenv("JT_NCCL_RANK");
        const char* env_lr = getenv("JT_NCCL_LOCAL_RANK");
        const char* rf = getenv("JT_NCCL_ROOTINFO_FILE");
        int world_rank = env_r ? std::atoi(env_r) : 0;
        int local_rank = env_lr ? std::atoi(env_lr) : 0;
        nccl_device_id = device_count ? (local_rank % device_count) : 0;
        checkCudaErrors(cudaSetDevice(nccl_device_id));
        event_queue.run_sync([]() {
            checkCudaErrors(cudaSetDevice(nccl_device_id));
        });
        if (world_rank == 0) {
            checkCudaErrors(ncclGetUniqueId(&id));
            if (rf) {
                std::string tmp = std::string(rf) + ".tmp";
                FILE* f = fopen(tmp.c_str(), "wb");
                if (f) { fwrite(&id, 1, sizeof(id), f); fclose(f); rename(tmp.c_str(), rf); }
            }
        } else if (rf) {
            for (int i = 0; i < 6000; i++) { // up to ~120s
                FILE* f = fopen(rf, "rb");
                if (f) {
                    fseek(f, 0, SEEK_END); long sz = ftell(f);
                    if (sz >= (long)sizeof(id)) {
                        fseek(f, 0, SEEK_SET);
                        size_t n = fread(&id, 1, sizeof(id), f); fclose(f);
                        if (n == sizeof(id)) break;
                    } else fclose(f);
                }
                struct timespec ts{0, 20*1000*1000}; nanosleep(&ts, nullptr);
            }
        }
        use_device_mpi = true;
        checkCudaErrors(ncclCommInitRank(&comm, world_size, id, world_rank));
        LOGi << "NCCL(env) init success dev" << nccl_device_id
             << "rank" << world_rank << "/" << world_size;
        return;
    }
#ifndef JT_NCCL_NO_MPI
    // MPI bootstrap (mpirun path). Compiled out in the MPI-free env/file build.
    if (!inside_mpi) return;
    nccl_device_id = mpi_local_rank;
    if (mpi_local_rank >= device_count) {
        LOGw << "mpi_local_rank(">>mpi_local_rank>>") is larger than device_count("
            >>device_count>>")";
        nccl_device_id = nccl_device_id % device_count;
    }
    LOGv << "NCCL init in device" << nccl_device_id << "local_rank" << mpi_local_rank;
    checkCudaErrors(cudaSetDevice(nccl_device_id));
    event_queue.run_sync([]() {
        checkCudaErrors(cudaSetDevice(nccl_device_id));
    });
    if (mpi_local_size > device_count) {
        // NCCL not support multiple process on one GPU,
        // failback use MPI
        return;
    }
    use_device_mpi = true;
    if (mpi_world_rank == 0)
        checkCudaErrors(ncclGetUniqueId(&id));
    MPI_CHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    checkCudaErrors(ncclCommInitRank(&comm, mpi_world_size, id, mpi_world_rank));
#endif
}

~nccl_initer() {
    if (!get_device_count()) return;
    if (!use_device_mpi) return;   // true for both MPI and env/file rendezvous
    checkCudaErrors(ncclCommDestroy(comm));
}

};

static nccl_initer nccl_init;

} // jittor