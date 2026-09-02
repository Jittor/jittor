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
#include "misc/collective_dtype.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

const char *_cudaGetErrorEnum(ncclResult_t error) {
    return ncclGetErrorString(error);
}

namespace jittor {

// The one NCCL dtype table, expanded from the canonical list in
// misc/collective_dtype.h. NCCL has no 16-bit integer type, so int16 is
// declared as a hole rather than quietly missing from the table.
static ncclDataType_t nccl_dtype_unsupported(NanoString dtype) {
    LOGf << "NCCL collectives do not support dtype" << dtype;
    return ncclFloat;
}

#define JT_NCCL_DTYPE_float16  ncclHalf
#define JT_NCCL_DTYPE_bfloat16 ncclBfloat16
#define JT_NCCL_DTYPE_float32  ncclFloat
#define JT_NCCL_DTYPE_float64  ncclFloat64
#define JT_NCCL_DTYPE_int16    nccl_dtype_unsupported(dtype)
#define JT_NCCL_DTYPE_int32    ncclInt
#define JT_NCCL_DTYPE_int64    ncclInt64
#define JT_NCCL_DTYPE_uint8    ncclUint8

ncclDataType_t nccl_dtype(NanoString dtype) {
    #define JT_NCCL_DTYPE_CASE(T) if (dtype == ns_##T) return JT_NCCL_DTYPE_##T;
    JT_COLLECTIVE_DTYPES(JT_NCCL_DTYPE_CASE)
    #undef JT_NCCL_DTYPE_CASE
    return nccl_dtype_unsupported(dtype);
}

ncclComm_t comm;
ncclUniqueId id;
int nccl_device_id = 0;
#ifdef JT_NCCL_NO_MPI
// Normally defined by mpi_wrapper.cc; provide them for the MPI-free build (the
// NCCL env/file rendezvous path / other nccl ops reference these). Only defined
// here when MPI is excluded, so there is no duplicate symbol in the MPI build.
int mpi_world_size = 1;
int mpi_world_rank = 0;
int mpi_local_size = 1;
int mpi_local_rank = 0;
bool inside_mpi = false;
bool use_device_mpi = false;
#endif


// NCCL's p2p transport treats a refused peer access as fatal, and the error it
// raises -- "unhandled cuda error" -- names neither the cause nor the cure. Name
// both before it escapes: NCCL's own explanation goes to stderr, which a test
// runner capturing output turns into a bare SIGABRT with nothing to go on.
static void init_nccl_comm(int world_size, int world_rank) {
    auto result = ncclCommInitRank(&comm, world_size, id, world_rank);
    if (result == ncclSuccess) return;
    LOGe << "ncclCommInitRank failed:" << ncclGetErrorString(result)
         << "\n  If NCCL reports that peer access is unsupported, this machine"
            " cannot do direct GPU-to-GPU transfers. Set NCCL_P2P_DISABLE=1 to"
            " route the collectives through shared memory instead."
         << "\n  Set NCCL_DEBUG=INFO for NCCL's own account of the failure.";
    checkCudaErrors(result);
}

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
        mpi_world_size = world_size;
        mpi_world_rank = world_rank;
        mpi_local_size = world_size;
        mpi_local_rank = local_rank;
        inside_mpi = true;
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
        init_nccl_comm(world_size, world_rank);
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
    init_nccl_comm(mpi_world_size, mpi_world_rank);
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
