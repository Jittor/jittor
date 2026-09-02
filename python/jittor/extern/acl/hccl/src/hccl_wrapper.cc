// ***************************************************************
// Copyright (c) 2025 Jittor.
// All Rights Reserved. 
// Maintainers:
//     Jiapeng Zhang <zjp24@mails.tsinghua.edu.cn>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include "hccl_wrapper.h"
#include "event_queue.h"
#include "acl_jittor.h"
#include "misc/collective_dtype.h"
#include <acl/acl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <ctime>

namespace jittor {

// The one HCCL dtype table, expanded from the canonical list in
// misc/collective_dtype.h. bfloat16 and int16 are declared as holes: CANN does
// define HCCL_DATA_TYPE_BFP16 / HCCL_DATA_TYPE_INT16, but no Ascend hardware
// was available to compile against, and naming an enum this build has never
// referenced is exactly the kind of unverified change this table exists to
// prevent. Adding them is a one-line change once a CANN build can be run.
static HcclDataType hccl_dtype_unsupported(NanoString dtype) {
    LOGf << "HCCL collectives do not support dtype" << dtype;
    return HcclDataType::HCCL_DATA_TYPE_FP32;
}

#define JT_HCCL_DTYPE_float16  HcclDataType::HCCL_DATA_TYPE_FP16
#define JT_HCCL_DTYPE_bfloat16 hccl_dtype_unsupported(dtype)
#define JT_HCCL_DTYPE_float32  HcclDataType::HCCL_DATA_TYPE_FP32
#define JT_HCCL_DTYPE_float64  HcclDataType::HCCL_DATA_TYPE_FP64
#define JT_HCCL_DTYPE_int16    hccl_dtype_unsupported(dtype)
#define JT_HCCL_DTYPE_int32    HcclDataType::HCCL_DATA_TYPE_INT32
#define JT_HCCL_DTYPE_int64    HcclDataType::HCCL_DATA_TYPE_INT64
#define JT_HCCL_DTYPE_uint8    HcclDataType::HCCL_DATA_TYPE_UINT8

HcclDataType hccl_dtype(NanoString dtype) {
    #define JT_HCCL_DTYPE_CASE(T) if (dtype == ns_##T) return JT_HCCL_DTYPE_##T;
    JT_COLLECTIVE_DTYPES(JT_HCCL_DTYPE_CASE)
    #undef JT_HCCL_DTYPE_CASE
    return hccl_dtype_unsupported(dtype);
}

HcclRootInfo root_info;
HcclComm comm;
uint32_t hccl_device_id = 0;
static bool hccl_inited = false;

#ifdef JT_HCCL_NO_MPI
// In MPI-free builds these globals are owned here (the mpi module isn't loaded).
int mpi_world_size = 1;
int mpi_world_rank = 0;
int mpi_local_rank = 0;
bool inside_mpi = false;
bool use_device_mpi = false;
#endif

// File-based rootinfo rendezvous, used when launched WITHOUT MPI (the conda
// OpenMPI + CANN combo crashes at MPI_Init on this box). The launcher sets:
//   JT_HCCL_WORLD_SIZE, JT_HCCL_RANK, JT_HCCL_LOCAL_RANK, JT_HCCL_ROOTINFO_FILE
// rank 0 writes its HcclRootInfo to the file; every rank reads it back, so no
// MPI_Bcast is needed. Returns true if env-driven mode was used.
static bool hccl_init_envfile() {
    const char* ws = getenv("JT_HCCL_WORLD_SIZE");
    const char* rk = getenv("JT_HCCL_RANK");
    const char* lr = getenv("JT_HCCL_LOCAL_RANK");
    const char* rf = getenv("JT_HCCL_ROOTINFO_FILE");
    if (!ws || !rk || !rf) return false;

    int world_size = atoi(ws);
    int world_rank = atoi(rk);
    int local_rank = lr ? atoi(lr) : world_rank;

    uint32_t device_count = 0;
    ACLCHECK(aclrtGetDeviceCount(&device_count));
    if (!device_count) return false;
    hccl_device_id = local_rank % device_count;
    ACLCHECK(aclrtSetDevice(hccl_device_id));

    string tmp_path = string(rf) + ".tmp";
    if (world_rank == 0) {
        HCCLCHECK(HcclGetRootInfo(&root_info));
        // write atomically: tmp then rename
        FILE* f = fopen(tmp_path.c_str(), "wb");
        if (!f) { LOGe << "cannot open rootinfo tmp" << tmp_path; return false; }
        fwrite(&root_info, 1, HCCL_ROOT_INFO_BYTES, f);
        fclose(f);
        rename(tmp_path.c_str(), rf);
    } else {
        // poll for the rootinfo file to appear and be full-size
        for (int i = 0; i < 6000; i++) { // up to ~120s
            FILE* f = fopen(rf, "rb");
            if (f) {
                fseek(f, 0, SEEK_END);
                long sz = ftell(f);
                if (sz >= (long)HCCL_ROOT_INFO_BYTES) {
                    fseek(f, 0, SEEK_SET);
                    size_t n = fread(&root_info, 1, HCCL_ROOT_INFO_BYTES, f);
                    fclose(f);
                    if (n == HCCL_ROOT_INFO_BYTES) break;
                } else fclose(f);
            }
            struct timespec ts{0, 20*1000*1000}; // 20ms
            nanosleep(&ts, nullptr);
            if (i == 5999) { LOGe << "timeout waiting for rootinfo file" << rf; return false; }
        }
    }
    LOGv << "HCCL(env) init dev" << hccl_device_id << "rank" << world_rank << "/" << world_size;
    HCCLCHECK(HcclCommInitRootInfo((uint32_t)world_size, &root_info,
                                     (uint32_t)world_rank, &comm));
    use_device_mpi = true;
    hccl_inited = true;
    LOGi << "HCCL(env) init success dev" << hccl_device_id
         << "rank" << world_rank << "/" << world_size;
    return true;
}

// Explicit HCCL communicator init. Decoupled from module load: doing a
// blocking collective (HcclCommInitRootInfo) inside a static constructor at
// dlopen time hung the `import jittor` path. Instead Python calls this once,
// after import completes and the device is fully live.
void hccl_init() {
    if (hccl_inited) return;
    // Prefer the MPI-free env/file rendezvous if the launcher set it up.
    if (hccl_init_envfile()) return;
#ifndef JT_HCCL_NO_MPI
    uint32_t device_count = 0;
    ACLCHECK(aclrtGetDeviceCount(&device_count));
    if (!device_count) return;
    if (!inside_mpi) return;
    hccl_device_id = mpi_local_rank;
    if (mpi_local_rank >= device_count) {
        LOGw << "mpi_local_rank(">>mpi_local_rank>>") is larger than device_count("
            >>device_count>>")";
        hccl_device_id = hccl_device_id % device_count;
    }
    LOGv << "HCCL init on device" << hccl_device_id << "local_rank" << mpi_local_rank
         << "world" << mpi_world_rank << "/" << mpi_world_size;
    // The acl initer already bound this process to hccl_device_id via
    // aclrtSetDevice(local_rank); re-assert it to be safe.
    ACLCHECK(aclrtSetDevice(hccl_device_id));
    use_device_mpi = true;
    if (mpi_world_rank == 0)
        HCCLCHECK(HcclGetRootInfo(&root_info));
    MPI_CHECK(MPI_Bcast(&root_info, HCCL_ROOT_INFO_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD));
    // Rank count / rank id for the collective communicator must come from
    // the MPI world, NOT the local device_count (which broke multi-node and
    // any run where ranks != visible devices).
    HCCLCHECK(HcclCommInitRootInfo(mpi_world_size, &root_info, mpi_world_rank, &comm));
    // NOTE: do NOT recreate aclstream here -- the acl initer already created
    // the global stream that all ops (incl. these collectives) run on.
    hccl_inited = true;
    LOGi << "HCCL init success on device" << hccl_device_id
         << "rank" << mpi_world_rank << "/" << mpi_world_size;
#else
    LOGw << "HCCL: JT_HCCL_* env not set and MPI disabled; cannot init.";
#endif
}

struct hccl_finalizer {
    ~hccl_finalizer() {
        if (!hccl_inited) return;
        // HCCLCHECK throws now; a throw from a destructor is std::terminate.
        HCCLCHECK_PEEK(HcclCommDestroy(comm));
    }
};
static hccl_finalizer hccl_finalizer;
}