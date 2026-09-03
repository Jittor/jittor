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
#include "misc/file_rendezvous.h"
#include <acl/acl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
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
static HcclComm world_comm;
uint32_t hccl_device_id = 0;
static bool hccl_inited = false;

struct HcclProcessGroupState {
    HcclComm communicator = nullptr;
    vector<int> ranks;
    int local_rank = -1;
    bool owns_communicator = false;
};

static vector<HcclProcessGroupState> hccl_process_groups;

static HcclProcessGroupState& hccl_process_group(int group_id) {
    if (group_id < 0 || group_id >= (int)hccl_process_groups.size())
        LOGf << "HCCL process group" << group_id << "does not exist";
    return hccl_process_groups[group_id];
}

HcclComm hccl_process_group_comm(int group_id) {
    auto& group = hccl_process_group(group_id);
    if (group.local_rank < 0 || !group.communicator)
        LOGf << "global rank" << mpi_world_rank << "is not a member of HCCL"
             << "process group" << group_id;
    return group.communicator;
}

int hccl_process_group_size(int group_id) {
    return (int)hccl_process_group(group_id).ranks.size();
}

int hccl_process_group_rank(int group_id) {
    return hccl_process_group(group_id).local_rank;
}

static void register_hccl_world_group(int world_size, int world_rank) {
    HcclProcessGroupState group;
    group.communicator = world_comm;
    group.local_rank = world_rank;
    group.ranks.reserve(world_size);
    for (int rank=0; rank<world_size; ++rank) group.ranks.push_back(rank);
    hccl_process_groups.clear();
    hccl_process_groups.push_back(group);
    hccl_inited = true;
}

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
    // JT_HCCL_WORLD_SIZE unset means the launcher did not ask for env mode, so
    // fall through to the MPI bootstrap. But once it IS set, compile_extern has
    // already turned MPI off (use_mpi=0) and compiled this module with
    // JT_HCCL_NO_MPI, so there is nothing to fall back to: a missing rank id or
    // rootinfo path can only end as a silent single-card run. 8.09.
    if (!ws) return false;
    if (!rk || !rf)
        LOGf << "HCCL(env): JT_HCCL_WORLD_SIZE is set but"
             << (rk ? "JT_HCCL_ROOTINFO_FILE" : "JT_HCCL_RANK") << "is not."
                " Every rank needs its own rank id, and one rootinfo path on a"
                " filesystem all of them share.";

    int world_size = atoi(ws);
    int world_rank = atoi(rk);
    int local_rank = lr ? atoi(lr) : world_rank;
    mpi_world_size = world_size;
    mpi_world_rank = world_rank;
    mpi_local_rank = local_rank;
    inside_mpi = true;
    if (world_size < 1 || world_rank < 0 || world_rank >= world_size)
        LOGf << "HCCL(env): JT_HCCL_RANK=" >> world_rank
             << "is not a rank of a JT_HCCL_WORLD_SIZE=" >> world_size << "job.";

    uint32_t device_count = 0;
    ACLCHECK(aclrtGetDeviceCount(&device_count));
    if (!device_count) return false;
    hccl_device_id = local_rank % device_count;
    ACLCHECK(aclrtSetDevice(hccl_device_id));

    // Shared rendezvous helper (misc/file_rendezvous.h), which throws on a
    // failed write or a timeout. What it replaces logged at LOGe and returned
    // false for both, and this function's caller reads false as "env mode not
    // in use": a rank whose peers never appeared went on to a silent
    // single-card run instead of failing the job. 8.09.
    rendezvous_require_unlocked(world_size, "HCCL(env)");
    if (world_rank == 0) {
        HCCLCHECK(HcclGetRootInfo(&root_info));
        rendezvous_write(rf, &root_info, HCCL_ROOT_INFO_BYTES);
    } else {
        rendezvous_read(rf, &root_info, HCCL_ROOT_INFO_BYTES, world_rank,
                        "the HCCL root info");
    }
    LOGv << "HCCL(env) init dev" << hccl_device_id << "rank" << world_rank << "/" << world_size;
    HCCLCHECK(HcclCommInitRootInfo((uint32_t)world_size, &root_info,
                                     (uint32_t)world_rank, &world_comm));
    use_device_mpi = true;
    register_hccl_world_group(world_size, world_rank);
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
    rendezvous_require_unlocked(mpi_world_size, "HCCL(MPI)");
    if (mpi_world_rank == 0)
        HCCLCHECK(HcclGetRootInfo(&root_info));
    MPI_CHECK(MPI_Bcast(&root_info, HCCL_ROOT_INFO_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD));
    // Rank count / rank id for the collective communicator must come from
    // the MPI world, NOT the local device_count (which broke multi-node and
    // any run where ranks != visible devices).
    HCCLCHECK(HcclCommInitRootInfo(
        mpi_world_size, &root_info, mpi_world_rank, &world_comm));
    // NOTE: do NOT recreate aclstream here -- the acl initer already created
    // the global stream that all ops (incl. these collectives) run on.
    register_hccl_world_group(mpi_world_size, mpi_world_rank);
    LOGi << "HCCL init success on device" << hccl_device_id
         << "rank" << mpi_world_rank << "/" << mpi_world_size;
#else
    LOGw << "HCCL: JT_HCCL_* env not set and MPI disabled; cannot init.";
#endif
}

int hccl_create_process_group(vector<int> ranks) {
    if (!hccl_inited || hccl_process_groups.empty())
        LOGf << "HCCL WORLD communicator must be initialized before creating"
                " a process group";
    if (ranks.empty()) LOGf << "HCCL process group ranks cannot be empty";

    unordered_set<int> seen;
    int local_rank = -1;
    for (int i=0; i<(int)ranks.size(); ++i) {
        int rank = ranks[i];
        if (rank < 0 || rank >= mpi_world_size)
            LOGf << "HCCL process group rank" << rank << "is outside world size"
                 << mpi_world_size;
        if (!seen.insert(rank).second)
            LOGf << "HCCL process group contains duplicate rank" << rank;
        if (rank == mpi_world_rank) local_rank = i;
    }

    int group_id = (int)hccl_process_groups.size();
    HcclRootInfo group_root_info;
    const int root = ranks[0];
#ifdef JT_HCCL_NO_MPI
    const char* rootinfo_path = getenv("JT_HCCL_ROOTINFO_FILE");
    if (!rootinfo_path || !rootinfo_path[0])
        LOGf << "HCCL process groups require JT_HCCL_ROOTINFO_FILE in"
                " MPI-free mode";
    string group_path = string(rootinfo_path) + ".pg" + S(group_id);
    rendezvous_require_unlocked((int)ranks.size(), "HCCL ProcessGroup(env)");
    if (mpi_world_rank == root) {
        HCCLCHECK(HcclGetRootInfo(&group_root_info));
        if (ranks.size() > 1)
            rendezvous_write(group_path, &group_root_info,
                             HCCL_ROOT_INFO_BYTES);
    } else if (local_rank >= 0) {
        rendezvous_read(group_path, &group_root_info, HCCL_ROOT_INFO_BYTES,
                        mpi_world_rank, "the HCCL process-group root info");
    }
#else
    rendezvous_require_unlocked(mpi_world_size, "HCCL ProcessGroup(MPI)");
    if (mpi_world_rank == root)
        HCCLCHECK(HcclGetRootInfo(&group_root_info));
    MPI_CHECK(MPI_Bcast(&group_root_info, HCCL_ROOT_INFO_BYTES, MPI_CHAR,
                        root, MPI_COMM_WORLD));
#endif

    HcclProcessGroupState group;
    group.ranks = ranks;
    group.local_rank = local_rank;
    if (local_rank >= 0) {
        HCCLCHECK(HcclCommInitRootInfo(
            (uint32_t)ranks.size(), &group_root_info, (uint32_t)local_rank,
            &group.communicator));
        group.owns_communicator = true;
    }
    hccl_process_groups.push_back(group);
    return group_id;
}

struct hccl_finalizer {
    ~hccl_finalizer() {
        if (!hccl_inited) return;
        for (int i=(int)hccl_process_groups.size()-1; i>=1; --i) {
            auto& group = hccl_process_groups[i];
            if (group.owns_communicator && group.communicator)
                HCCLCHECK_PEEK(HcclCommDestroy(group.communicator));
        }
        // HCCLCHECK throws now; a throw from a destructor is std::terminate.
        HCCLCHECK_PEEK(HcclCommDestroy(world_comm));
    }
};
static hccl_finalizer hccl_finalizer;
}
