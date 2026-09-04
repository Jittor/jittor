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
#include "var.h"
#include "mem/allocator.h"
#include "misc/collective_dtype.h"
#include "misc/file_rendezvous.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <thread>
#include <unordered_set>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

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

ncclUniqueId id;
int nccl_device_id = 0;
static vector<int> nccl_pending_unique_id;

// ---------------------------------------------------------------- 8.02
// Bucket scope state. All of it is per-process and only touched from the
// executor thread and the Python calls that bracket a bucket, in that order.

// Between nccl_bucket_begin() and nccl_bucket_end().
static bool nccl_bucket_open = false;
// ncclGroupStart() was issued and still needs its ncclGroupEnd().
static bool nccl_group_opened = false;
// This bucket is allowed to leave the join outstanding. Cleared the moment any
// block cannot be pinned, because holding the block is what makes running
// ahead of the collective safe.
static bool nccl_bucket_defer = false;
// Device the collectives in this bucket ran on. nccl_bucket_end() records the
// done event there, and it is not necessarily whatever device is current when
// Python closes the scope.
static int nccl_bucket_device = -1;

static bool nccl_hold(int device, Var* v) {
    if (!v || !v->mem_ptr || !v->allocator) return false;
    if (!v->allocator->can_share()) return false;
    // Take the extra reference first; cuda_side_stream_hold_block owns it and
    // drops it when the join resolves.
    v->allocator->share_with(v->size, v->allocation);
    return cuda_side_stream_hold_block(
        CUDA_COMMUNICATION_STREAM, device,
        v->mem_ptr, v->allocation, v->size, v->allocator);
}

cudaStream_t nccl_stream_begin() {
    int device = current_device();
    cuda_side_stream_wait_default(
        CUDA_COMMUNICATION_STREAM, device, device);
    if (nccl_bucket_open) {
        nccl_bucket_device = device;
        if (!nccl_group_opened) {
            checkCudaErrors(ncclGroupStart());
            nccl_group_opened = true;
        }
    }
    return cuda_side_stream(CUDA_COMMUNICATION_STREAM, device);
}

void nccl_stream_end(Var* x, Var* y) {
    int device = current_device();
    if (!nccl_bucket_open) {
        cuda_default_stream_wait_side(
            CUDA_COMMUNICATION_STREAM, device, device);
        return;
    }
    // Inside a bucket the join belongs to nccl_bucket_end(), for both join
    // policies. Joining here would record a done event on a communication
    // stream that has nothing on it yet -- ncclGroupEnd() has not run, so none
    // of the collectives are submitted -- and order nothing at all. That is
    // not a missed optimisation but a silent race, and it is exactly what the
    // first version of this did: the profiler showed compute overlapping the
    // collective even with defer_join=False, which should be impossible.
    //
    // Grouping also means both buffers have to stay reserved until
    // ncclGroupEnd(), whatever the join policy: NCCL captured the pointers,
    // and the default stream is free to run ahead and have the allocator hand
    // those blocks to something else in the meantime.
    if (nccl_hold(device, x) && nccl_hold(device, y))
        return;
    // An allocator that cannot hand one block to two owners leaves no way to
    // keep the default stream off it. Submit what the group has so far and
    // order it now, rather than keep grouping over a buffer that may be
    // recycled underneath us. Correctness over both grouping and overlap --
    // the same trade fetch_op makes.
    LOGw << "nccl bucket cannot reserve its buffers; submitting this"
         << "collective without grouping or overlap";
    if (nccl_group_opened) {
        checkCudaErrors(ncclGroupEnd());
        nccl_group_opened = false;
    }
    nccl_bucket_defer = false;
    // Joins and releases whatever the bucket had already reserved; the join
    // covers them because the communication stream runs in order.
    cuda_side_stream_resolve_join(CUDA_COMMUNICATION_STREAM);
    cuda_default_stream_wait_side(
        CUDA_COMMUNICATION_STREAM, device, device);
}

void nccl_bucket_begin(bool defer_join) {
    if (nccl_bucket_open)
        LOGf << "nccl_bucket_begin inside another bucket";
    if (cuda_side_stream_any_join_pending(CUDA_COMMUNICATION_STREAM))
        LOGf << "nccl_bucket_begin with a join still outstanding:"
             << "call nccl_comm_wait() before opening the next bucket";
    nccl_bucket_open = true;
    nccl_bucket_defer = defer_join;
    nccl_bucket_device = -1;
    // ncclGroupStart is issued lazily, at the first collective, so a bucket
    // that ends up containing none (the graph was already synced, say) does
    // not leave a group open.
    nccl_group_opened = false;
}

void nccl_bucket_end() {
    if (!nccl_bucket_open)
        LOGf << "nccl_bucket_end without nccl_bucket_begin";
    nccl_bucket_open = false;
    if (nccl_group_opened) {
        // Everything the bucket recorded is submitted to the communication
        // stream here, as one group. Only after this is there anything on the
        // stream for a join to order against.
        checkCudaErrors(ncclGroupEnd());
        nccl_group_opened = false;
    }
    bool defer = nccl_bucket_defer;
    nccl_bucket_defer = false;
    if (nccl_bucket_device < 0) return;   // the bucket held no collectives
    if (defer) {
        cuda_side_stream_defer_join(
            CUDA_COMMUNICATION_STREAM, nccl_bucket_device);
        return;
    }
    // Synchronous bucket: group the launches but keep the old ordering, so the
    // default stream is behind the collectives when the scope closes. Also
    // releases the blocks the grouping had to reserve.
    cuda_side_stream_resolve_join(CUDA_COMMUNICATION_STREAM);
}

bool nccl_comm_wait() {
    return cuda_side_stream_resolve_join(CUDA_COMMUNICATION_STREAM) > 0;
}
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


struct NcclProcessGroupState {
    ncclComm_t communicator = nullptr;
    vector<int> ranks;
    int local_rank = -1;
    bool owns_communicator = false;
};

static vector<NcclProcessGroupState> nccl_process_groups;
static bool nccl_comm_created = false;

static NcclProcessGroupState& nccl_process_group(int group_id) {
    if (group_id < 0 || group_id >= (int)nccl_process_groups.size())
        LOGf << "NCCL process group" << group_id << "does not exist";
    return nccl_process_groups[group_id];
}

ncclComm_t nccl_process_group_comm(int group_id) {
    auto& group = nccl_process_group(group_id);
    if (group.local_rank < 0 || !group.communicator)
        LOGf << "global rank" << mpi_world_rank << "is not a member of NCCL"
             << "process group" << group_id;
    return group.communicator;
}

int nccl_process_group_size(int group_id) {
    return (int)nccl_process_group(group_id).ranks.size();
}

int nccl_process_group_rank(int group_id) {
    return nccl_process_group(group_id).local_rank;
}

// NCCL's p2p transport treats a refused peer access as fatal, and the error it
// raises -- "unhandled cuda error" -- names neither the cause nor the cure. Name
// both before it escapes: NCCL's own explanation goes to stderr, which a test
// runner capturing output turns into a bare SIGABRT with nothing to go on.
static ncclComm_t init_nccl_comm(int world_size, int world_rank,
                                 const ncclUniqueId& unique_id) {
    ncclComm_t communicator = nullptr;
    auto result = ncclCommInitRank(
        &communicator, world_size, unique_id, world_rank);
    if (result == ncclSuccess) return communicator;
    LOGe << "ncclCommInitRank failed:" << ncclGetErrorString(result)
         << "\n  If NCCL reports that peer access is unsupported, this machine"
            " cannot do direct GPU-to-GPU transfers. Set NCCL_P2P_DISABLE=1 to"
            " route the collectives through shared memory instead."
         << "\n  Set NCCL_DEBUG=INFO for NCCL's own account of the failure.";
    checkCudaErrors(result);
    return nullptr;
}


// ---------------------------------------------------------------------------
// Watchdog: notice that a peer has died, instead of waiting for it forever.
//
// When a rank dies, the survivors stay parked in the collective they were
// running -- no message, no exit, GPUs at 100%. Measured on two 4090s: kill
// rank 1 mid-all_reduce and rank 0 was still sitting there two minutes later
// with nothing printed since the step before. Neither ncclCommGetAsyncError
// nor ncclCommAbort appeared anywhere in this tree.
//
// Two detectors, because one is not enough:
//
// * ncclCommGetAsyncError. This is the documented way, and it is what catches
//   a network transport dropping a connection. It did NOT catch the case
//   above: with both ranks on one host and no peer access (NCCL_P2P_DISABLE=1,
//   which this box needs), the transport is shared memory, there is no socket
//   to break, and the communicator stays in ncclSuccess while the kernel spins.
// * Peer heartbeats. Every rank touches <rootinfo>.hb<rank> once per interval
//   in this same thread; a rank whose file has not changed for
//   JT_NCCL_WATCHDOG_STALE_S is gone. This works whatever the transport is,
//   and unlike the async error it can say WHICH rank stopped -- the thing the
//   operator actually needs, and something NCCL's API never reports. It needs
//   the shared-file rendezvous, which already requires storage every rank can
//   see; under mpirun there is no such file and the launcher kills the job
//   itself, so only the async check runs there.
//
// Staleness is judged by "this file has not changed in N seconds of MY clock",
// never by comparing a file's mtime to my clock, so a shared filesystem whose
// clock differs from this host's does not make every peer look dead.
//
// Either detector then calls ncclCommAbort, which raises the flag NCCL's
// device kernels poll: the stuck collective returns an error instead of
// spinning, and the rank dies the ordinary way, through checkCudaErrors, with
// a traceback naming the operator.
//
// JT_NCCL_WATCHDOG_INTERVAL_S: poll/heartbeat period, default 5s; <= 0
// disables the thread. JT_NCCL_WATCHDOG_STALE_S: how long a peer's heartbeat
// may stand still before it counts as dead, default 4x the interval.
// JT_NCCL_WATCHDOG_GRACE_S: how long to wait after the abort for this rank to
// die on its own before the thread ends the process, default 30s; <= 0 means
// never, which is the pre-8.09 behavior of hanging indefinitely.
// ---------------------------------------------------------------------------
static std::thread nccl_watchdog_thread;
static std::atomic<bool> nccl_watchdog_stop{false};
static pid_t nccl_watchdog_pid = 0;

static double nccl_env_seconds(const char* name, double fallback) {
    if (const char* env = getenv(name)) {
        char* end = nullptr;
        double v = strtod(env, &end);
        if (end != env) return v;
        LOGw << "NCCL:" << name << "is not a number (" >> env >> "), using"
             << fallback;
    }
    return fallback;
}

// Sleep up to `seconds`, waking early if asked to stop. False means stop.
static bool nccl_watchdog_sleep(double seconds) {
    const double slice = 0.05;
    for (double slept = 0; slept < seconds; slept += slice) {
        if (nccl_watchdog_stop.load(std::memory_order_relaxed)) return false;
        struct timespec ts{0, (long)(slice * 1e9)};
        nanosleep(&ts, nullptr);
    }
    return !nccl_watchdog_stop.load(std::memory_order_relaxed);
}

static string nccl_heartbeat_path(const string& rootinfo, int rank) {
    return rootinfo + ".hb" + S(rank);
}

// Touch our own heartbeat. Creating it when absent covers the first round and
// an operator who cleaned the directory mid-run.
static void nccl_heartbeat_touch(const string& path) {
    if (utimensat(AT_FDCWD, path.c_str(), nullptr, 0) == 0) return;
    FILE* f = fopen(path.c_str(), "wb");
    if (f) { fclose(f); return; }
    static bool complained = false;
    if (!complained) {
        complained = true;
        LOGw << "NCCL watchdog: cannot write heartbeat" << path >> ":"
             << strerror(errno)
             << "-- peers will consider this rank dead. Only the async-error"
                " check is left here.";
    }
}

// One peer's heartbeat as this rank has observed it. `stalled_for` is measured
// on the local steady clock, so the shared filesystem's clock is never
// compared with this host's.
struct nccl_peer_beat {
    bool seen = false;
    struct timespec mtime {0, 0};
    std::chrono::steady_clock::time_point last_change;
};

static void nccl_watchdog_die(int world_rank, double grace, const string& why) {
    LOGe << "NCCL watchdog: rank" << world_rank >> ":" << why
         << "\n  Every surviving rank would otherwise wait for the missing one"
            " indefinitely -- no error, no exit, GPUs at 100%. Aborting the"
            " communicator, so the collective in flight fails instead of"
            " spinning."
         << "\n  The first failing rank's log is the cause; this message is"
            " only the consequence."
         << "\n  JT_NCCL_WATCHDOG_INTERVAL_S=0 disables this check.";
    auto& world = nccl_process_group(0);
    ncclCommAbort(world.communicator);
    world.communicator = nullptr;
    // An aborted communicator must not then be destroyed; tell nccl_shutdown
    // there is nothing left to tear down.
    nccl_comm_created = false;
    if (grace <= 0) return;
    // The abort should let this rank die through its own error path. If it
    // does not, it is stuck somewhere NCCL cannot reach, and the choice is
    // between ending it and hanging forever -- hanging forever is the thing
    // being removed.
    if (!nccl_watchdog_sleep(grace)) return;
    LOGe << "NCCL watchdog: rank" << world_rank << "still alive" << grace
         << "s after the abort; exiting the process.";
    flush_log();
    std::_Exit(1);
}

static void nccl_watchdog_loop(double interval, double stale, double grace,
                               int world_size, int world_rank, string rootinfo) {
    const bool beats = rootinfo.size();
    const string own = beats ? nccl_heartbeat_path(rootinfo, world_rank) : "";
    vector<nccl_peer_beat> peers(beats ? world_size : 0);
    auto now = std::chrono::steady_clock::now();
    for (auto& p : peers) p.last_change = now;

    while (nccl_watchdog_sleep(interval)) {
        if (beats) nccl_heartbeat_touch(own);

        ncclResult_t async = ncclSuccess;
        // A failure of the query itself is not a peer failure; only the stop
        // flag ends this loop.
        if (ncclCommGetAsyncError(nccl_process_group_comm(0), &async) == ncclSuccess &&
            async != ncclSuccess && async != ncclInProgress) {
            nccl_watchdog_die(world_rank, grace,
                string("the communicator reports ") + ncclGetErrorString(async) +
                ", which means a peer connection failed");
            return;
        }
        if (!beats) continue;

        now = std::chrono::steady_clock::now();
        string dead;
        for (int r = 0; r < world_size; r++) {
            if (r == world_rank) continue;
            auto& p = peers[r];
            struct stat st;
            if (stat(nccl_heartbeat_path(rootinfo, r).c_str(), &st) == 0) {
                if (!p.seen || st.st_mtim.tv_sec != p.mtime.tv_sec ||
                        st.st_mtim.tv_nsec != p.mtime.tv_nsec) {
                    p.seen = true;
                    p.mtime = st.st_mtim;
                    p.last_change = now;
                    continue;
                }
            }
            double stalled = std::chrono::duration<double>(
                now - p.last_change).count();
            if (stalled > stale)
                dead += (dead.size() ? ", " : "") + S(r) +
                        " (silent for " + S((int)stalled) + "s)";
        }
        if (dead.size()) {
            nccl_watchdog_die(world_rank, grace,
                "rank(s) " + dead + " stopped updating their heartbeat, so"
                " they are gone");
            return;
        }
    }
}

static string nccl_watchdog_own_beat;

static void nccl_watchdog_start(int world_size, int world_rank,
                                const char* rootinfo) {
    if (world_size <= 1) return;   // no peer to lose
    double interval = nccl_env_seconds("JT_NCCL_WATCHDOG_INTERVAL_S", 5.0);
    if (interval <= 0) {
        LOGv << "NCCL watchdog disabled by JT_NCCL_WATCHDOG_INTERVAL_S";
        return;
    }
    double stale = nccl_env_seconds("JT_NCCL_WATCHDOG_STALE_S", 4 * interval);
    double grace = nccl_env_seconds("JT_NCCL_WATCHDOG_GRACE_S", 30.0);
    string rf = (rootinfo && rootinfo[0]) ? string(rootinfo) : string();
    if (rf.size()) nccl_watchdog_own_beat = nccl_heartbeat_path(rf, world_rank);
    nccl_watchdog_pid = getpid();
    nccl_watchdog_thread = std::thread(nccl_watchdog_loop, interval, stale,
                                       grace, world_size, world_rank, rf);
    LOGv << "NCCL watchdog: every" << interval >> "s, peers stale after"
         << stale >> "s, heartbeats" << (rf.size() ? "on" : "off");
}

static void nccl_watchdog_join() {
    if (!nccl_watchdog_thread.joinable()) return;
    if (getpid() != nccl_watchdog_pid) {
        // A fork()ed child (jittor's dataset workers fork). Only the calling
        // thread survives a fork, so this handle names a thread that does not
        // exist here: joining it is undefined, and ~std::thread on a joinable
        // handle is std::terminate. Detach is neither.
        nccl_watchdog_thread.detach();
        return;
    }
    nccl_watchdog_stop.store(true, std::memory_order_relaxed);
    nccl_watchdog_thread.join();
    // Leaving a stale heartbeat behind would make the next job started against
    // the same rootinfo path see a peer that is not there.
    if (nccl_watchdog_own_beat.size()) {
        remove(nccl_watchdog_own_beat.c_str());
        nccl_watchdog_own_beat.clear();
    }
}

// See cublas_shutdown: report, never raise, and idempotent. A rank that is
// already failing is exactly the one whose communicator will refuse to be
// destroyed, and it must still be able to print why it failed.
void nccl_shutdown() {
    // The watchdog polls the WORLD communicator; it has to be gone before any
    // process-group communicator is destroyed.
    nccl_watchdog_join();
    for (int i=(int)nccl_process_groups.size()-1; i>=1; --i) {
        auto& group = nccl_process_groups[i];
        if (group.owns_communicator && group.communicator) {
            peekCudaErrorsAlways(ncclCommDestroy(group.communicator));
            group.communicator = nullptr;
        }
    }
    if (!nccl_comm_created) {
        nccl_process_groups.clear();
        return;
    }
    nccl_comm_created = false;
    auto& world = nccl_process_group(0);
    if (world.communicator)
        peekCudaErrorsAlways(ncclCommDestroy(world.communicator));
    world.communicator = nullptr;
    nccl_process_groups.clear();
}

static void register_nccl_world_group(ncclComm_t communicator,
                                      int world_size, int world_rank) {
    NcclProcessGroupState group;
    group.communicator = communicator;
    group.local_rank = world_rank;
    group.ranks.reserve(world_size);
    for (int rank=0; rank<world_size; ++rank) group.ranks.push_back(rank);
    nccl_process_groups.clear();
    nccl_process_groups.push_back(group);
    nccl_comm_created = true;
}

int nccl_create_process_group(vector<int> ranks) {
    if (!nccl_comm_created || nccl_process_groups.empty())
        LOGf << "NCCL WORLD communicator must be initialized before creating"
                " a process group";
    if (ranks.empty()) LOGf << "NCCL process group ranks cannot be empty";

    unordered_set<int> seen;
    int local_rank = -1;
    for (int i=0; i<(int)ranks.size(); ++i) {
        int rank = ranks[i];
        if (rank < 0 || rank >= mpi_world_size)
            LOGf << "NCCL process group rank" << rank << "is outside world size"
                 << mpi_world_size;
        if (!seen.insert(rank).second)
            LOGf << "NCCL process group contains duplicate rank" << rank;
        if (rank == mpi_world_rank) local_rank = i;
    }

    int group_id = (int)nccl_process_groups.size();
    ncclUniqueId group_unique_id;
    const int root = ranks[0];
#ifdef JT_NCCL_NO_MPI
    const char* rootinfo = getenv("JT_NCCL_ROOTINFO_FILE");
    if (!rootinfo || !rootinfo[0])
        LOGf << "NCCL process groups require JT_NCCL_ROOTINFO_FILE in"
                " MPI-free mode";
    string group_path = string(rootinfo) + ".pg" + S(group_id);
    rendezvous_require_unlocked((int)ranks.size(), "NCCL ProcessGroup(env)");
    if (mpi_world_rank == root) {
        checkCudaErrors(ncclGetUniqueId(&group_unique_id));
        if (ranks.size() > 1)
            rendezvous_write(group_path, &group_unique_id,
                             sizeof(group_unique_id));
    } else if (local_rank >= 0) {
        rendezvous_read(group_path, &group_unique_id, sizeof(group_unique_id),
                        mpi_world_rank, "the NCCL process-group unique id");
    }
#else
    // new_group is collective over WORLD even for non-members. Broadcasting
    // the new id on MPI_COMM_WORLD gives every rank the same group sequence;
    // only members then enter ncclCommInitRank.
    rendezvous_require_unlocked(mpi_world_size, "NCCL ProcessGroup(MPI)");
    if (mpi_world_rank == root)
        checkCudaErrors(ncclGetUniqueId(&group_unique_id));
    MPI_CHECK(MPI_Bcast((void*)&group_unique_id, sizeof(group_unique_id),
                        MPI_BYTE, root, MPI_COMM_WORLD));
#endif

    NcclProcessGroupState group;
    group.ranks = ranks;
    group.local_rank = local_rank;
    if (local_rank >= 0) {
        group.communicator = init_nccl_comm(
            (int)ranks.size(), local_rank, group_unique_id);
        group.owns_communicator = true;
    }
    nccl_process_groups.push_back(group);
    return group_id;
}

// See nccl_wrapper.h for why this is a function called from Python rather than
// the static constructor it used to be. nccl_comm_created (set by
// init_nccl_comm, cleared by nccl_shutdown) is the "already done" flag, so
// there is one piece of state rather than two that can disagree.
void nccl_init() {
    if (nccl_comm_created) return;
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
        // Through the placement runtime rather than straight to the driver, so
        // the memory pools and the library handles agree with it about which
        // device this rank is on.
        set_current_device(nccl_device_id);
        event_queue.run_sync([]() {
            checkCudaErrors(cudaSetDevice(nccl_device_id));
        });
        // Rendezvous through the shared helper (misc/file_rendezvous.h), which
        // fails loudly on timeout. What this replaces did not: non-zero ranks
        // polled for a hardcoded 121 s and then fell through WITHOUT CHECKING
        // WHETHER THEY HAD READ ANYTHING, handing the still-zero id to
        // ncclCommInitRank. And when JT_NCCL_ROOTINFO_FILE was unset there was
        // no wait at all: the uninitialized id went straight in. Whether that
        // ends in a permanent hang or in NCCL's "internal error - please report
        // this issue to the NCCL developers" depends on the NCCL build; neither
        // names the rank that never showed up, which is the one thing the
        // operator needs. 8.09.
        if (world_size < 1 || world_rank < 0 || world_rank >= world_size)
            LOGf << "NCCL(env): JT_NCCL_RANK=" >> world_rank
                 << "is not a rank of a JT_NCCL_WORLD_SIZE=" >> world_size << "job.";
        rendezvous_require_unlocked(world_size, "NCCL(env)");
        bool supplied_unique_id = !nccl_pending_unique_id.empty();
        if (supplied_unique_id) {
            if (nccl_pending_unique_id.size() != sizeof(id))
                LOGf << "NCCL store returned" << nccl_pending_unique_id.size()
                     << "unique-id bytes; expected" << sizeof(id);
            for (int i=0; i<(int)nccl_pending_unique_id.size(); ++i) {
                int value = nccl_pending_unique_id[i];
                if (value < 0 || value > 255)
                    LOGf << "NCCL store unique-id byte" << i << "is" << value
                         << "instead of an unsigned byte";
                ((unsigned char*)&id)[i] = (unsigned char)value;
            }
            nccl_pending_unique_id.clear();
        }
        if (!supplied_unique_id && world_size > 1 && (!rf || !rf[0]))
            LOGf << "NCCL(env): JT_NCCL_WORLD_SIZE=" >> world_size
                 << "but JT_NCCL_ROOTINFO_FILE is not set. Every rank needs the"
                    " same path, on a filesystem all of them share, to exchange"
                    " the NCCL unique id.";
        if (!supplied_unique_id && world_rank == 0) {
            checkCudaErrors(ncclGetUniqueId(&id));
            if (rf && rf[0])
                rendezvous_write(rf, &id, sizeof(id));
        } else if (!supplied_unique_id) {
            rendezvous_read(rf, &id, sizeof(id), world_rank, "the NCCL unique id");
        }
        use_device_mpi = true;
        register_nccl_world_group(
            init_nccl_comm(world_size, world_rank, id),
            world_size, world_rank);
        nccl_watchdog_start(world_size, world_rank, rf);
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
    set_current_device(nccl_device_id);
    event_queue.run_sync([]() {
        checkCudaErrors(cudaSetDevice(nccl_device_id));
    });
    if (mpi_local_size > device_count) {
        // NCCL not support multiple process on one GPU,
        // failback use MPI
        return;
    }
    use_device_mpi = true;
    rendezvous_require_unlocked(mpi_world_size, "NCCL(MPI)");
    if (mpi_world_rank == 0)
        checkCudaErrors(ncclGetUniqueId(&id));
    MPI_CHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    register_nccl_world_group(
        init_nccl_comm(mpi_world_size, mpi_world_rank, id),
        mpi_world_size, mpi_world_rank);
    nccl_watchdog_start(mpi_world_size, mpi_world_rank, nullptr);
#endif
}

vector<int> nccl_get_unique_id() {
    ncclUniqueId generated;
    checkCudaErrors(ncclGetUniqueId(&generated));
    vector<int> result(sizeof(generated));
    for (int i=0; i<(int)result.size(); ++i)
        result[i] = ((unsigned char*)&generated)[i];
    return result;
}

void nccl_init_with_unique_id(vector<int> unique_id) {
    nccl_pending_unique_id = move(unique_id);
    nccl_init();
    nccl_pending_unique_id.clear();
}

// The communicator is built by nccl_init() now, but something still has to
// tear it down at exit, so what used to be nccl_initer's destructor lives on
// by itself.
struct nccl_finalizer {
    ~nccl_finalizer() { nccl_shutdown(); }
};

static nccl_finalizer nccl_final;

} // jittor
