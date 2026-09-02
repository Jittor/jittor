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
#include "misc/file_rendezvous.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <thread>
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


static bool nccl_comm_created = false;

// NCCL's p2p transport treats a refused peer access as fatal, and the error it
// raises -- "unhandled cuda error" -- names neither the cause nor the cure. Name
// both before it escapes: NCCL's own explanation goes to stderr, which a test
// runner capturing output turns into a bare SIGABRT with nothing to go on.
static void init_nccl_comm(int world_size, int world_rank) {
    auto result = ncclCommInitRank(&comm, world_size, id, world_rank);
    if (result == ncclSuccess) { nccl_comm_created = true; return; }
    LOGe << "ncclCommInitRank failed:" << ncclGetErrorString(result)
         << "\n  If NCCL reports that peer access is unsupported, this machine"
            " cannot do direct GPU-to-GPU transfers. Set NCCL_P2P_DISABLE=1 to"
            " route the collectives through shared memory instead."
         << "\n  Set NCCL_DEBUG=INFO for NCCL's own account of the failure.";
    checkCudaErrors(result);
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
    ncclCommAbort(comm);
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
        if (ncclCommGetAsyncError(comm, &async) == ncclSuccess &&
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
    // The watchdog polls `comm`; it has to be gone before the comm is.
    nccl_watchdog_join();
    if (!nccl_comm_created) return;
    nccl_comm_created = false;
    peekCudaErrorsAlways(ncclCommDestroy(comm));
    comm = nullptr;
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
        checkCudaErrors(cudaSetDevice(nccl_device_id));
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
        if (world_size > 1 && (!rf || !rf[0]))
            LOGf << "NCCL(env): JT_NCCL_WORLD_SIZE=" >> world_size
                 << "but JT_NCCL_ROOTINFO_FILE is not set. Every rank needs the"
                    " same path, on a filesystem all of them share, to exchange"
                    " the NCCL unique id.";
        if (world_rank == 0) {
            checkCudaErrors(ncclGetUniqueId(&id));
            if (rf && rf[0])
                rendezvous_write(rf, &id, sizeof(id));
        } else {
            rendezvous_read(rf, &id, sizeof(id), world_rank, "the NCCL unique id");
        }
        use_device_mpi = true;
        init_nccl_comm(world_size, world_rank);
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
    rendezvous_require_unlocked(mpi_world_size, "NCCL(MPI)");
    if (mpi_world_rank == 0)
        checkCudaErrors(ncclGetUniqueId(&id));
    MPI_CHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    init_nccl_comm(mpi_world_size, mpi_world_rank);
    nccl_watchdog_start(mpi_world_size, mpi_world_rank, nullptr);
#endif
}

// The communicator is built by nccl_init() now, but something still has to
// tear it down at exit, so what used to be nccl_initer's destructor lives on
// by itself.
struct nccl_finalizer {
    ~nccl_finalizer() { nccl_shutdown(); }
};

static nccl_finalizer nccl_final;

} // jittor
