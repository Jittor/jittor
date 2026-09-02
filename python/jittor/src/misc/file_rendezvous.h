// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
//
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "common.h"
#include "lock.h"

/**
The shared-file rendezvous used by the MPI-free NCCL and HCCL bootstraps.

Rank 0 writes the communicator's unique id to a file on shared storage; every
other rank polls until that file is complete. Both backends carried their own
copy of this loop and both gave up quietly at the end of it:

* NCCL (`nccl_wrapper.cc`) polled 6000 times at 20 ms -- a hardcoded 121 s that
  nothing could configure -- and then fell through **without checking whether
  it had read anything**. The still-zero id went to `ncclCommInitRank`. On
  NCCL 2.18.3 that answers `ncclInternalError`, "please report this issue to
  the NCCL developers": two minutes of apparent hang, and then a jittor launch
  misconfiguration reported as a bug in NVIDIA's library. Builds that accept
  the id, and files caught half-written, block instead. And when
  `JT_NCCL_ROOTINFO_FILE` was unset there was no wait at all -- the
  uninitialized id went in immediately.
* HCCL (`hccl_wrapper.cc`) logged and returned false, which its caller read as
  "env mode not in use" and turned into a silent single-card run.

Between them that is every bad outcome available: a job that hangs, a job that
blames a third party for a launch mistake, and a job that trains N independent
models. So this implementation throws, and the message says which rank was
waiting, for what, for how long, and what to check.

`JT_RENDEZVOUS_TIMEOUT_S` (default 120) sets the budget, so a test can ask for
a short one and a slow shared filesystem can ask for a long one. The wait is
measured on a real clock rather than by counting sleeps: the poll interval plus
the cost of each `fopen` made the old attempt count an underestimate of the
elapsed time, so "120s" was never 120s.

Header-only on purpose: NCCL and HCCL are built as two separate custom-op
modules with different flags, and neither links the other's objects.
*/

namespace jittor {

/**
Refuse to wait for the other ranks while holding jittor's build lock.

The lock is one flock over the whole cache directory, and `import jittor` holds
it for its entire duration. So a rank that blocks for its peers while holding
it blocks them from ever arriving: they need that same lock to compile before
they can reach the rendezvous. The result is a deadlock that looks like nothing
at all -- one rank at 100% CPU inside MPI_Bcast, the others asleep on a file
lock, no output from any of them.

`compile_custom_ops` already drops the lock around its dlopen ("unlock scope
when initialize") precisely because the communicator used to be built by a
static constructor there. 8.09 moved that build to an explicit call, which put
it back inside the lock and reproduced the deadlock on a cold two-rank MPI run;
`setup_nccl`/`setup_hccl` now wrap the call in `lock.unlock_scope()`. This
check is here so that if anything undoes that, the job says so instead of
hanging.
*/
inline void rendezvous_require_unlocked(int world_size, const char* who) {
    if (world_size <= 1 || !lock_is_held()) return;
    LOGf << who << ": about to wait for the other ranks while this process"
            " holds jittor's build lock. They need that lock to compile before"
            " they can get here, so the job would deadlock with no diagnosis"
            " -- one rank spinning, the rest asleep on a file lock."
         << "\n  The call has to be inside jittor_utils.lock.unlock_scope();"
            " see setup_nccl() in compile_extern.py.";
}

inline double rendezvous_timeout_s() {
    if (const char* env = getenv("JT_RENDEZVOUS_TIMEOUT_S")) {
        double v = atof(env);
        if (v > 0) return v;
    }
    return 120.0;
}

// Write `bytes` from `blob` to `path` atomically -- temporary file plus rename
// -- so a reader never sees a half-written id. Throws if any step fails.
inline void rendezvous_write(const string& path, const void* blob, size_t bytes) {
    string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f)
        LOGf << "rendezvous: cannot create" << tmp >> ":" << strerror(errno)
             << "\n  The rendezvous directory must exist and be writable by"
                " every rank.";
    size_t written = fwrite(blob, 1, bytes, f);
    int flush_err = fflush(f);
    fclose(f);
    if (written != bytes || flush_err)
        LOGf << "rendezvous: short write to" << tmp >> ":" << written << "of"
             << bytes << "bytes" >> ";" << strerror(errno);
    if (rename(tmp.c_str(), path.c_str()) != 0)
        LOGf << "rendezvous: cannot rename" << tmp << "to" << path >> ":"
             << strerror(errno);
}

// Poll `path` until it holds at least `bytes`, then read them into `blob`.
// Throws on timeout instead of letting the caller carry on with garbage.
// `what` names the payload for the error message ("the NCCL unique id").
inline void rendezvous_read(const string& path, void* blob, size_t bytes,
                            int world_rank, const char* what) {
    const double timeout = rendezvous_timeout_s();
    const auto deadline = std::chrono::steady_clock::now()
        + std::chrono::duration<double>(timeout);
    while (1) {
        FILE* f = fopen(path.c_str(), "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long size = ftell(f);
            if (size >= (long)bytes) {
                fseek(f, 0, SEEK_SET);
                size_t n = fread(blob, 1, bytes, f);
                fclose(f);
                if (n == bytes) return;
            } else {
                fclose(f);
            }
        }
        if (std::chrono::steady_clock::now() >= deadline) break;
        struct timespec ts{0, 20*1000*1000};   // 20ms
        nanosleep(&ts, nullptr);
    }
    LOGf << "rendezvous timeout: rank" << world_rank << "waited" << timeout
         << "s for" << what << "at" << path >> ", which never appeared."
         << "\n  Rank 0 writes this file. Either rank 0 failed to start, or it"
            " does not see the same path: every rank must name one path on a"
            " filesystem all of them share."
         << "\n  Raise JT_RENDEZVOUS_TIMEOUT_S if the storage is merely slow.";
}

} // jittor
