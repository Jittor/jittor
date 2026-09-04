// ***************************************************************
// Copyright (c) 2025 Jittor.
// All Rights Reserved. 
// Maintainers:
//     Jiapeng Zhang <zjp24@mails.tsinghua.edu.cn>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#pragma once
// HCCL ops need the ACL runtime stream (aclstream) and ACL types; pull these in
// for both build modes (the JIT-compiled op .cc only includes this header).
#include "acl_jittor.h"
// JT_HCCL_NO_MPI: build the HCCL ops WITHOUT any MPI dependency (env/file-based
// rendezvous only). On Ascend the conda-OpenMPI + CANN combo crashes, so the
// default multi-card path avoids libmpi entirely.
#ifdef JT_HCCL_NO_MPI
#include "common.h"
namespace jittor {
    // Minimal stand-ins for the few MPI globals the ops reference, so we don't
    // pull in mpi_wrapper.h / libmpi. These are unused in env/file mode.
    EXTERN_LIB int mpi_world_size;
    EXTERN_LIB int mpi_world_rank;
    EXTERN_LIB int mpi_local_rank;
    EXTERN_LIB bool inside_mpi;
    EXTERN_LIB bool use_device_mpi;
}
#else
#include "mpi_wrapper.h"
#endif

// These used to log at LOGe and `return`. Inside an operator's jit_run() that
// meant the collective silently did not run: the output var kept whatever
// happened to be in it, the rank carried on, and nothing above a log line said
// so. A failed collective is not recoverable -- the ranks are already out of
// step -- so throw, and let the rank die loudly enough that the job can be torn
// down instead of every rank continuing with garbage. 6.B03.
//
// Note these evaluate `ret` exactly once; the previous ACLCHECK evaluated it
// twice on the failure path.
#define ACLCHECK(ret) do {\
    auto _acl_r = (ret);\
    if (_acl_r != ACL_SUCCESS)\
        LOGf << "acl error in" << #ret << "retcode:" << _acl_r;\
} while(0)

#define HCCLCHECK(ret) do {\
    auto _hccl_r = (ret);\
    if (_hccl_r != HCCL_SUCCESS)\
        LOGf << "hccl error in" << #ret\
             << HcclGetErrorString(_hccl_r) << "retcode:" << _hccl_r;\
} while(0)

// Shutdown-only variant: destructors must not throw (that is std::terminate),
// so during teardown we report and carry on.
#define HCCLCHECK_PEEK(ret) do {\
    auto _hccl_r = (ret);\
    if (_hccl_r != HCCL_SUCCESS)\
        LOGe << "hccl error during shutdown, ignored:" << #ret\
             << HcclGetErrorString(_hccl_r) << "retcode:" << _hccl_r;\
} while(0)

#include <hccl.h>
// hccl_dtype() below takes a NanoString; in the JT_HCCL_NO_MPI build we do not
// pull in mpi_wrapper.h, so include it here rather than rely on that path.
#include "misc/nano_string.h"

namespace jittor {

/**
Map a jittor dtype to the HCCL datatype used to send it.

This is the only HCCL dtype table; the four collective operators all go
through it. It is expanded from the same canonical dtype list as MPI's and
NCCL's tables (misc/collective_dtype.h) so the three cannot drift apart.

Raises (LOGf) on a dtype this table has no entry for, instead of expanding to
nothing (which used to be a confusing compile error inside generated code).
*/
    HcclDataType hccl_dtype(NanoString dtype);

    EXTERN_LIB HcclRootInfo root_info;
    EXTERN_LIB uint32_t hccl_device_id;

    // Group 0 is WORLD; later ids own independent HCCL communicators.
    // @pyjt(hccl_create_process_group)
    int hccl_create_process_group(vector<int> ranks);
    // @pyjt(hccl_process_group_size)
    int hccl_process_group_size(int group_id=0);
    // @pyjt(hccl_process_group_rank)
    int hccl_process_group_rank(int group_id=0);
    HcclComm hccl_process_group_comm(int group_id=0);

    // Initialize the HCCL communicator for this rank. Must be called once,
    // after `import jittor` completes (not during it). Safe to call repeatedly.
    // @pyjt(hccl_init)
    void hccl_init();

/**
Whether a collective still brackets itself with a full device+stream sync.

`JT_HCCL_COLLECTIVE_SYNC=full` (the default) keeps the historical behaviour;
`stream-order` drops the syncs and relies on stream ordering instead. Read once
and cached.

**Why this is a switch and not a deletion.** Every HCCL collective used to
issue `aclrtSynchronizeDevice()` + `aclrtSynchronizeStream(aclstream)` both
before and after the call -- four full synchronisations per collective, which
serialises the whole NPU pipeline around every gradient exchange (8.02, audit
06-backends §distributed, severity: critical). They are redundant *by reading*:
there is one global `aclstream` (`acl_jittor.h`) and every ACL operator
enqueues onto it, so the collective is already ordered after the work that
produced its input and before the work that consumes its output.

But "redundant by reading" is not "verified", and this repository has no Ascend
hardware, so the removal cannot be run even once here. Deleting the syncs
outright would ship an unverified change to the one code path whose failure
mode is a silently wrong gradient on someone else's cluster. Behind a switch,
the on-machine step is one environment variable and an A/B comparison rather
than a code edit -- see `agent/manuals/hccl-on-device-verification.md`.

Flip the default to `stream-order` once that document's checklist passes on a
real multi-card Ascend node; the switch (and this comment) can go with it.
*/
bool hccl_collective_full_sync();

// Bracket one collective. The four operators call these instead of writing the
// synchronisations out, so there is one place to change and one place to read.
inline void hccl_collective_begin() {
    if (hccl_collective_full_sync()) {
        ACLCHECK(aclrtSynchronizeDevice());
        ACLCHECK(aclrtSynchronizeStream(aclstream));
    }
}

inline void hccl_collective_end() {
    if (hccl_collective_full_sync()) {
        ACLCHECK(aclrtSynchronizeDevice());
        ACLCHECK(aclrtSynchronizeStream(aclstream));
    }
}

} // jittor
