// ***************************************************************
// Copyright (c) 2023 Jittor.
// All Rights Reserved. 
// Maintainers:
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
// JT_NCCL_NO_MPI: build NCCL ops WITHOUT MPI (env/file rendezvous only), for the
// no-mpirun DDP path. mpi_wrapper.h hard-includes <mpi.h>, so in this mode we
// declare the few MPI globals the wrapper references directly (mirrors the HCCL
// no-mpi build) and never pull in libmpi.
#ifdef JT_NCCL_NO_MPI
#include "common.h"
namespace jittor {
    EXTERN_LIB int mpi_world_size;
    EXTERN_LIB int mpi_world_rank;
    EXTERN_LIB int mpi_local_size;
    EXTERN_LIB int mpi_local_rank;
    EXTERN_LIB bool inside_mpi;
    EXTERN_LIB bool use_device_mpi;
}
#else
#include "mpi_wrapper.h"
#endif

#include <cuda_runtime.h>
#include <nccl.h>
#include "utils/log.h"
#include "helper_cuda.h"
// nccl_dtype() below takes a NanoString; in the JT_NCCL_NO_MPI build we do not
// pull in mpi_wrapper.h, so include it here rather than rely on that path.
#include "misc/nano_string.h"
#include "misc/cuda_flags.h"
#include "misc/cuda_streams.h"

// helper_cuda.h guards this overload behind `#ifdef NCCL_H_`, so it only appears
// when nccl.h was included BEFORE it. Its own include guard makes the include
// above a no-op in any translation unit that already pulled it in earlier -- a
// JIT'd nccl op does, through the generated preamble -- and then
// `checkCudaErrors(ncclResult_t)` resolves against the cudaError_t overload and
// fails to compile. Declaring it here, after nccl.h, holds either way: the call
// in `check` is dependent, so ADL finds this at the point of instantiation.
const char *_cudaGetErrorEnum(ncclResult_t error);

namespace jittor {

// Destroys the communicator, reporting a failure instead of raising. Idempotent.
void nccl_shutdown();
EXTERN_LIB ncclUniqueId id;
EXTERN_LIB int nccl_device_id;

struct Var;

/**
Put the collective on this device's communication stream, ordered after the
default-stream work that produced its input.

Inside a bucket scope this is also where `ncclGroupStart()` is issued, on the
first collective of the bucket: a group has to be open around the NCCL calls
themselves, and the calls happen here, during graph execution, not while the
Python-level scope object is being constructed.
*/
cudaStream_t nccl_stream_begin();

/**
Close out one collective. Takes the operator's input and output because the
deferred-join path has to keep both blocks reserved -- see
`cuda_side_stream_defer_join`.

Outside a bucket scope (and inside a synchronous one) this joins the default
stream back immediately, which is the conservative behaviour every collective
had before 8.02: correct, but it also means nothing can overlap with the
collective.
*/
void nccl_stream_end(Var* x, Var* y);

/**
Bucket several collectives into one NCCL group, and optionally let the default
stream run ahead of them.

Two independent things, deliberately behind one scope because they are only
useful together:

* **Grouping** (`ncclGroupStart`/`ncclGroupEnd`) submits the whole bucket in
  one go instead of one launch per tensor. This is what makes gradient
  bucketing worth doing: N small all-reduces cost N launches otherwise.
* **`defer_join`** leaves the comm->compute event unwaited when the bucket
  closes, so default-stream compute enqueued afterwards overlaps with the
  collectives. `nccl_comm_wait()` is what orders the default stream behind
  them again, and it must be called before the results are read.

**Contract, because violating it is silent.** A group defers the NCCL calls
until `ncclGroupEnd()`, so nothing executed inside the scope may consume a
collective's output -- it has not run yet. In practice that means the scope
should contain the collectives and a single `jt.sync` of exactly those
collective outputs, so the collectives are the sinks of the graph being
executed. Producers of the inputs are fine; they run on the default stream and
touch none of the outputs.

`nccl_bucket_begin` refuses to open a second bucket while a previous deferred
join is still outstanding, rather than quietly stacking held blocks.
*/
// @pyjt(nccl_bucket_begin)
void nccl_bucket_begin(bool defer_join=true);
// @pyjt(nccl_bucket_end)
void nccl_bucket_end();

/**
Order the default stream behind every outstanding collective and release the
blocks the deferred join was holding. No-op when nothing is outstanding.
Returns whether it actually joined anything, so a test can prove the overlap
window existed rather than assuming it.
*/
// @pyjt(nccl_comm_wait)
bool nccl_comm_wait();

/**
Map a jittor dtype to the NCCL datatype used to send it.

This is the only NCCL dtype table; the five collective operators all go
through it. It is expanded from the same canonical dtype list as MPI's and
HCCL's tables (misc/collective_dtype.h), so the three cannot drift apart --
they already had: before this, nccl_all_reduce_op.cc was the one operator of
the five whose table had no bfloat16 entry, so a bf16 all-reduce failed to
compile while bf16 broadcast/reduce/all_gather/reduce_scatter worked.

Raises (LOGf) on a dtype NCCL has no type for, instead of expanding to nothing.
*/
ncclDataType_t nccl_dtype(NanoString dtype);

// Generate/expose the opaque bootstrap id as bytes. Store transport belongs
// to Python; the NCCL wrapper only consumes the exact binary payload.
// @pyjt(nccl_get_unique_id)
vector<int> nccl_get_unique_id();
// @pyjt(nccl_init_with_unique_id)
void nccl_init_with_unique_id(vector<int> unique_id);

/**
Create and query NCCL process groups.

Group 0 is the communicator built by nccl_init(). Every later group owns an
independent communicator whose local rank order follows `ranks`. All world
ranks must call nccl_create_process_group in the same order, matching
torch.distributed.new_group's collective contract.
*/
// @pyjt(nccl_create_process_group)
int nccl_create_process_group(vector<int> ranks);
// @pyjt(nccl_process_group_size)
int nccl_process_group_size(int group_id=0);
// @pyjt(nccl_process_group_rank)
int nccl_process_group_rank(int group_id=0);
ncclComm_t nccl_process_group_comm(int group_id=0);

/**
Build this rank's NCCL communicator. Call once, after `import jittor` has
loaded this module; safe to call again (it returns immediately).

This used to be a static constructor (`static nccl_initer nccl_init;`), so it
ran during dlopen. HCCL moved off that shape and its wrapper says why: a
blocking rendezvous plus a blocking communicator build at load time hangs
`import jittor` with no way to interrupt or report it. NCCL kept the static
constructor, and there it is worse -- a failure has nowhere to go. The
exception has to unwind through the C frames of the dynamic loader, finds no
handler, and the process dies in std::terminate; `import jittor` never gets to
raise anything a caller could catch or a traceback could point at. 8.09.
*/
// @pyjt(nccl_init)
void nccl_init();

} // jittor
