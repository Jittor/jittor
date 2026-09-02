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

// helper_cuda.h guards this overload behind `#ifdef NCCL_H_`, so it only appears
// when nccl.h was included BEFORE it. Its own include guard makes the include
// above a no-op in any translation unit that already pulled it in earlier -- a
// JIT'd nccl op does, through the generated preamble -- and then
// `checkCudaErrors(ncclResult_t)` resolves against the cudaError_t overload and
// fails to compile. Declaring it here, after nccl.h, holds either way: the call
// in `check` is dependent, so ADL finds this at the point of instantiation.
const char *_cudaGetErrorEnum(ncclResult_t error);

namespace jittor {

EXTERN_LIB ncclComm_t comm;

// Destroys the communicator, reporting a failure instead of raising. Idempotent.
void nccl_shutdown();
EXTERN_LIB ncclUniqueId id;
EXTERN_LIB int nccl_device_id;

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
