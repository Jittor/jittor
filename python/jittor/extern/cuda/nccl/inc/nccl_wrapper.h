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
EXTERN_LIB ncclUniqueId id;
EXTERN_LIB int nccl_device_id;

} // jittor
