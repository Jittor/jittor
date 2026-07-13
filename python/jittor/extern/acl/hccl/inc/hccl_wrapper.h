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

#define ACLCHECK(ret) do {\
    if(ret != ACL_SUCCESS)\
    {\
        LOGe << "retcode: " << ret;\
        return;\
    }\
} while(0)\

#define HCCLCHECK(ret) do {\
    if(ret != HCCL_SUCCESS)\
    {\
        LOGe << HcclGetErrorString(ret) << " retcode: " << ret;\
        return;\
    }\
} while(0)\

// Return-value variants for functions that return a status (not void).
#define ACLCHECK_R(ret, rv) do {\
    if((ret) != ACL_SUCCESS) { LOGe << "acl retcode: " << (ret); return (rv); }\
} while(0)

#define HCCLCHECK_R(ret, rv) do {\
    auto _r = (ret);\
    if(_r != HCCL_SUCCESS) { LOGe << HcclGetErrorString(_r) << " retcode: " << _r; return (rv); }\
} while(0)

#include <hccl.h>

namespace jittor {

    EXTERN_LIB HcclRootInfo root_info;
    EXTERN_LIB HcclComm comm;
    EXTERN_LIB uint32_t hccl_device_id;

    // Initialize the HCCL communicator for this rank. Must be called once,
    // after `import jittor` completes (not during it). Safe to call repeatedly.
    // @pyjt(hccl_init)
    void hccl_init();

} // jittor
