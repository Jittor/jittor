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
#include "mpi_wrapper.h"

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
} while(0)

#include <hccl.h>

namespace jittor {

    EXTERN_LIB HcclRootInfo root_info;
    EXTERN_LIB HcclComm comm;
    EXTERN_LIB uint32_t hccl_device_id;

} // jittor
