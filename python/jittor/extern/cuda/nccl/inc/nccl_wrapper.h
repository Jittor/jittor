// ***************************************************************
// Copyright (c) 2022 Jittor.
// All Rights Reserved. 
// Maintainers:
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "mpi_wrapper.h"

#include <cuda_runtime.h>
#include <nccl.h>
#include "utils/log.h"
#include "helper_cuda.h"

namespace jittor {

EXTERN_LIB ncclComm_t comm;
EXTERN_LIB ncclUniqueId id;
EXTERN_LIB int nccl_device_id;

} // jittor
