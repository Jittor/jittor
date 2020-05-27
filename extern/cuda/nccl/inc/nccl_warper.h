// ***************************************************************
// Copyright (c) 2020 Jittor.
// Authors:
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "mpi_warper.h"

#include <cuda_runtime.h>
#include <nccl.h>
#include <helper_cuda.h>

namespace jittor {

extern ncclComm_t comm;
extern ncclUniqueId id;
extern int nccl_device_id;

} // jittor
