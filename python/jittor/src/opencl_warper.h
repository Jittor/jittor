// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <CL/cl.h>
#include <random>
#include "common.h"
#include "misc/opencl_flags.h"


namespace jittor {

extern cl_context opencl_context;
extern cl_command_queue opencl_queue;

} // jittor
