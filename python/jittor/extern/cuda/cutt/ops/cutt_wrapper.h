// ***************************************************************
// Copyright (c) 2019 
//     Dun Liang <randonlang@gmail.com>
//     Guowei Yang <471184555@qq.com>
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "executor.h"
#include "CudaUtils.h"

void jt_alloc(void** p, size_t len, size_t& allocation);

void jt_free(void* p, size_t len, size_t& allocation);