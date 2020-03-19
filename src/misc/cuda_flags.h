// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

#ifdef HAS_CUDA
DECLARE_FLAG(int, use_cuda);
#else
constexpr int use_cuda = 0;
#endif

} // jittor