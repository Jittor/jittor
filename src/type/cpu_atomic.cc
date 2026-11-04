// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "type/cpu_atomic.h"
#include "type/floor_divide_compute.h"

namespace jittor {

std::atomic_flag lock = ATOMIC_FLAG_INIT;;
std::atomic<int> int_division_by_zero{0};

} // jittor
