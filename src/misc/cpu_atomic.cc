// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "misc/cpu_atomic.h"

namespace jittor {

std::atomic_flag lock = ATOMIC_FLAG_INIT;;

} // jittor
