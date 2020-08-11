// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Wenyang Zhou <576825820@qq.com>
//     Dun Liang <randonlang@gmail.com>
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

// @pyjt(set_lock_path)
void set_lock_path(string path);

void lock();

void unlock();

struct lock_guard {
    inline lock_guard() { lock(); }
    inline ~lock_guard() { unlock(); }
};

} // jittor
