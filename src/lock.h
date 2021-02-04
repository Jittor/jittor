// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Wenyang Zhou <576825820@qq.com>
//     Dun Liang <randonlang@gmail.com>
// 
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

extern int _has_lock;

struct lock_guard {
    int has_lock = 0;
    inline lock_guard() { 
        if (_has_lock) return;
        has_lock = 1;
        lock(); 
    }
    inline ~lock_guard() {
        if (!has_lock) return;
        unlock();
    }
};

} // jittor
