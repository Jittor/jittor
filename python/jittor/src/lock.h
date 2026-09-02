// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
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

// Hand the Python side's lock descriptor (a HANDLE on Windows) and its
// current state over to C++. From this point on both languages take the same
// kind of lock on the same open file description, and share one _has_lock.
// @pyjt(set_lock_fd)
void set_lock_fd(int64 fd, bool has_lock);

void lock();

void unlock();

// Re-entrant entry points used from Python; they are no-ops when this process
// already holds the lock, so a Python lock_scope nested inside a C++
// lock_guard (or the other way round) does not release it early.
// @pyjt(lock_acquire)
void lock_acquire();

// @pyjt(lock_release)
void lock_release();

// @pyjt(lock_is_held)
bool lock_is_held();

EXTERN_LIB int _has_lock;

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
