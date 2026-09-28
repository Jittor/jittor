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
#include "core/common.h"

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

// Take the build lock for a scope, from any thread.
//
// lock_guard used to be "if nobody holds it, flock; on the way out, unflock",
// with the "does anybody hold it" question answered by reading the plain int
// _has_lock. That is only safe while exactly one thread ever builds a guard:
// two compile workers both reading _has_lock==0 would both flock (harmless,
// the descriptor already holds it) and then the *first* one to finish would
// unflock while the second was still writing into the cache. The parallel
// compiler worked around this by taking one guard on the main thread around
// the whole batch, which is also why a batch of pure cache hits paid for the
// lock.
//
// These two keep a depth count under a mutex instead, so the flock is taken by
// whichever thread arrives first and released only when the last one leaves.
// Nested guards on one thread still cost nothing, and a lock already held by
// the Python side is still left alone -- returning false means "not ours".
bool build_lock_enter();
void build_lock_leave();

struct lock_guard {
    bool has_lock;
    inline lock_guard() : has_lock(build_lock_enter()) {}
    inline ~lock_guard() {
        if (!has_lock) return;
        build_lock_leave();
    }
    lock_guard(const lock_guard&) = delete;
    lock_guard& operator=(const lock_guard&) = delete;
};

} // jittor
