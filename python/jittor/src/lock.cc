// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers:
//     Wenyang Zhou <576825820@qq.com>
//     Dun Liang <randonlang@gmail.com>
//
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#include <fileapi.h>
#include <process.h>
#include <io.h>
#define getpid _getpid
#define open _open
#else
#include <unistd.h>
#include <sys/file.h>
#endif
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <chrono>
#include <thread>

#include "lock.h"

namespace jittor {

// The descriptor is opened by jittor_utils/lock.py and handed over here, so
// that both languages hold one lock of one kind on one open file description.
//
// This used to be a second open() of the same path plus fcntl(F_SETLKW), a
// POSIX record lock. On Linux a record lock and the flock() the Python side
// takes are independent lock families: neither excludes the other, so both
// sides could be "holding jittor.lock" at once and compile into the same cache
// directory. Record locks also have the rule that closing *any* descriptor for
// the file releases all of the process's record locks on it, which meant a
// Python lock object being garbage-collected quietly released this one.
#ifdef _WIN32
static HANDLE lock_handle = INVALID_HANDLE_VALUE;
#endif
static int lock_fd = -1;
int _has_lock = 0;

DEFINE_FLAG(bool, disable_lock, 0, "Disable file lock");

static double env_seconds(const char* name, double _default) {
    const char* v = getenv(name);
    if (!v || !v[0]) return _default;
    char* end = nullptr;
    double parsed = strtod(v, &end);
    if (end == v) {
        LOGw << name << "=" << v << "is not a number, using" << _default;
        return _default;
    }
    return parsed;
}

// Keep these in sync with jittor_utils/lock.py, which documents them.
static double lock_timeout() {
    static double v = env_seconds("JT_LOCK_TIMEOUT", 1800.0);
    return v;
}
static double lock_report_after() {
    static double v = env_seconds("JT_LOCK_REPORT_AFTER", 30.0);
    return v;
}

static string self_cmdline() {
    string cmd;
#ifndef _WIN32
    FILE* f = fopen("/proc/self/cmdline", "rb");
    if (f) {
        char buf[512];
        size_t n = fread(buf, 1, sizeof(buf), f);
        fclose(f);
        for (size_t i=0; i<n; i++)
            cmd += buf[i] ? buf[i] : ' ';
    }
#endif
    return cmd;
}

// Same JSON record shape the Python side writes, so either can read the other's.
static void write_holder() {
#ifndef _WIN32
    if (lock_fd < 0) return;
    auto now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    string record = string("{\"pid\": ") + std::to_string(getpid())
        + ", \"time\": " + std::to_string((int64)now)
        + ", \"cmd\": \"" + self_cmdline() + "\"}\n";
    if (ftruncate(lock_fd, 0) != 0) return;
    ssize_t written = pwrite(lock_fd, record.c_str(), record.size(), 0);
    (void)written;
#endif
}

static string describe_holder() {
#ifndef _WIN32
    if (lock_fd < 0) return "holder unknown";
    char buf[4096];
    ssize_t n = pread(lock_fd, buf, sizeof(buf)-1, 0);
    if (n <= 0) return "holder unknown (no record was written)";
    buf[n] = 0;
    return string("holder record: ") + buf;
#else
    return "holder unknown";
#endif
}

void set_lock_fd(int64 fd, bool has_lock) {
#ifdef _WIN32
    lock_handle = (HANDLE)fd;
#else
    lock_fd = (int)fd;
#endif
    _has_lock = has_lock ? 1 : 0;
    LOGv << "SHARE LOCK fd:" << fd << "has_lock:" << _has_lock
         << "Pid:" << getpid();
}

void lock() {
    if (disable_lock) return;
#ifdef _WIN32
    ASSERT(lock_handle != INVALID_HANDLE_VALUE);
    OVERLAPPED offset = {0, 0, 0, 0, NULL};
    ASSERT(LockFileEx(lock_handle, 2, 0, -0x10000, 0, &offset));
#else
    ASSERT(lock_fd >= 0);
    auto start = std::chrono::steady_clock::now();
    bool reported = false;
    while (1) {
        if (flock(lock_fd, LOCK_EX | LOCK_NB) == 0) break;
        if (errno != EWOULDBLOCK && errno != EINTR)
            LOGf << "could not lock the build lock:" << strerror(errno)
                 << ". Set disable_lock=1 to build without it (unsafe if"
                 << "anything else builds into the same cache).";
        double waited = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        if (!reported && waited >= lock_report_after()) {
            LOGw << "waiting for the build lock," << describe_holder();
            reported = true;
        }
        if (lock_timeout() > 0 && waited >= lock_timeout())
            LOGf << "timed out after" << waited
                 << "s waiting for the build lock," << describe_holder()
                 << ". Raise JT_LOCK_TIMEOUT (0 waits forever) if a cold"
                 << "build really takes this long.";
        std::this_thread::sleep_for(std::chrono::milliseconds(
            waited < 1 ? 50 : 500));
    }
    write_holder();
#endif
    _has_lock = 1;
    LOGvv << "LOCK Pid:" << getpid();
}

void unlock() {
    if (disable_lock) return;
#ifdef _WIN32
    ASSERT(lock_handle != INVALID_HANDLE_VALUE);
    OVERLAPPED offset = {0, 0, 0, 0, NULL};
    ASSERT(UnlockFileEx(lock_handle, 0, -0x10000, 0, &offset));
#else
    ASSERT(lock_fd >= 0);
    ASSERT(flock(lock_fd, LOCK_UN) == 0);
#endif
    _has_lock = 0;
    LOGvv << "UNLOCK Pid:" << getpid();
}

void lock_acquire() {
    if (_has_lock) return;
    lock();
}

void lock_release() {
    if (!_has_lock) return;
    unlock();
}

bool lock_is_held() {
    return _has_lock != 0;
}

} // jittor
