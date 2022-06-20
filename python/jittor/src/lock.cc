// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
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
#endif
#include <fcntl.h>
#include <errno.h>

#include "lock.h"

namespace jittor {

static int lock_fd = -1;
int _has_lock = 0;

DEFINE_FLAG(bool, disable_lock, 0, "Disable file lock");

void set_lock_path(string path) {
    lock_fd = open(_to_winstr(path).c_str(), O_RDWR);
    ASSERT(lock_fd >= 0);
    LOGv << "OPEN LOCK path:" << path << "Pid:" << getpid();
}
 
void lock() {
    if (disable_lock) return;
    ASSERT(lock_fd >= 0);
#ifdef _WIN32
	OVERLAPPED offset = {0, 0, 0, 0, NULL};
    auto hfile = (HANDLE)_get_osfhandle(lock_fd);
    ASSERT(LockFileEx(hfile, 2, 0, -0x10000, 0, &offset));
#else
    struct flock lock = {
        .l_type = F_WRLCK,
        .l_whence = SEEK_SET,
        .l_start = 0,
        .l_len = 0
    };
    ASSERT(fcntl(lock_fd, F_SETLKW, &lock) == 0);
#endif
    _has_lock = 1;
    LOGvv << "LOCK Pid:" << getpid();
}
 
void unlock() {
    if (disable_lock) return;
    ASSERT(lock_fd >= 0);
#ifdef _WIN32
	OVERLAPPED offset = {0, 0, 0, 0, NULL};
    auto hfile = (HANDLE)_get_osfhandle(lock_fd);
    ASSERT(UnlockFileEx(hfile, 0, -0x10000, 0, &offset));
#else
    struct flock lock = {
        .l_type = F_UNLCK,
        .l_whence = SEEK_SET,
        .l_start = 0,
        .l_len = 0
    };
    ASSERT(fcntl(lock_fd, F_SETLKW, &lock) == 0);
#endif
    _has_lock = 0;
    LOGvv << "UNLOCK Pid:" << getpid();
}

} // jittor