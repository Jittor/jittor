// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Wenyang Zhou <576825820@qq.com>
//     Dun Liang <randonlang@gmail.com>
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#include "lock.h"

namespace jittor {

static int lock_fd = -1;
int _has_lock = 0;

void set_lock_path(string path) {
    lock_fd = open(path.c_str(), O_RDWR);
    ASSERT(lock_fd >= 0);
    LOGv << "OPEN LOCK path:" << path << "Pid:" << getpid();
}
 
void lock() {
    ASSERT(lock_fd >= 0);
    struct flock lock = {
        .l_type = F_WRLCK,
        .l_whence = SEEK_SET,
        .l_start = 0,
        .l_len = 0
    };
    ASSERT(fcntl(lock_fd, F_SETLKW, &lock) == 0);
    _has_lock = 1;
    LOGvv << "LOCK Pid:" << getpid();
}
 
void unlock() {
    ASSERT(lock_fd >= 0);
    struct flock lock = {
        .l_type = F_UNLCK,
        .l_whence = SEEK_SET,
        .l_start = 0,
        .l_len = 0
    };
    ASSERT(fcntl(lock_fd, F_SETLKW, &lock) == 0);
    _has_lock = 0;
    LOGvv << "UNLOCK Pid:" << getpid();
}

} // jittor