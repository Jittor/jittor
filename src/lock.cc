// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
// Dun Liang <randonlang@gmail.com>. 
// Wenyang Zhou <576825820@qq.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "lock.h"
#include "jit_compiler.h"
#include "utils/cache_compile.h"

namespace jittor {

DECLARE_FLAG(string, cache_path);

void lock_init(struct flock *lock, short type, short whence, off_t start, off_t len)
{
    if (lock == NULL)
        return;
 
    lock->l_type = type;
    lock->l_whence = whence;
    lock->l_start = start;
    lock->l_len = len;
}
 
int lock()
{
    auto lock_path = jittor::jit_compiler::join(cache_path, "../jittor.lock");
    const char* lockfilepath = lock_path.c_str();
    int fd = open(lockfilepath, O_RDWR);
    if (fd < 0)
    {
        return -1;
    }
    struct flock lock;
    lock_init(&lock, F_WRLCK, SEEK_SET, 0, 0);
    if (fcntl(fd, F_SETLKW, &lock) != 0)
    {
        return -1;
    }
    // printf("Pid: %ld process lock to write the file.\n", (long)getpid());
    return 0;
}
 
int unlock()
{
    auto lock_path = jittor::jit_compiler::join(cache_path, "../jittor.lock");
    const char* lockfilepath = lock_path.c_str();
    int fd = open(lockfilepath, O_RDWR);
    if (fd < 0)
    {
        return -1;
    }
    struct flock lock;
    lock_init(&lock, F_UNLCK, SEEK_SET, 0, 0);
    if (fcntl(fd, F_SETLKW, &lock) != 0)
    {
        return -1;
    }
    // printf("Pid: %ld process release the file.\n", (long)getpid());
    return 0;
}

} // jittor