// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
// Dun Liang <randonlang@gmail.com>. 
// Wenyang Zhou <576825820@qq.com>. 
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

namespace jittor {
int lock();

int unlock();

} // jittor
