// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include <functional>

namespace jittor {

struct Deleter {
    std::function<void()> del;
    inline Deleter(std::function<void()>&& func) : del(move(func)) {}
    inline ~Deleter() { del(); }
};

} // jittor
