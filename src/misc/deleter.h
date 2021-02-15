// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
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
