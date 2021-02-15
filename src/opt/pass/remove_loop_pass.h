// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "opt/pass/pass.h"

namespace jittor {

// this is a debug pass, remove i-th loop, key: removei
struct RemoveLoopPass : Pass {
    RemoveLoopPass() : Pass("remove_loop") {};
    void run() override;
};

} // jittor
