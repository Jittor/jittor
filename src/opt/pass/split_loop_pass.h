// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "opt/pass/pass.h"

namespace jittor {

struct SplitLoopPass : Pass {
    int number_of_ranges_after_split;

    SplitLoopPass() : Pass("split_loop"), number_of_ranges_after_split(0) {};
    void run() override;
};

} // jittor
