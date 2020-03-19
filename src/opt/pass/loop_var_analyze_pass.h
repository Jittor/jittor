// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "opt/pass/pass.h"

namespace jittor {

struct LoopVarAnalyzePass : Pass {
    // total number of loop ranges
    int number_of_ranges;

    LoopVarAnalyzePass() : Pass("loop_var_analyze"), number_of_ranges(0) {};
    void run() override;
};

} // jittor
