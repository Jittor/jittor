// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "opt/pass/pass.h"

namespace jittor {

struct SolveConflictDefinePass : Pass {
    SolveConflictDefinePass() : Pass("solve_conflict_define") {};
    void run() override;
};

} // jittor
