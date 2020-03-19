// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/solve_conflict_define_pass.h"

namespace jittor {

void SolveConflictDefinePass::run() {
    ir->solve_conflict_define();
}

} // jittor