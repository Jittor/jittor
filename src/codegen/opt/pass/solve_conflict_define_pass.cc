// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "core/var.h"
#include "codegen/opt/pass_manager.h"
#include "codegen/opt/pass/solve_conflict_define_pass.h"

namespace jittor {

void SolveConflictDefinePass::run() {
    ir->solve_conflict_define();
}

} // jittor