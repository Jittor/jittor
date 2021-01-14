// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/rename_loop_index_pass.h"

namespace jittor {

void RenameLoopIndexPass::run() {
    // TODO: move out rename_loop_index
    ir->rename_loop_index();
}

} // jittor