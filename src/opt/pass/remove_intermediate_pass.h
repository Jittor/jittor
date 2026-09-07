// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "opt/pass/pass.h"

namespace jittor {

struct RemoveIntermediatePass : Pass {
    RemoveIntermediatePass() : Pass("remove_intermediate") {
        reads = {kir::lvalue, kir::rvalue, kir::code, kir::used};
        writes = {kir::lvalue, kir::rvalue, kir::dtype, kir::code};
    };
    void run() override;
};

} // jittor
