// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "fused_op.h"
#include "opt/kernel_ir.h"

namespace jittor {

struct Pass {
    FusedOp* op;
    KernelIR* all;
    KernelIR* ir;
    PassManager* pm;
    string name;

    Pass(const string& name);
    virtual ~Pass();

    void init(PassManager* pm);
    virtual void run() = 0;
};

} // jittor
