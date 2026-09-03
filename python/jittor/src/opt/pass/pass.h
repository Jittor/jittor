// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
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
    // The KernelIR attributes this pass reads, and the ones it produces.
    //
    // These were 14 string literals spread over 13 pass files whose only
    // documentation was a comment in kernel_ir.h, so the order the passes
    // depend on -- who has to run before whom for an attribute to be there --
    // was not written down anywhere and nothing checked it. Declared here, the
    // pass manager can: it walks the pipeline in order and refuses a pass that
    // reads an attribute nothing before it produces. Attributes the parser
    // itself sets (lvalue, rvalue, code, dtype, loop_id, raw, ...) count as
    // produced from the start; see PassManager's seed.
    vector<const char*> reads, writes;

    Pass(const string& name);
    virtual ~Pass();

    void init(PassManager* pm);
    virtual void run() = 0;
};

} // jittor
