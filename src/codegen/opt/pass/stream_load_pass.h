// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "codegen/opt/pass/pass.h"

namespace jittor {

// Lets a CUDA kernel load an input evict-first on the runs that read it for
// the last time. Every load of a fused input goes through jt_stream_ld, which
// picks the plain or the streaming load by one bit of an argument the host
// fills from FusedOp::streamed_inputs; see stream_dying_inputs. Last, after
// every pass that matches statements by their text.
struct StreamLoadPass : Pass {
    StreamLoadPass() : Pass("stream_load") {
        reads = {kir::code, kir::dtype, kir::lvalue, kir::rvalue};
        writes = {kir::code, kir::rvalue};
    };
    void run() override;
};

} // jittor
