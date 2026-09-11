// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
// AscendC fused-kernel path for float32 elementwise groups.
//
// The generic ACL path walks a fused group and issues one aclnn launch per
// node. This path generates one AscendC kernel for the whole group, compiles
// it with ccec (the device compiler the CUDA backend calls nvcc) and launches
// it once. It handles only what it can prove correct -- float32, contiguous,
// identical element counts, unary/binary nodes with a bit-exact AscendC
// intrinsic -- and reports failure for everything else so the caller keeps
// using the per-node path. It never falls back to another backend.
#pragma once
#include "core/common.h"

namespace jittor
{
    struct FusedOp;

    // Runs `fop` as a single generated AscendC kernel. Returns false without
    // any side effect when the group is not eligible or the feature is off,
    // in which case the caller must run its normal per-node path.
    bool exec_fused_ascendc(FusedOp *fop);

    // Number of groups this path has executed, for tests and benchmarks.
    // Reachable from python through the core library's jt_acl_ascendc_*
    // C entry points; see bench/asc_bench.py.
    int64 acl_ascendc_fused_count();

    // Nodes inside those groups: the launches this path removed is
    // (nodes - groups).
    int64 acl_ascendc_fused_node_count();

    // "<count>\t<reason>" per line, for the groups this path declined while
    // the acl_ascendc_report environment variable was set.
    string acl_ascendc_reject_report();

} // jittor
