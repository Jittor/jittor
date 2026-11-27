// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"
#include "core/var.h"

namespace jittor {

// Fused inference FP8 / MXFP8 linear projection on a mapped accelerator backend:
//   mode == 0: block-wise dynamic activation quant (1 x block_size) + QuantMatmulV5
//   mode == 1: OCP MXFP8 dynamic activation quant (1 x block_size) + QuantMatmulV5
//
// Keeps temporary quantized activation buffers on the op itself (matching
// MappedRmsNormOp::rstd_scratch), eliminating intermediate graph nodes, CodeOp
// attribute maps, and per-call reshape overhead while keeping buffer addresses
// stable for device-graph capture/replay.
struct MappedFp8LinearOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var* x;
    Var* weight;
    Var* weight_scale;
    Var* y;
    VarPtr qx_scratch;
    VarPtr sx_scratch;
    int64 mode;
    int64 block_size;
    NanoString out_dtype;

    MappedFp8LinearOp(Var* x, Var* weight, Var* weight_scale,
                      int64 mode, int64 block_size, NanoString out_dtype = ns_bfloat16);

    const char* name() const override { return "mapped_fp8_linear"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
