// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"
#include "core/var.h"

namespace jittor {

// Inference-only fused BNSD FlashAttention (aclnnFlashAttentionScoreV2)
// bypassing Python CodeOp serialization, dummy tensor allocations, and
// attribute cache hashing. Owns softmax_max/softmax_sum scratch buffers so
// their addresses remain stable across device-graph capture/replay.
struct MappedFlashAttentionOp : Op {
    static constexpr uint32 backend_mask = OpBackendAccelerator;
    Var* q;
    Var* k;
    Var* v;
    Var* atten_mask;
    Var* out;
    VarPtr softmax_max_scratch;
    VarPtr softmax_sum_scratch;
    double scale;
    int64 sparse_mode;
    bool has_mask;
    bool mask_is_realshift;

    MappedFlashAttentionOp(Var* q, Var* k, Var* v, Var* atten_mask,
                           double scale, int64 sparse_mode,
                           bool has_mask, bool mask_is_realshift=false);

    const char* name() const override { return "mapped_flash_attention"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor
