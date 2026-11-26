// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "core/common.h"
#include "codegen/jit_key.h"

namespace jittor {

struct Op;
struct FusedOp;

DECLARE_FLAG(string, jit_tier);
DECLARE_FLAG(int, jit_tier1_threshold);

// Fast integer mode accessor: 0=tier0, 1=tier1 (default), 2=auto (tiered async).
int jit_tier_mode();

enum class T0DType : uint8_t {
    F32 = 0,
    F64 = 1,
    F16 = 2,
    BF16 = 3,
    I32 = 4,
    I64 = 5,
    I8 = 6,
    U8 = 7,
    BOOL = 8,
    INVALID = 255,
};

enum class T0UnaryOp : uint8_t {
    Cast = 0,
    Abs,
    Negative,
    LogicalNot,
    BitwiseNot,
    Conj,
    Relu,
    Exp,
    Log,
    Log1p,
    Sqrt,
    Round,
    Floor,
    Ceil,
    RoundInt,
    FloorInt,
    CeilInt,
    Sin,
    Asin,
    Sinh,
    Asinh,
    Cos,
    Acos,
    Cosh,
    Acosh,
    Tan,
    Atan,
    Tanh,
    Atanh,
    Sigmoid,
    Erf,
    Erfinv,
    Invalid = 255,
};

enum class T0BinaryOp : uint8_t {
    Add = 0,
    Subtract,
    Multiply,
    Divide,
    FloorDivide,
    Mod,
    Fmod,
    Pow,
    Maximum,
    Minimum,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    LeftShift,
    RightShift,
    Invalid = 255,
};

enum class T0ReduceOp : uint8_t {
    Add = 0,
    Multiply,
    Maximum,
    Minimum,
    Mean,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    Invalid = 255,
};

struct T0ShapeMeta {
    int32_t rank = 0;
    int32_t num = 0;
    int32_t shape[8] = {};
    int32_t above[8] = {};
    int32_t xstride[8] = {};
    int32_t ystride[8] = {};
    int32_t cstride[8] = {};
    uint8_t xstrided = 0;
    uint8_t ystrided = 0;
    uint8_t cstrided = 0;
};

struct T0ReduceMeta {
    int32_t out_rank = 0;
    int32_t red_rank = 0;
    int32_t out_num = 0;
    int32_t red_num = 0;
    int32_t out_shape[8] = {};
    int32_t out_above[8] = {};
    int32_t out_xstride[8] = {};
    int32_t red_shape[8] = {};
    int32_t red_above[8] = {};
    int32_t red_xstride[8] = {};
};

#if defined(IS_CUDA) && !defined(IS_COREX)
void cuda_tier0_fill_scalar(void* dst, int dsize, uint64_t raw_bytes, int32_t count);
void cuda_tier0_unary(T0UnaryOp op, T0DType tx, T0DType ty,
                      const void* x, void* y, const T0ShapeMeta& meta);
void cuda_tier0_binary(T0BinaryOp op, T0DType tx, T0DType ty, T0DType tz,
                       const void* x, const void* y, void* z, const T0ShapeMeta& meta);
void cuda_tier0_ternary(T0DType tc, T0DType tx, T0DType ty, T0DType tz,
                        const void* cond, const void* x, const void* y, void* z,
                        const T0ShapeMeta& meta);
void cuda_tier0_broadcast(T0DType tx, const void* x, void* z, const T0ShapeMeta& meta);
void cuda_tier0_reduce(T0ReduceOp op, T0DType tx, T0DType ty,
                       const void* x, void* y, const T0ReduceMeta& meta);
#endif

// Eligibility check for Tier-0 execution (single Op or FusedOp).
bool tier0_can_execute(Op* op, bool is_fused);

// Whether parallel_compile_all_ops should skip blocking up-front compile for this op.
bool tier0_should_skip_parallel_compile(Op* op, bool is_fused);

// Try executing a non-fused Op via Tier-0 (and schedule async Tier-1 if auto).
bool tier0_try_execute_op(Op* op, JK& jk);

// Try executing a FusedOp via Tier-0 topological meta-op interpreter (and schedule async Tier-1 if auto).
bool tier0_try_execute_fused(FusedOp* fop, JK& jk);

// Internal diagnostic counters for tests/benchmarks:
// [tier_mode, tier0_op_execs, tier0_fused_execs, bg_enqueued, bg_promoted, bg_pending]
// @pyjt(_jit_tier_stats)
vector<int64> jit_tier_stats();

// Wait for any in-flight asynchronous Tier-1 background compilations to finish (for tests).
// @pyjt(_jit_tier_wait_bg)
void jit_tier_wait_bg();

} // namespace jittor
