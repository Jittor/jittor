// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "runtime/tier0_ops.h"

#include <Python.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#ifdef __linux__
#include <pthread.h>
#endif

#include "codegen/jit_compiler.h"
#include "codegen/nvrtc_compiler.h"
#include "codegen/op_compiler.h"
#include "codegen/opt/tuner_manager.h"
#include "core/executor.h"
#include "core/fused_op.h"
#include "core/op.h"
#include "core/var.h"
#include "mem/allocator.h"
#include "mem/swap.h"
#include "ops/binary_op.h"
#include "ops/broadcast_to_op.h"
#include "ops/composite/array_op.h"
#include "ops/composite/op_dispatch.h"
#include "ops/op_register.h"
#include "ops/reduce_op.h"
#include "ops/ternary_op.h"
#include "ops/unary_op.h"
#include "runtime/backend.h"
#include "runtime/backend_streams.h"
#include "runtime/device.h"
#include "runtime/graph_capture.h"
#include "runtime/jit_policy.h"
#include "runtime/profiler/profiler.h"
#include "runtime/profiler/step_trace.h"
#include "type/cpu_math.h"
#include "type/floor_divide_compute.h"
#include "type/minmax_compute.h"
#include "utils/flags.h"
#include "utils/str_utils.h"

namespace jittor {

EXTERN_LIB void register_cleanup_callback(void (*cb)());
DECLARE_FLAG(int, keep_graph);
DECLARE_FLAG(int, profiler_enable);
DECLARE_FLAG(int, jit_search_kernel);

static std::atomic<int> g_jit_tier_mode{1};

DEFINE_FLAG_WITH_SETTER(string, jit_tier, "1",
    "JIT compilation tier policy: '1' (default, standard Tier-1 JIT), "
    "'0' (Tier-0 precompiled meta-ops with zero compilation where supported), "
    "'2' or 'auto' (execute cold ops immediately via Tier-0 and asynchronously "
    "promote hot ops to Tier-1).");

void setter_jit_tier(const string& old_value, const string& value) {
    (void)old_value;
    if (value == "1" || value == "tier1") {
        g_jit_tier_mode.store(1, std::memory_order_release);
    } else if (value == "0" || value == "tier0") {
        g_jit_tier_mode.store(0, std::memory_order_release);
    } else if (value == "2" || value == "auto") {
        g_jit_tier_mode.store(2, std::memory_order_release);
    } else {
        USER_ERROR << "jit_tier must be '0', '1', '2', or 'auto'; got '" >> value >> "'";
    }
}

DEFINE_FLAG(int, jit_tier1_threshold, 2,
    "Execution count before auto-tiered JIT keys are promoted asynchronously to Tier-1.");

int jit_tier_mode() {
    return g_jit_tier_mode.load(std::memory_order_acquire);
}

namespace {

std::atomic<int64> g_tier0_op_execs{0};
std::atomic<int64> g_tier0_fused_execs{0};
std::atomic<int64> g_bg_enqueued{0};
std::atomic<int64> g_bg_promoted{0};

// =========================================================================
// Dtype & Op enum mappers
// =========================================================================

static inline T0DType to_t0_dtype(NanoString ns) {
    if (ns == ns_float32) return T0DType::F32;
    if (ns == ns_float64) return T0DType::F64;
    if (ns == ns_float16) return T0DType::F16;
    if (ns == ns_bfloat16) return T0DType::BF16;
    if (ns == ns_int32) return T0DType::I32;
    if (ns == ns_int64) return T0DType::I64;
    if (ns == ns_int8) return T0DType::I8;
    if (ns == ns_uint8) return T0DType::U8;
    if (ns == ns_bool) return T0DType::BOOL;
    return T0DType::INVALID;
}

static inline bool t0_is_float(T0DType dt) {
    return dt == T0DType::F32 || dt == T0DType::F64 || dt == T0DType::F16 || dt == T0DType::BF16;
}

static inline T0UnaryOp to_t0_unary(NanoString ns) {
    if (ns == ns_cast) return T0UnaryOp::Cast;
    if (ns == ns_abs) return T0UnaryOp::Abs;
    if (ns == ns_negative) return T0UnaryOp::Negative;
    if (ns == ns_logical_not) return T0UnaryOp::LogicalNot;
    if (ns == ns_bitwise_not) return T0UnaryOp::BitwiseNot;
    if (ns == ns_conj) return T0UnaryOp::Conj;
    if (ns == ns_relu) return T0UnaryOp::Relu;
    if (ns == ns_exp) return T0UnaryOp::Exp;
    if (ns == ns_log) return T0UnaryOp::Log;
    if (ns == ns_log1p) return T0UnaryOp::Log1p;
    if (ns == ns_sqrt) return T0UnaryOp::Sqrt;
    if (ns == ns_round) return T0UnaryOp::Round;
    if (ns == ns_floor) return T0UnaryOp::Floor;
    if (ns == ns_ceil) return T0UnaryOp::Ceil;
    if (ns == ns_round_int) return T0UnaryOp::RoundInt;
    if (ns == ns_floor_int) return T0UnaryOp::FloorInt;
    if (ns == ns_ceil_int) return T0UnaryOp::CeilInt;
    if (ns == ns_sin) return T0UnaryOp::Sin;
    if (ns == ns_asin) return T0UnaryOp::Asin;
    if (ns == ns_sinh) return T0UnaryOp::Sinh;
    if (ns == ns_asinh) return T0UnaryOp::Asinh;
    if (ns == ns_cos) return T0UnaryOp::Cos;
    if (ns == ns_acos) return T0UnaryOp::Acos;
    if (ns == ns_cosh) return T0UnaryOp::Cosh;
    if (ns == ns_acosh) return T0UnaryOp::Acosh;
    if (ns == ns_tan) return T0UnaryOp::Tan;
    if (ns == ns_atan) return T0UnaryOp::Atan;
    if (ns == ns_tanh) return T0UnaryOp::Tanh;
    if (ns == ns_atanh) return T0UnaryOp::Atanh;
    if (ns == ns_sigmoid) return T0UnaryOp::Sigmoid;
    if (ns == ns_erf) return T0UnaryOp::Erf;
    if (ns == ns_erfinv) return T0UnaryOp::Erfinv;
    return T0UnaryOp::Invalid;
}

static inline T0BinaryOp to_t0_binary(NanoString ns) {
    if (ns == ns_add) return T0BinaryOp::Add;
    if (ns == ns_subtract) return T0BinaryOp::Subtract;
    if (ns == ns_multiply) return T0BinaryOp::Multiply;
    if (ns == ns_divide) return T0BinaryOp::Divide;
    if (ns == ns_floor_divide) return T0BinaryOp::FloorDivide;
    if (ns == ns_mod) return T0BinaryOp::Mod;
    if (ns == ns_fmod) return T0BinaryOp::Fmod;
    if (ns == ns_pow) return T0BinaryOp::Pow;
    if (ns == ns_maximum) return T0BinaryOp::Maximum;
    if (ns == ns_minimum) return T0BinaryOp::Minimum;
    if (ns == ns_less) return T0BinaryOp::Less;
    if (ns == ns_less_equal) return T0BinaryOp::LessEqual;
    if (ns == ns_greater) return T0BinaryOp::Greater;
    if (ns == ns_greater_equal) return T0BinaryOp::GreaterEqual;
    if (ns == ns_equal) return T0BinaryOp::Equal;
    if (ns == ns_not_equal) return T0BinaryOp::NotEqual;
    if (ns == ns_logical_and) return T0BinaryOp::LogicalAnd;
    if (ns == ns_logical_or) return T0BinaryOp::LogicalOr;
    if (ns == ns_logical_xor) return T0BinaryOp::LogicalXor;
    if (ns == ns_bitwise_and) return T0BinaryOp::BitwiseAnd;
    if (ns == ns_bitwise_or) return T0BinaryOp::BitwiseOr;
    if (ns == ns_bitwise_xor) return T0BinaryOp::BitwiseXor;
    if (ns == ns_left_shift) return T0BinaryOp::LeftShift;
    if (ns == ns_right_shift) return T0BinaryOp::RightShift;
    return T0BinaryOp::Invalid;
}

static inline T0ReduceOp to_t0_reduce(NanoString ns) {
    if (ns == ns_add) return T0ReduceOp::Add;
    if (ns == ns_multiply) return T0ReduceOp::Multiply;
    if (ns == ns_maximum) return T0ReduceOp::Maximum;
    if (ns == ns_minimum) return T0ReduceOp::Minimum;
    if (ns == ns_mean) return T0ReduceOp::Mean;
    if (ns == ns_logical_and) return T0ReduceOp::LogicalAnd;
    if (ns == ns_logical_or) return T0ReduceOp::LogicalOr;
    if (ns == ns_logical_xor) return T0ReduceOp::LogicalXor;
    if (ns == ns_bitwise_and) return T0ReduceOp::BitwiseAnd;
    if (ns == ns_bitwise_or) return T0ReduceOp::BitwiseOr;
    if (ns == ns_bitwise_xor) return T0ReduceOp::BitwiseXor;
    return T0ReduceOp::Invalid;
}

static bool var_bounds_ok(Var* v) {
    if (!v || v->dsize() <= 0) return false;
    if (to_t0_dtype(v->dtype()) == T0DType::INVALID) return false;
    if (v->shape.size() > 8) return false;
    if (v->num < 0 || v->num >= std::numeric_limits<int32_t>::max()) return false;
    if (v->storage_span_bytes() / v->dsize() >= std::numeric_limits<int32_t>::max()) return false;
    for (size_t d = 0; d < v->shape.size(); ++d) {
        if (v->shape[d] < 0 || v->shape[d] >= std::numeric_limits<int32_t>::max()) return false;
        int64 st = v->storage_stride((uint)d);
        if (st < 0 || st >= std::numeric_limits<int32_t>::max()) return false;
    }
    return true;
}

static bool subop_supported(Op* op, BackendId backend) {
    if (!op) return false;
    if (!op->codegen().extra_flags.empty()) return false;
    static const OpId id_array = op_ids::array();
    static const OpId id_unary = get_op_id("unary");
    static const OpId id_binary = get_op_id("binary");
    static const OpId id_ternary = get_op_id("ternary");
    static const OpId id_broadcast = get_op_id("broadcast_to");
    static const OpId id_reduce = get_op_id("reduce");

    OpId oid = op->type_id();
    if (oid == id_array) {
        auto* aop = static_cast<ArrayOp*>(op);
        return aop->output->num == 1 && aop->host_value_known() && var_bounds_ok(aop->output);
    }
    if (oid == id_unary) {
        auto* uop = static_cast<UnaryOp*>(op);
        if (uop->layout_value) return false;
        if (!var_bounds_ok(uop->x) || !var_bounds_ok(uop->y) || !uop->y->is_contiguous()) return false;
        T0UnaryOp uo = to_t0_unary(uop->ns);
        if (uo == T0UnaryOp::Invalid) return false;
        T0DType tx = to_t0_dtype(uop->x->dtype());
        T0DType ty = to_t0_dtype(uop->y->dtype());
        if (uo == T0UnaryOp::Negative && (tx == T0DType::BOOL || ty == T0DType::BOOL)) return false;
        if (uo == T0UnaryOp::BitwiseNot && (t0_is_float(tx) || t0_is_float(ty))) return false;
        if (uo >= T0UnaryOp::Exp && !t0_is_float(ty) &&
            uo != T0UnaryOp::RoundInt && uo != T0UnaryOp::FloorInt && uo != T0UnaryOp::CeilInt)
            return false;
        return true;
    }
    if (oid == id_binary) {
        auto* bop = static_cast<BinaryOp*>(op);
        if (bop->layout_value) return false;
        if (!var_bounds_ok(bop->x) || !var_bounds_ok(bop->y) || !var_bounds_ok(bop->z) || !bop->z->is_contiguous())
            return false;
        T0BinaryOp bo = to_t0_binary(bop->ns);
        if (bo == T0BinaryOp::Invalid) return false;
        T0DType tx = to_t0_dtype(bop->x->dtype());
        T0DType ty = to_t0_dtype(bop->y->dtype());
        T0DType tz = to_t0_dtype(bop->z->dtype());
        if (bo == T0BinaryOp::Divide && !t0_is_float(tz)) return false;
        if (tz == T0DType::BOOL) {
            bool ok = (bo == T0BinaryOp::Add || bo == T0BinaryOp::Subtract || bo == T0BinaryOp::Multiply ||
                       bo == T0BinaryOp::Maximum || bo == T0BinaryOp::Minimum ||
                       (bo >= T0BinaryOp::Less && bo <= T0BinaryOp::BitwiseXor));
            if (!ok) return false;
        }
        if (bo >= T0BinaryOp::BitwiseAnd && (t0_is_float(tx) || t0_is_float(ty) || t0_is_float(tz)))
            return false;
        return true;
    }
    if (oid == id_ternary) {
        auto* top = static_cast<TernaryOp*>(op);
        return var_bounds_ok(top->cond) && var_bounds_ok(top->x) &&
               var_bounds_ok(top->y) && var_bounds_ok(top->z) && top->z->is_contiguous();
    }
    if (oid == id_broadcast) {
        auto* btop = static_cast<BroadcastToOp*>(op);
        if (btop->expand_is_view) return false;
        if (btop->x->dtype() != btop->z->dtype()) return false;
        return var_bounds_ok(btop->x) && var_bounds_ok(btop->z) && btop->z->is_contiguous();
    }
    if (oid == id_reduce) {
        auto* rop = static_cast<ReduceOp*>(op);
        if (!var_bounds_ok(rop->x) || !var_bounds_ok(rop->y) || !rop->y->is_contiguous()) return false;
        T0ReduceOp ro = to_t0_reduce(rop->ns);
        if (ro == T0ReduceOp::Invalid) return false;
        T0DType ty = to_t0_dtype(rop->y->dtype());
        if (ro >= T0ReduceOp::BitwiseAnd && t0_is_float(ty)) return false;
        return true;
    }
    (void)backend;
    return false;
}

static bool backend_tier0_allowed(BackendId backend) {
    if (backend == BackendId::Cpu) return true;
#if defined(IS_CUDA) && !defined(IS_COREX)
    if (backend == BackendId::Cuda) {
        // Precompiled CUDA Tier-0 kernels use strict IEEE math; fall back to
        // compiled JIT / NVRTC when caller explicitly selects fast or backend math.
        return runtime_flag_cuda_kernel_math() == "strict";
    }
#endif
    return false;
}

// =========================================================================
// Shape metadata builders
// =========================================================================

static T0ShapeMeta build_shape_meta(Var* ref_out, Var* x, Var* y = nullptr, Var* cond = nullptr) {
    T0ShapeMeta m{};
    m.rank = (int32_t)ref_out->shape.size();
    m.num = (int32_t)ref_out->num;
    for (int d = 0; d < m.rank; ++d) m.shape[d] = (int32_t)ref_out->shape[d];
    int32_t acc = 1;
    for (int d = m.rank - 1; d >= 0; --d) {
        m.above[d] = acc;
        acc *= m.shape[d];
    }
    if (x) {
        m.xstrided = x->is_contiguous() ? 0 : 1;
        if (m.xstrided) {
            for (int d = 0; d < m.rank; ++d) m.xstride[d] = (int32_t)x->storage_stride((uint)d);
        }
    }
    if (y) {
        m.ystrided = y->is_contiguous() ? 0 : 1;
        if (m.ystrided) {
            for (int d = 0; d < m.rank; ++d) m.ystride[d] = (int32_t)y->storage_stride((uint)d);
        }
    }
    if (cond) {
        m.cstrided = cond->is_contiguous() ? 0 : 1;
        if (m.cstrided) {
            for (int d = 0; d < m.rank; ++d) m.cstride[d] = (int32_t)cond->storage_stride((uint)d);
        }
    }
    return m;
}

static T0ShapeMeta build_broadcast_meta(BroadcastToOp* btop) {
    T0ShapeMeta m{};
    m.rank = (int32_t)btop->z->shape.size();
    m.num = (int32_t)btop->z->num;
    for (int d = 0; d < m.rank; ++d) m.shape[d] = (int32_t)btop->z->shape[d];
    int32_t acc = 1;
    for (int d = m.rank - 1; d >= 0; --d) {
        m.above[d] = acc;
        acc *= m.shape[d];
    }
    if (btop->x->num == 1) {
        m.xstrided = 0;  // broadcast kernel reads x[0] when !xstrided
    } else {
        m.xstrided = 1;
        int xi = (int)btop->x->shape.size() - 1;
        for (int d = m.rank - 1; d >= 0; --d) {
            if ((btop->bcast_mask >> d) & 1u) {
                m.xstride[d] = 0;
                if ((btop->keepdims_mask >> d) & 1u) --xi;
            } else {
                m.xstride[d] = (xi >= 0) ? (int32_t)btop->x->storage_stride((uint)xi) : 0;
                --xi;
            }
        }
    }
    return m;
}

static T0ReduceMeta build_reduce_meta(ReduceOp* rop) {
    T0ReduceMeta m{};
    int xdim = (int)rop->x->shape.size();
    m.out_num = (int32_t)rop->y->num;
    m.red_num = (m.out_num > 0) ? (int32_t)(rop->x->num / rop->y->num) : 0;
    for (int d = 0; d < xdim; ++d) {
        int32_t sz = (int32_t)rop->x->shape[d];
        int32_t st = (int32_t)rop->x->storage_stride((uint)d);
        if ((rop->reduce_mask >> d) & 1u) {
            m.red_shape[m.red_rank] = sz;
            m.red_xstride[m.red_rank] = st;
            m.red_rank++;
        } else {
            m.out_shape[m.out_rank] = sz;
            m.out_xstride[m.out_rank] = st;
            m.out_rank++;
        }
    }
    int32_t acc = 1;
    for (int d = m.out_rank - 1; d >= 0; --d) {
        m.out_above[d] = acc;
        acc *= m.out_shape[d];
    }
    acc = 1;
    for (int d = m.red_rank - 1; d >= 0; --d) {
        m.red_above[d] = acc;
        acc *= m.red_shape[d];
    }
    return m;
}

// =========================================================================
// CPU Tier-0 Meta-Op Kernels (Exact Jittor Numerics)
// =========================================================================

static inline int32_t cpu_offset(int32_t idx, int32_t rank,
                                 const int32_t* shape,
                                 const int32_t* above,
                                 const int32_t* stride) {
    int32_t off = 0;
    for (int32_t d = 0; d < rank; ++d) {
        if (stride[d] == 0) continue;
        int32_t c;
        if (rank == 1) c = idx;
        else if (d == 0) c = idx / above[0];
        else if (d == rank - 1) c = idx % shape[d];
        else c = (idx / above[d]) % shape[d];
        off += c * stride[d];
    }
    return off;
}

static inline uint16_t cpu_f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t u = x & 0x7fffffffu;
    if (u > 0x7f800000u) return 0x7fffu;
    uint32_t sign = (x >> 16) & 0x8000u;
    if (u > 0x477fefffu) return (uint16_t)(sign | 0x7c00u);
    if (u < 0x33000001u) return (uint16_t)sign;
    uint32_t exponent = (u >> 23) & 0xffu;
    uint32_t mantissa = u & 0x7fffffu;
    uint32_t shift;
    if (exponent > 0x70u) {
        shift = 13u;
        exponent -= 0x70u;
    } else {
        shift = 0x7eu - exponent;
        exponent = 0u;
        mantissa |= 0x800000u;
    }
    uint32_t lsb = 1u << shift;
    uint32_t lsb_s1 = lsb >> 1;
    uint32_t lsb_m1 = lsb - 1u;
    uint32_t remainder = mantissa & lsb_m1;
    mantissa >>= shift;
    if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 1u))) {
        ++mantissa;
        if (!(mantissa & 0x3ffu)) {
            ++exponent;
            mantissa = 0u;
        }
    }
    return (uint16_t)(sign | (exponent << 10) | mantissa);
}

static inline float cpu_f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u;
    uint32_t exponent = (h >> 10) & 0x1fu;
    uint32_t mantissa = (uint32_t)(h & 0x3ffu) << 13;
    if (exponent == 0x1fu) {
        if (mantissa) { sign = 0u; mantissa = 0x7fffffu; }
        exponent = 0xffu;
    } else if (!exponent) {
        if (mantissa) {
            exponent = 0x71u;
            while (!(mantissa & 0x400000u)) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x7fffffu;
        }
    } else {
        exponent += 0x70u;
    }
    uint32_t bits = (sign << 31) | (exponent << 23) | mantissa;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline uint16_t cpu_f32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
    uint32_t lsb = (bits >> 16) & 1u;
    return (uint16_t)((bits + 0x7fffu + lsb) >> 16);
}

static inline float cpu_bf16_to_f32(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline float cpu_load_f32(const void* p, int32_t idx, T0DType dt) {
    switch (dt) {
        case T0DType::F32:  return static_cast<const float*>(p)[idx];
        case T0DType::F64:  return static_cast<float>(static_cast<const double*>(p)[idx]);
        case T0DType::F16:  return cpu_f16_to_f32(static_cast<const uint16_t*>(p)[idx]);
        case T0DType::BF16: return cpu_bf16_to_f32(static_cast<const uint16_t*>(p)[idx]);
        case T0DType::I32:  return static_cast<float>(static_cast<const int32_t*>(p)[idx]);
        case T0DType::I64:  return static_cast<float>(static_cast<const int64_t*>(p)[idx]);
        case T0DType::I8:   return static_cast<float>(static_cast<const int8_t*>(p)[idx]);
        case T0DType::U8:   return static_cast<float>(static_cast<const uint8_t*>(p)[idx]);
        case T0DType::BOOL: return static_cast<const uint8_t*>(p)[idx] ? 1.0f : 0.0f;
        default: return 0.0f;
    }
}

static inline double cpu_load_f64(const void* p, int32_t idx, T0DType dt) {
    switch (dt) {
        case T0DType::F64:  return static_cast<const double*>(p)[idx];
        case T0DType::F32:  return static_cast<double>(static_cast<const float*>(p)[idx]);
        case T0DType::F16:  return static_cast<double>(cpu_f16_to_f32(static_cast<const uint16_t*>(p)[idx]));
        case T0DType::BF16: return static_cast<double>(cpu_bf16_to_f32(static_cast<const uint16_t*>(p)[idx]));
        case T0DType::I32:  return static_cast<double>(static_cast<const int32_t*>(p)[idx]);
        case T0DType::I64:  return static_cast<double>(static_cast<const int64_t*>(p)[idx]);
        case T0DType::I8:   return static_cast<double>(static_cast<const int8_t*>(p)[idx]);
        case T0DType::U8:   return static_cast<double>(static_cast<const uint8_t*>(p)[idx]);
        case T0DType::BOOL: return static_cast<const uint8_t*>(p)[idx] ? 1.0 : 0.0;
        default: return 0.0;
    }
}

static inline int64_t cpu_load_i64(const void* p, int32_t idx, T0DType dt) {
    switch (dt) {
        case T0DType::I64:  return static_cast<const int64_t*>(p)[idx];
        case T0DType::I32:  return static_cast<int64_t>(static_cast<const int32_t*>(p)[idx]);
        case T0DType::I8:   return static_cast<int64_t>(static_cast<const int8_t*>(p)[idx]);
        case T0DType::U8:   return static_cast<int64_t>(static_cast<const uint8_t*>(p)[idx]);
        case T0DType::BOOL: return static_cast<const uint8_t*>(p)[idx] ? 1LL : 0LL;
        case T0DType::F32:  return static_cast<int64_t>(static_cast<const float*>(p)[idx]);
        case T0DType::F64:  return static_cast<int64_t>(static_cast<const double*>(p)[idx]);
        case T0DType::F16:  return static_cast<int64_t>(cpu_f16_to_f32(static_cast<const uint16_t*>(p)[idx]));
        case T0DType::BF16: return static_cast<int64_t>(cpu_bf16_to_f32(static_cast<const uint16_t*>(p)[idx]));
        default: return 0LL;
    }
}

static inline bool cpu_load_bool(const void* p, int32_t idx, T0DType dt) {
    switch (dt) {
        case T0DType::BOOL: return static_cast<const uint8_t*>(p)[idx] != 0;
        case T0DType::I8:   return static_cast<const int8_t*>(p)[idx] != 0;
        case T0DType::U8:   return static_cast<const uint8_t*>(p)[idx] != 0;
        case T0DType::I32:  return static_cast<const int32_t*>(p)[idx] != 0;
        case T0DType::I64:  return static_cast<const int64_t*>(p)[idx] != 0;
        case T0DType::F32:  return static_cast<const float*>(p)[idx] != 0.0f;
        case T0DType::F64:  return static_cast<const double*>(p)[idx] != 0.0;
        case T0DType::F16:  return cpu_f16_to_f32(static_cast<const uint16_t*>(p)[idx]) != 0.0f;
        case T0DType::BF16: return cpu_bf16_to_f32(static_cast<const uint16_t*>(p)[idx]) != 0.0f;
        default: return false;
    }
}

static inline void cpu_store_f32(void* p, int32_t idx, T0DType dt, float v) {
    switch (dt) {
        case T0DType::F32:  static_cast<float*>(p)[idx] = v; break;
        case T0DType::F64:  static_cast<double*>(p)[idx] = static_cast<double>(v); break;
        case T0DType::F16:  static_cast<uint16_t*>(p)[idx] = cpu_f32_to_f16(v); break;
        case T0DType::BF16: static_cast<uint16_t*>(p)[idx] = cpu_f32_to_bf16(v); break;
        case T0DType::I32:  static_cast<int32_t*>(p)[idx] = static_cast<int32_t>(v); break;
        case T0DType::I64:  static_cast<int64_t*>(p)[idx] = static_cast<int64_t>(v); break;
        case T0DType::I8:   static_cast<int8_t*>(p)[idx] = static_cast<int8_t>(v); break;
        case T0DType::U8:   static_cast<uint8_t*>(p)[idx] = static_cast<uint8_t>(v); break;
        case T0DType::BOOL: static_cast<uint8_t*>(p)[idx] = (v != 0.0f) ? 1 : 0; break;
        default: break;
    }
}

static inline void cpu_store_f64(void* p, int32_t idx, T0DType dt, double v) {
    switch (dt) {
        case T0DType::F64:  static_cast<double*>(p)[idx] = v; break;
        case T0DType::F32:  static_cast<float*>(p)[idx] = static_cast<float>(v); break;
        case T0DType::F16:  static_cast<uint16_t*>(p)[idx] = cpu_f32_to_f16(static_cast<float>(v)); break;
        case T0DType::BF16: static_cast<uint16_t*>(p)[idx] = cpu_f32_to_bf16(static_cast<float>(v)); break;
        case T0DType::I32:  static_cast<int32_t*>(p)[idx] = static_cast<int32_t>(v); break;
        case T0DType::I64:  static_cast<int64_t*>(p)[idx] = static_cast<int64_t>(v); break;
        case T0DType::I8:   static_cast<int8_t*>(p)[idx] = static_cast<int8_t>(v); break;
        case T0DType::U8:   static_cast<uint8_t*>(p)[idx] = static_cast<uint8_t>(v); break;
        case T0DType::BOOL: static_cast<uint8_t*>(p)[idx] = (v != 0.0) ? 1 : 0; break;
        default: break;
    }
}

static inline void cpu_store_i64(void* p, int32_t idx, T0DType dt, int64_t v) {
    switch (dt) {
        case T0DType::I64:  static_cast<int64_t*>(p)[idx] = v; break;
        case T0DType::I32:  static_cast<int32_t*>(p)[idx] = static_cast<int32_t>(v); break;
        case T0DType::I8:   static_cast<int8_t*>(p)[idx] = static_cast<int8_t>(v); break;
        case T0DType::U8:   static_cast<uint8_t*>(p)[idx] = static_cast<uint8_t>(v); break;
        case T0DType::BOOL: static_cast<uint8_t*>(p)[idx] = (v != 0) ? 1 : 0; break;
        case T0DType::F32:  static_cast<float*>(p)[idx] = static_cast<float>(v); break;
        case T0DType::F64:  static_cast<double*>(p)[idx] = static_cast<double>(v); break;
        case T0DType::F16:  static_cast<uint16_t*>(p)[idx] = cpu_f32_to_f16(static_cast<float>(v)); break;
        case T0DType::BF16: static_cast<uint16_t*>(p)[idx] = cpu_f32_to_bf16(static_cast<float>(v)); break;
        default: break;
    }
}

static inline float cpu_apply_unary_f32(T0UnaryOp op, float x, T0DType ty) {
    switch (op) {
        case T0UnaryOp::Cast:
        case T0UnaryOp::Conj:     return x;
        case T0UnaryOp::Abs:      return std::fabs(x);
        case T0UnaryOp::Negative: return -x;
        case T0UnaryOp::Relu:     return x > 0.0f ? x : 0.0f;
        case T0UnaryOp::Exp:      return std::exp(x);
        case T0UnaryOp::Log:      return std::log(x);
        case T0UnaryOp::Log1p:    return std::log1p(x);
        case T0UnaryOp::Sqrt:     return std::sqrt(x);
        case T0UnaryOp::Round:    return std::nearbyint(x);
        case T0UnaryOp::Floor:    return std::floor(x);
        case T0UnaryOp::Ceil:     return std::ceil(x);
        case T0UnaryOp::RoundInt: return std::round(x);
        case T0UnaryOp::FloorInt: return std::floor(x);
        case T0UnaryOp::CeilInt:  return std::ceil(x);
        case T0UnaryOp::Sin:      return std::sin(x);
        case T0UnaryOp::Asin:     return std::asin(x);
        case T0UnaryOp::Sinh:     return std::sinh(x);
        case T0UnaryOp::Asinh:    return std::asinh(x);
        case T0UnaryOp::Cos:      return std::cos(x);
        case T0UnaryOp::Acos:     return std::acos(x);
        case T0UnaryOp::Cosh:     return std::cosh(x);
        case T0UnaryOp::Acosh:    return std::acosh(x);
        case T0UnaryOp::Tan:      return std::tan(x);
        case T0UnaryOp::Atan:     return std::atan(x);
        case T0UnaryOp::Tanh:     return std::tanh(x);
        case T0UnaryOp::Atanh:    return std::atanh(x);
        case T0UnaryOp::Sigmoid: {
            if (ty == T0DType::F16) return 1.0f / (1.0f + std::exp(std::min(-x, 30.0f)));
            if (ty == T0DType::BF16) return 1.0f / (1.0f + std::exp(std::min(-x, 300.0f)));
            return 1.0f / (1.0f + std::exp(-x));
        }
        case T0UnaryOp::Erf:      return (ty == T0DType::F32) ? _fast_erff(x) : std::erf(x);
        case T0UnaryOp::Erfinv:   return _erfinv(x);
        default: return x;
    }
}

static inline double cpu_apply_unary_f64(T0UnaryOp op, double x) {
    switch (op) {
        case T0UnaryOp::Cast:
        case T0UnaryOp::Conj:     return x;
        case T0UnaryOp::Abs:      return std::fabs(x);
        case T0UnaryOp::Negative: return -x;
        case T0UnaryOp::Relu:     return x > 0.0 ? x : 0.0;
        case T0UnaryOp::Exp:      return std::exp(x);
        case T0UnaryOp::Log:      return std::log(x);
        case T0UnaryOp::Log1p:    return std::log1p(x);
        case T0UnaryOp::Sqrt:     return std::sqrt(x);
        case T0UnaryOp::Round:    return std::nearbyint(x);
        case T0UnaryOp::Floor:    return std::floor(x);
        case T0UnaryOp::Ceil:     return std::ceil(x);
        case T0UnaryOp::RoundInt: return std::round(x);
        case T0UnaryOp::FloorInt: return std::floor(x);
        case T0UnaryOp::CeilInt:  return std::ceil(x);
        case T0UnaryOp::Sin:      return std::sin(x);
        case T0UnaryOp::Asin:     return std::asin(x);
        case T0UnaryOp::Sinh:     return std::sinh(x);
        case T0UnaryOp::Asinh:    return std::asinh(x);
        case T0UnaryOp::Cos:      return std::cos(x);
        case T0UnaryOp::Acos:     return std::acos(x);
        case T0UnaryOp::Cosh:     return std::cosh(x);
        case T0UnaryOp::Acosh:    return std::acosh(x);
        case T0UnaryOp::Tan:      return std::tan(x);
        case T0UnaryOp::Atan:     return std::atan(x);
        case T0UnaryOp::Tanh:     return std::tanh(x);
        case T0UnaryOp::Atanh:    return std::atanh(x);
        case T0UnaryOp::Sigmoid:  return 1.0 / (1.0 + std::exp(-x));
        case T0UnaryOp::Erf:      return std::erf(x);
        case T0UnaryOp::Erfinv:   return _erfinv(x);
        default: return x;
    }
}

static void cpu_tier0_unary(T0UnaryOp op, T0DType tx, T0DType ty,
                            const void* x, void* y, const T0ShapeMeta& m) {
    if (m.num <= 0) return;
    // Fast contiguous float32 path
    if (!m.xstrided && tx == T0DType::F32 && ty == T0DType::F32 &&
        op != T0UnaryOp::LogicalNot && op != T0UnaryOp::BitwiseNot) {
        const float* xp = static_cast<const float*>(x);
        float* yp = static_cast<float*>(y);
        for (int32_t i = 0; i < m.num; ++i)
            yp[i] = cpu_apply_unary_f32(op, xp[i], T0DType::F32);
        return;
    }
    for (int32_t i = 0; i < m.num; ++i) {
        int32_t xi = m.xstrided ? cpu_offset(i, m.rank, m.shape, m.above, m.xstride) : i;
        if (op == T0UnaryOp::LogicalNot) {
            static_cast<uint8_t*>(y)[i] = cpu_load_bool(x, xi, tx) ? 0 : 1;
        } else if (op == T0UnaryOp::BitwiseNot) {
            if (ty == T0DType::BOOL)
                static_cast<uint8_t*>(y)[i] = cpu_load_bool(x, xi, tx) ? 0 : 1;
            else
                cpu_store_i64(y, i, ty, ~cpu_load_i64(x, xi, tx));
        } else if (op == T0UnaryOp::Cast && ty == T0DType::BOOL) {
            static_cast<uint8_t*>(y)[i] = cpu_load_bool(x, xi, tx) ? 1 : 0;
        } else if (ty == T0DType::F64 ||
                   (tx == T0DType::F64 && (op == T0UnaryOp::Cast ||
                    op == T0UnaryOp::RoundInt || op == T0UnaryOp::FloorInt || op == T0UnaryOp::CeilInt))) {
            cpu_store_f64(y, i, ty, cpu_apply_unary_f64(op, cpu_load_f64(x, xi, tx)));
        } else if (t0_is_float(ty) || op == T0UnaryOp::RoundInt || op == T0UnaryOp::FloorInt || op == T0UnaryOp::CeilInt) {
            cpu_store_f32(y, i, ty, cpu_apply_unary_f32(op, cpu_load_f32(x, xi, tx), ty));
        } else {
            int64_t v = cpu_load_i64(x, xi, tx);
            if (op == T0UnaryOp::Abs) v = (v < 0) ? (int64_t)(0ULL - (uint64_t)v) : v;
            else if (op == T0UnaryOp::Negative) v = (int64_t)(0ULL - (uint64_t)v);
            else if (op == T0UnaryOp::Relu) v = (v > 0) ? v : 0;
            cpu_store_i64(y, i, ty, v);
        }
    }
}

static void cpu_tier0_binary(T0BinaryOp op, T0DType tx, T0DType ty, T0DType tz,
                             const void* x, const void* y, void* z, const T0ShapeMeta& m) {
    if (m.num <= 0) return;
    // Fast contiguous float32 arithmetic paths
    if (!m.xstrided && !m.ystrided && tx == T0DType::F32 && ty == T0DType::F32 && tz == T0DType::F32) {
        const float* xp = static_cast<const float*>(x);
        const float* yp = static_cast<const float*>(y);
        float* zp = static_cast<float*>(z);
        switch (op) {
            case T0BinaryOp::Add:      for (int32_t i=0;i<m.num;++i) zp[i] = xp[i] + yp[i]; return;
            case T0BinaryOp::Subtract: for (int32_t i=0;i<m.num;++i) zp[i] = xp[i] - yp[i]; return;
            case T0BinaryOp::Multiply: for (int32_t i=0;i<m.num;++i) zp[i] = xp[i] * yp[i]; return;
            case T0BinaryOp::Divide:   for (int32_t i=0;i<m.num;++i) zp[i] = xp[i] / yp[i]; return;
            case T0BinaryOp::Maximum:  for (int32_t i=0;i<m.num;++i) zp[i] = _max(xp[i], yp[i]); return;
            case T0BinaryOp::Minimum:  for (int32_t i=0;i<m.num;++i) zp[i] = _min(xp[i], yp[i]); return;
            default: break;
        }
    }
    for (int32_t i = 0; i < m.num; ++i) {
        int32_t xi = m.xstrided ? cpu_offset(i, m.rank, m.shape, m.above, m.xstride) : i;
        int32_t yi = m.ystrided ? cpu_offset(i, m.rank, m.shape, m.above, m.ystride) : i;

        if (op == T0BinaryOp::LogicalAnd || op == T0BinaryOp::LogicalOr || op == T0BinaryOp::LogicalXor) {
            bool a = cpu_load_bool(x, xi, tx);
            bool b = cpu_load_bool(y, yi, ty);
            bool r = (op == T0BinaryOp::LogicalAnd) ? (a && b)
                   : (op == T0BinaryOp::LogicalOr)  ? (a || b) : (a != b);
            static_cast<uint8_t*>(z)[i] = r ? 1 : 0;
            continue;
        }
        if (op == T0BinaryOp::Less || op == T0BinaryOp::LessEqual ||
            op == T0BinaryOp::Greater || op == T0BinaryOp::GreaterEqual ||
            op == T0BinaryOp::Equal || op == T0BinaryOp::NotEqual) {
            bool r = false;
            if (tx == T0DType::F64 || ty == T0DType::F64) {
                double a = cpu_load_f64(x, xi, tx), b = cpu_load_f64(y, yi, ty);
                switch (op) {
                    case T0BinaryOp::Less:         r = a < b; break;
                    case T0BinaryOp::LessEqual:    r = a <= b; break;
                    case T0BinaryOp::Greater:      r = a > b; break;
                    case T0BinaryOp::GreaterEqual: r = a >= b; break;
                    case T0BinaryOp::Equal:        r = a == b; break;
                    case T0BinaryOp::NotEqual:     r = a != b; break;
                    default: break;
                }
            } else if (t0_is_float(tx) || t0_is_float(ty)) {
                float a = cpu_load_f32(x, xi, tx), b = cpu_load_f32(y, yi, ty);
                switch (op) {
                    case T0BinaryOp::Less:         r = a < b; break;
                    case T0BinaryOp::LessEqual:    r = a <= b; break;
                    case T0BinaryOp::Greater:      r = a > b; break;
                    case T0BinaryOp::GreaterEqual: r = a >= b; break;
                    case T0BinaryOp::Equal:        r = a == b; break;
                    case T0BinaryOp::NotEqual:     r = a != b; break;
                    default: break;
                }
            } else {
                int64_t a = cpu_load_i64(x, xi, tx), b = cpu_load_i64(y, yi, ty);
                switch (op) {
                    case T0BinaryOp::Less:         r = a < b; break;
                    case T0BinaryOp::LessEqual:    r = a <= b; break;
                    case T0BinaryOp::Greater:      r = a > b; break;
                    case T0BinaryOp::GreaterEqual: r = a >= b; break;
                    case T0BinaryOp::Equal:        r = a == b; break;
                    case T0BinaryOp::NotEqual:     r = a != b; break;
                    default: break;
                }
            }
            static_cast<uint8_t*>(z)[i] = r ? 1 : 0;
            continue;
        }
        if (tz == T0DType::F64) {
            double a = cpu_load_f64(x, xi, tx), b = cpu_load_f64(y, yi, ty), r = 0.0;
            switch (op) {
                case T0BinaryOp::Add:         r = a + b; break;
                case T0BinaryOp::Subtract:    r = a - b; break;
                case T0BinaryOp::Multiply:    r = a * b; break;
                case T0BinaryOp::Divide:      r = a / b; break;
                case T0BinaryOp::FloorDivide: r = _floor_divide_float(a, b); break;
                case T0BinaryOp::Mod:         r = _floor_mod_float(a, b); break;
                case T0BinaryOp::Fmod:        r = _fmod_ieee(a, b); break;
                case T0BinaryOp::Pow:         r = std::pow(a, b); break;
                case T0BinaryOp::Maximum:     r = _max(a, b); break;
                case T0BinaryOp::Minimum:     r = _min(a, b); break;
                default: break;
            }
            static_cast<double*>(z)[i] = r;
        } else if (t0_is_float(tz)) {
            float a = cpu_load_f32(x, xi, tx), b = cpu_load_f32(y, yi, ty), r = 0.0f;
            switch (op) {
                case T0BinaryOp::Add:         r = a + b; break;
                case T0BinaryOp::Subtract:    r = a - b; break;
                case T0BinaryOp::Multiply:    r = a * b; break;
                case T0BinaryOp::Divide:      r = a / b; break;
                case T0BinaryOp::FloorDivide: r = _floor_divide_float(a, b); break;
                case T0BinaryOp::Mod:         r = _floor_mod_float(a, b); break;
                case T0BinaryOp::Fmod:        r = _fmod_ieee(a, b); break;
                case T0BinaryOp::Pow:         r = std::pow(a, b); break;
                case T0BinaryOp::Maximum:     r = _max(a, b); break;
                case T0BinaryOp::Minimum:     r = _min(a, b); break;
                default: break;
            }
            cpu_store_f32(z, i, tz, r);
        } else if (tz == T0DType::I32) {
            int32_t a = static_cast<int32_t>(cpu_load_i64(x, xi, tx));
            int32_t b = static_cast<int32_t>(cpu_load_i64(y, yi, ty));
            int32_t r = 0;
            switch (op) {
                case T0BinaryOp::Add:         r = a + b; break;
                case T0BinaryOp::Subtract:    r = a - b; break;
                case T0BinaryOp::Multiply:    r = a * b; break;
                case T0BinaryOp::FloorDivide: r = _floor_divide<int32_t>(a, b); break;
                case T0BinaryOp::Mod:         r = _floor_mod<int32_t>(a, b); break;
                case T0BinaryOp::Fmod:        r = _fmod_int<int32_t>(a, b); break;
                case T0BinaryOp::Pow:         r = static_cast<int32_t>(std::pow((double)a, (double)b)); break;
                case T0BinaryOp::Maximum:     r = _max(a, b); break;
                case T0BinaryOp::Minimum:     r = _min(a, b); break;
                case T0BinaryOp::BitwiseAnd:  r = a & b; break;
                case T0BinaryOp::BitwiseOr:   r = a | b; break;
                case T0BinaryOp::BitwiseXor:  r = a ^ b; break;
                case T0BinaryOp::LeftShift:   r = a << b; break;
                case T0BinaryOp::RightShift:  r = a >> b; break;
                default: break;
            }
            static_cast<int32_t*>(z)[i] = r;
        } else if (tz == T0DType::I64) {
            int64_t a = cpu_load_i64(x, xi, tx);
            int64_t b = cpu_load_i64(y, yi, ty);
            int64_t r = 0;
            switch (op) {
                case T0BinaryOp::Add:         r = a + b; break;
                case T0BinaryOp::Subtract:    r = a - b; break;
                case T0BinaryOp::Multiply:    r = a * b; break;
                case T0BinaryOp::FloorDivide: r = _floor_divide<int64_t>(a, b); break;
                case T0BinaryOp::Mod:         r = _floor_mod<int64_t>(a, b); break;
                case T0BinaryOp::Fmod:        r = _fmod_int<int64_t>(a, b); break;
                case T0BinaryOp::Pow:         r = static_cast<int64_t>(std::pow((double)a, (double)b)); break;
                case T0BinaryOp::Maximum:     r = _max(a, b); break;
                case T0BinaryOp::Minimum:     r = _min(a, b); break;
                case T0BinaryOp::BitwiseAnd:  r = a & b; break;
                case T0BinaryOp::BitwiseOr:   r = a | b; break;
                case T0BinaryOp::BitwiseXor:  r = a ^ b; break;
                case T0BinaryOp::LeftShift:   r = a << b; break;
                case T0BinaryOp::RightShift:  r = a >> b; break;
                default: break;
            }
            static_cast<int64_t*>(z)[i] = r;
        } else if (tz == T0DType::BOOL) {
            bool a = cpu_load_bool(x, xi, tx);
            bool b = cpu_load_bool(y, yi, ty);
            bool r = false;
            switch (op) {
                case T0BinaryOp::Add:
                case T0BinaryOp::BitwiseOr:
                case T0BinaryOp::Maximum:     r = a || b; break;
                case T0BinaryOp::Multiply:
                case T0BinaryOp::BitwiseAnd:
                case T0BinaryOp::Minimum:     r = a && b; break;
                case T0BinaryOp::BitwiseXor:
                case T0BinaryOp::Subtract:    r = a != b; break;
                default: r = false; break;
            }
            static_cast<uint8_t*>(z)[i] = r ? 1 : 0;
        } else {
            // I8 or U8
            int32_t a = static_cast<int32_t>(cpu_load_i64(x, xi, tx));
            int32_t b = static_cast<int32_t>(cpu_load_i64(y, yi, ty));
            int32_t r = 0;
            switch (op) {
                case T0BinaryOp::Add:         r = a + b; break;
                case T0BinaryOp::Subtract:    r = a - b; break;
                case T0BinaryOp::Multiply:    r = a * b; break;
                case T0BinaryOp::FloorDivide:
                    if (tz == T0DType::I8) r = _floor_divide<int8_t>((int8_t)a, (int8_t)b);
                    else r = _floor_divide<uint8_t>((uint8_t)a, (uint8_t)b);
                    break;
                case T0BinaryOp::Mod:
                    if (tz == T0DType::I8) r = _floor_mod<int8_t>((int8_t)a, (int8_t)b);
                    else r = _floor_mod<uint8_t>((uint8_t)a, (uint8_t)b);
                    break;
                case T0BinaryOp::Fmod:
                    if (tz == T0DType::I8) r = _fmod_int<int8_t>((int8_t)a, (int8_t)b);
                    else r = _fmod_int<uint8_t>((uint8_t)a, (uint8_t)b);
                    break;
                case T0BinaryOp::Pow:         r = static_cast<int32_t>(std::pow((double)a, (double)b)); break;
                case T0BinaryOp::Maximum:     r = a > b ? a : b; break;
                case T0BinaryOp::Minimum:     r = a < b ? a : b; break;
                case T0BinaryOp::BitwiseAnd:  r = a & b; break;
                case T0BinaryOp::BitwiseOr:   r = a | b; break;
                case T0BinaryOp::BitwiseXor:  r = a ^ b; break;
                case T0BinaryOp::LeftShift:   r = a << b; break;
                case T0BinaryOp::RightShift:  r = (tz == T0DType::I8) ? ((int8_t)a >> b) : ((uint8_t)a >> b); break;
                default: break;
            }
            cpu_store_i64(z, i, tz, r);
        }
    }
}

static void cpu_tier0_ternary(T0DType tc, T0DType tx, T0DType ty, T0DType tz,
                              const void* cond, const void* x, const void* y, void* z,
                              const T0ShapeMeta& m) {
    if (m.num <= 0) return;
    for (int32_t i = 0; i < m.num; ++i) {
        int32_t ci = m.cstrided ? cpu_offset(i, m.rank, m.shape, m.above, m.cstride) : i;
        int32_t xi = m.xstrided ? cpu_offset(i, m.rank, m.shape, m.above, m.xstride) : i;
        int32_t yi = m.ystrided ? cpu_offset(i, m.rank, m.shape, m.above, m.ystride) : i;
        bool c = cpu_load_bool(cond, ci, tc);
        if (tz == T0DType::F64) {
            static_cast<double*>(z)[i] = c ? cpu_load_f64(x, xi, tx) : cpu_load_f64(y, yi, ty);
        } else if (t0_is_float(tz)) {
            cpu_store_f32(z, i, tz, c ? cpu_load_f32(x, xi, tx) : cpu_load_f32(y, yi, ty));
        } else if (tz == T0DType::BOOL) {
            static_cast<uint8_t*>(z)[i] = (c ? cpu_load_bool(x, xi, tx) : cpu_load_bool(y, yi, ty)) ? 1 : 0;
        } else {
            cpu_store_i64(z, i, tz, c ? cpu_load_i64(x, xi, tx) : cpu_load_i64(y, yi, ty));
        }
    }
}

static void cpu_tier0_broadcast(T0DType tx, const void* x, void* z, const T0ShapeMeta& m) {
    if (m.num <= 0) return;
    if (!m.xstrided) {
        switch (tx) {
            case T0DType::BOOL:
            case T0DType::I8:
            case T0DType::U8:
                std::memset(z, static_cast<const uint8_t*>(x)[0], (size_t)m.num);
                return;
            case T0DType::F16:
            case T0DType::BF16: {
                uint16_t v = static_cast<const uint16_t*>(x)[0];
                auto* zp = static_cast<uint16_t*>(z);
                for (int32_t i = 0; i < m.num; ++i) zp[i] = v;
                return;
            }
            case T0DType::F32:
            case T0DType::I32: {
                uint32_t v = static_cast<const uint32_t*>(x)[0];
                auto* zp = static_cast<uint32_t*>(z);
                for (int32_t i = 0; i < m.num; ++i) zp[i] = v;
                return;
            }
            case T0DType::F64:
            case T0DType::I64: {
                uint64_t v = static_cast<const uint64_t*>(x)[0];
                auto* zp = static_cast<uint64_t*>(z);
                for (int32_t i = 0; i < m.num; ++i) zp[i] = v;
                return;
            }
            default: return;
        }
    }
    for (int32_t i = 0; i < m.num; ++i) {
        int32_t xi = cpu_offset(i, m.rank, m.shape, m.above, m.xstride);
        switch (tx) {
            case T0DType::BOOL:
            case T0DType::I8:
            case T0DType::U8:
                static_cast<uint8_t*>(z)[i] = static_cast<const uint8_t*>(x)[xi]; break;
            case T0DType::F16:
            case T0DType::BF16:
                static_cast<uint16_t*>(z)[i] = static_cast<const uint16_t*>(x)[xi]; break;
            case T0DType::F32:
            case T0DType::I32:
                static_cast<uint32_t*>(z)[i] = static_cast<const uint32_t*>(x)[xi]; break;
            case T0DType::F64:
            case T0DType::I64:
                static_cast<uint64_t*>(z)[i] = static_cast<const uint64_t*>(x)[xi]; break;
            default: break;
        }
    }
}

static void cpu_tier0_reduce(T0ReduceOp op, T0DType tx, T0DType ty,
                             const void* x, void* y, const T0ReduceMeta& m) {
    if (m.out_num <= 0) return;
    const bool is_logical = (op == T0ReduceOp::LogicalAnd || op == T0ReduceOp::LogicalOr || op == T0ReduceOp::LogicalXor);
    const bool use_float = !is_logical && t0_is_float(ty);

    auto init_f64 = [&]() -> double {
        switch (op) {
            case T0ReduceOp::Add:
            case T0ReduceOp::Mean:     return 0.0;
            case T0ReduceOp::Multiply: return 1.0;
            case T0ReduceOp::Maximum:  return -std::numeric_limits<double>::infinity();
            case T0ReduceOp::Minimum:  return std::numeric_limits<double>::infinity();
            default: return 0.0;
        }
    };
    auto init_i64 = [&]() -> int64_t {
        switch (op) {
            case T0ReduceOp::Add:
            case T0ReduceOp::LogicalOr:
            case T0ReduceOp::LogicalXor:
            case T0ReduceOp::BitwiseOr:
            case T0ReduceOp::BitwiseXor: return 0LL;
            case T0ReduceOp::Multiply:
            case T0ReduceOp::LogicalAnd: return 1LL;
            case T0ReduceOp::BitwiseAnd: return -1LL;
            case T0ReduceOp::Maximum:
                if (ty == T0DType::I8) return -128LL;
                if (ty == T0DType::U8 || ty == T0DType::BOOL) return 0LL;
                if (ty == T0DType::I32) return std::numeric_limits<int32_t>::lowest();
                return std::numeric_limits<int64_t>::lowest();
            case T0ReduceOp::Minimum:
                if (ty == T0DType::I8) return 127LL;
                if (ty == T0DType::U8) return 255LL;
                if (ty == T0DType::BOOL) return 1LL;
                if (ty == T0DType::I32) return std::numeric_limits<int32_t>::max();
                return std::numeric_limits<int64_t>::max();
            default: return 0LL;
        }
    };

    for (int32_t out_idx = 0; out_idx < m.out_num; ++out_idx) {
        if (m.red_num == 0) {
            if (op == T0ReduceOp::Mean) cpu_store_f64(y, out_idx, ty, std::numeric_limits<double>::quiet_NaN());
            else if (use_float) cpu_store_f64(y, out_idx, ty, init_f64());
            else cpu_store_i64(y, out_idx, ty, init_i64());
            continue;
        }
        int32_t base = cpu_offset(out_idx, m.out_rank, m.out_shape, m.out_above, m.out_xstride);
        if (use_float) {
            double acc = init_f64();
            for (int32_t r = 0; r < m.red_num; ++r) {
                int32_t off = base + cpu_offset(r, m.red_rank, m.red_shape, m.red_above, m.red_xstride);
                double v = cpu_load_f64(x, off, tx);
                switch (op) {
                    case T0ReduceOp::Add:
                    case T0ReduceOp::Mean:     acc += v; break;
                    case T0ReduceOp::Multiply:
                        acc *= v;
                        if (ty == T0DType::F16) acc = (double)cpu_f16_to_f32(cpu_f32_to_f16((float)acc));
                        else if (ty == T0DType::BF16) acc = (double)cpu_bf16_to_f32(cpu_f32_to_bf16((float)acc));
                        break;
                    case T0ReduceOp::Maximum:  acc = _max(acc, v); break;
                    case T0ReduceOp::Minimum:  acc = _min(acc, v); break;
                    default: break;
                }
            }
            if (op == T0ReduceOp::Mean) acc /= (double)m.red_num;
            cpu_store_f64(y, out_idx, ty, acc);
        } else {
            int64_t acc = init_i64();
            for (int32_t r = 0; r < m.red_num; ++r) {
                int32_t off = base + cpu_offset(r, m.red_rank, m.red_shape, m.red_above, m.red_xstride);
                int64_t v = is_logical ? (cpu_load_bool(x, off, tx) ? 1LL : 0LL) : cpu_load_i64(x, off, tx);
                switch (op) {
                    case T0ReduceOp::Add:        acc = (int64_t)((uint64_t)acc + (uint64_t)v); break;
                    case T0ReduceOp::Multiply:   acc = (int64_t)((uint64_t)acc * (uint64_t)v); break;
                    case T0ReduceOp::Maximum:    acc = acc > v ? acc : v; break;
                    case T0ReduceOp::Minimum:    acc = acc < v ? acc : v; break;
                    case T0ReduceOp::LogicalAnd: acc = (acc && v) ? 1LL : 0LL; break;
                    case T0ReduceOp::LogicalOr:  acc = (acc || v) ? 1LL : 0LL; break;
                    case T0ReduceOp::LogicalXor: acc = ((acc != 0) != (v != 0)) ? 1LL : 0LL; break;
                    case T0ReduceOp::BitwiseAnd: acc &= v; break;
                    case T0ReduceOp::BitwiseOr:  acc |= v; break;
                    case T0ReduceOp::BitwiseXor: acc ^= v; break;
                    default: break;
                }
            }
            cpu_store_i64(y, out_idx, ty, acc);
        }
    }
}

// =========================================================================
// Unified Single Sub-Op Tier-0 Dispatch (CPU or CUDA)
// =========================================================================

static void execute_tier0_subop(Op* op, BackendId backend) {
    static const OpId id_array = op_ids::array();
    static const OpId id_unary = get_op_id("unary");
    static const OpId id_binary = get_op_id("binary");
    static const OpId id_ternary = get_op_id("ternary");
    static const OpId id_broadcast = get_op_id("broadcast_to");
    static const OpId id_reduce = get_op_id("reduce");

    const bool is_cuda = (backend == BackendId::Cuda);
    OpId oid = op->type_id();

    if (oid == id_array) {
        auto* aop = static_cast<ArrayOp*>(op);
        if (aop->output->size == 0) return;
        const char* src = aop->ptr<char>();
        void* dst = aop->output->mem_ptr;
        if (!is_cuda) {
            std::memcpy(dst, src, (size_t)aop->output->size);
        } else {
#if defined(IS_CUDA) && !defined(IS_COREX)
            uint64_t raw = 0;
            std::memcpy(&raw, src, std::min<size_t>((size_t)aop->output->size, sizeof(raw)));
            cuda_tier0_fill_scalar(dst, aop->output->dsize(), raw, 1);
#endif
        }
        return;
    }

    if (oid == id_unary) {
        auto* uop = static_cast<UnaryOp*>(op);
        T0ShapeMeta meta = build_shape_meta(uop->y, uop->x);
        if (meta.num == 0) return;
        const void* xp = uop->x->ptr<void>();
        void* yp = uop->y->mem_ptr;
        T0UnaryOp uo = to_t0_unary(uop->ns);
        T0DType tx = to_t0_dtype(uop->x->dtype());
        T0DType ty = to_t0_dtype(uop->y->dtype());
        if (!is_cuda) cpu_tier0_unary(uo, tx, ty, xp, yp, meta);
#if defined(IS_CUDA) && !defined(IS_COREX)
        else cuda_tier0_unary(uo, tx, ty, xp, yp, meta);
#endif
        return;
    }

    if (oid == id_binary) {
        auto* bop = static_cast<BinaryOp*>(op);
        T0ShapeMeta meta = build_shape_meta(bop->z, bop->x, bop->y);
        if (meta.num == 0) return;
        const void* xp = bop->x->ptr<void>();
        const void* yp = bop->y->ptr<void>();
        void* zp = bop->z->mem_ptr;
        T0BinaryOp bo = to_t0_binary(bop->ns);
        T0DType tx = to_t0_dtype(bop->x->dtype());
        T0DType ty = to_t0_dtype(bop->y->dtype());
        T0DType tz = to_t0_dtype(bop->z->dtype());
        if (!is_cuda) cpu_tier0_binary(bo, tx, ty, tz, xp, yp, zp, meta);
#if defined(IS_CUDA) && !defined(IS_COREX)
        else cuda_tier0_binary(bo, tx, ty, tz, xp, yp, zp, meta);
#endif
        return;
    }

    if (oid == id_ternary) {
        auto* top = static_cast<TernaryOp*>(op);
        T0ShapeMeta meta = build_shape_meta(top->z, top->x, top->y, top->cond);
        if (meta.num == 0) return;
        const void* cp = top->cond->ptr<void>();
        const void* xp = top->x->ptr<void>();
        const void* yp = top->y->ptr<void>();
        void* zp = top->z->mem_ptr;
        T0DType tc = to_t0_dtype(top->cond->dtype());
        T0DType tx = to_t0_dtype(top->x->dtype());
        T0DType ty = to_t0_dtype(top->y->dtype());
        T0DType tz = to_t0_dtype(top->z->dtype());
        if (!is_cuda) cpu_tier0_ternary(tc, tx, ty, tz, cp, xp, yp, zp, meta);
#if defined(IS_CUDA) && !defined(IS_COREX)
        else cuda_tier0_ternary(tc, tx, ty, tz, cp, xp, yp, zp, meta);
#endif
        return;
    }

    if (oid == id_broadcast) {
        auto* btop = static_cast<BroadcastToOp*>(op);
        T0ShapeMeta meta = build_broadcast_meta(btop);
        if (meta.num == 0) return;
        const void* xp = btop->x->ptr<void>();
        void* zp = btop->z->mem_ptr;
        T0DType tx = to_t0_dtype(btop->x->dtype());
        if (!is_cuda) cpu_tier0_broadcast(tx, xp, zp, meta);
#if defined(IS_CUDA) && !defined(IS_COREX)
        else cuda_tier0_broadcast(tx, xp, zp, meta);
#endif
        return;
    }

    if (oid == id_reduce) {
        auto* rop = static_cast<ReduceOp*>(op);
        T0ReduceMeta meta = build_reduce_meta(rop);
        if (meta.out_num == 0) return;
        const void* xp = (rop->x->num > 0) ? rop->x->ptr<void>() : nullptr;
        void* yp = rop->y->mem_ptr;
        T0ReduceOp ro = to_t0_reduce(rop->ns);
        T0DType tx = to_t0_dtype(rop->x->dtype());
        T0DType ty = to_t0_dtype(rop->y->dtype());
        if (!is_cuda) cpu_tier0_reduce(ro, tx, ty, xp, yp, meta);
#if defined(IS_CUDA) && !defined(IS_COREX)
        else cuda_tier0_reduce(ro, tx, ty, xp, yp, meta);
#endif
        return;
    }
}

// =========================================================================
// RAII Temporary Buffer Guard for FusedOp Intermediates & Aliased Outputs
// =========================================================================

struct TempVarGuard {
    Var* var = nullptr;
    Allocation alloc;
    void* saved_mem_ptr = nullptr;
    Allocator* saved_allocator = nullptr;
    size_t saved_allocation = 0;

    TempVarGuard(Var* v, size_t bytes)
        : var(v),
          alloc(bytes ? runtime_executor().alloc_temp(bytes) : Allocation()),
          saved_mem_ptr(v->mem_ptr),
          saved_allocator(v->allocator),
          saved_allocation(v->allocation) {
        var->mem_ptr = alloc.ptr;
        var->allocator = alloc.allocator;
        var->allocation = alloc.allocation;
    }

    TempVarGuard(TempVarGuard&& other) noexcept
        : var(other.var),
          alloc(std::move(other.alloc)),
          saved_mem_ptr(other.saved_mem_ptr),
          saved_allocator(other.saved_allocator),
          saved_allocation(other.saved_allocation) {
        other.var = nullptr;
    }

    TempVarGuard(const TempVarGuard&) = delete;
    TempVarGuard& operator=(const TempVarGuard&) = delete;

    ~TempVarGuard() {
        if (var) {
            var->mem_ptr = saved_mem_ptr;
            var->allocator = saved_allocator;
            var->allocation = saved_allocation;
        }
    }
};

// =========================================================================
// Asynchronous Tier-1 Background Compilation Worker (ZERO Op*/Var* Pointers!)
// =========================================================================

struct BgCompileJob {
    bool is_fused = false;
    bool is_nvrtc = false;
    bool is_cuda = false;
    std::string prev_jit_key;
    std::string new_jit_key;
    std::string src;
    std::string extra_flags;
    NvrtcCompileSnapshot nvrtc_snap;
};

struct Tier1BgWorker {
    std::mutex mu;
    std::condition_variable cv;
    std::condition_variable idle_cv;
    std::deque<BgCompileJob> queue;
    std::unordered_map<std::string, int> hit_counts;
    std::unordered_set<std::string> enqueued;
    std::atomic<bool> stopping{false};
    bool active = false;
    std::thread* thread = nullptr;
};

// Leaked on purpose (same pattern as async_exec.cc's Worker*) so cleanup
// callbacks and exit teardown never race with static destructor ordering.
static Tier1BgWorker* bg_worker = new Tier1BgWorker();

static void bg_worker_main() {
    get_thread_name() = "T1BG";
    for (;;) {
        BgCompileJob job;
        {
            std::unique_lock<std::mutex> lock(bg_worker->mu);
            bg_worker->cv.wait(lock, []() {
                return bg_worker->stopping.load(std::memory_order_acquire) || !bg_worker->queue.empty();
            });
            if (bg_worker->stopping.load(std::memory_order_acquire)) break;
            job = std::move(bg_worker->queue.front());
            bg_worker->queue.pop_front();
            bg_worker->active = true;
        }
        try {
            if (job.is_nvrtc) {
                auto kernel = nvrtc_compile_snapshot(job.nvrtc_snap);
                if (kernel && !bg_worker->stopping.load(std::memory_order_acquire)) {
                    auto ctx = std::make_shared<FusedOpContext>();
                    ctx->nvrtc_kernel = std::move(kernel);
                    ctx->entry = &nvrtc_fused_entry;
                    jit_fused_ops.set(job.prev_jit_key, ctx);
                    jit_key_mapper.set(job.prev_jit_key, job.prev_jit_key);
                    g_bg_promoted.fetch_add(1, std::memory_order_relaxed);
                }
            } else {
                jit_op_entry_t entry = jit_compiler::compile(
                    job.new_jit_key, job.src, job.is_cuda, job.extra_flags);
                if (entry && !bg_worker->stopping.load(std::memory_order_acquire)) {
                    if (job.is_fused) {
                        auto ctx = std::make_shared<FusedOpContext>();
                        ctx->entry = entry;
                        jit_fused_ops.set(job.prev_jit_key, ctx);
                        jit_fused_ops.set(job.new_jit_key, ctx);
                        jit_key_mapper.set(job.prev_jit_key, job.new_jit_key);
                    } else {
                        jit_ops.set(job.prev_jit_key, entry);
                        jit_key_mapper.set(job.prev_jit_key, job.new_jit_key);
                    }
                    g_bg_promoted.fetch_add(1, std::memory_order_relaxed);
                }
            }
        } catch (const std::exception& e) {
            LOGv << "Tier-1 background compilation skipped:" << e.what();
        } catch (...) {
            LOGv << "Tier-1 background compilation skipped (unknown error)";
        }
        {
            std::lock_guard<std::mutex> lock(bg_worker->mu);
            bg_worker->active = false;
        }
        bg_worker->idle_cv.notify_all();
    }
}

static void bg_stop_worker() {
    std::thread* t = nullptr;
    {
        std::lock_guard<std::mutex> lock(bg_worker->mu);
        if (!bg_worker->thread) return;
        bg_worker->stopping.store(true, std::memory_order_release);
        bg_worker->queue.clear();
        t = bg_worker->thread;
        bg_worker->thread = nullptr;
    }
    bg_worker->cv.notify_all();
    // Release Python GIL while joining to prevent any exit-time GIL inversion.
    PyThreadState* py_state = (Py_IsInitialized() && PyGILState_Check())
        ? PyEval_SaveThread() : nullptr;
    if (t->joinable()) t->join();
    if (py_state) PyEval_RestoreThread(py_state);
    delete t;
    {
        std::lock_guard<std::mutex> lock(bg_worker->mu);
        bg_worker->stopping.store(false, std::memory_order_release);
        bg_worker->active = false;
    }
}

#ifdef __linux__
static void bg_before_fork() {
    bg_stop_worker();
}
static void bg_in_child() {
    bg_worker = new Tier1BgWorker();
}
#endif

static void bg_ensure_started_locked() {
    if (bg_worker->thread) return;
    static bool hooks_registered = false;
    if (!hooks_registered) {
        hooks_registered = true;
        register_cleanup_callback(&bg_stop_worker);
#ifdef __linux__
        pthread_atfork(&bg_before_fork, nullptr, &bg_in_child);
#endif
    }
    bg_worker->stopping.store(false, std::memory_order_release);
    bg_worker->thread = new std::thread(bg_worker_main);
}

// Build a 100% self-contained BgCompileJob synchronously on the calling thread
// (while op/fop and its Vars are guaranteed live) with ZERO Op*/Var* pointers.
static bool build_tier1_bg_job(Op* op, bool is_fused, const std::string& jit_key, BgCompileJob& job) {
    if (is_fused) {
        auto* fop = static_cast<FusedOp*>(op);
        // Fast path: NVRTC snapshot for eligible CUDA fused elementwise kernels (< 0.05 ms!)
        if (use_nvrtc && nvrtc_build_snapshot(fop, jit_key, job.nvrtc_snap)) {
            job.is_fused = true;
            job.is_nvrtc = true;
            job.is_cuda = true;
            job.prev_jit_key = jit_key;
            job.new_jit_key = jit_key;
            return true;
        }
        // Standard Tier-1 C++/CUDA codegen snapshot (pure strings, no raw pointers)
        FusedOpContext* saved_ctx = fop->context;
        shared_ptr<FusedOpContext> saved_owner = fop->context_owner;
        loop_options_t* saved_opts = fop->loop_options;
        loop_options_t saved_tuned = fop->loop_options_tuned;
        struct RestoreGuard {
            FusedOp* f;
            FusedOpContext* ctx;
            shared_ptr<FusedOpContext> owner;
            loop_options_t* opts;
            loop_options_t tuned;
            ~RestoreGuard() {
                f->context = ctx;
                f->context_owner = std::move(owner);
                f->loop_options = opts;
                f->loop_options_tuned = std::move(tuned);
            }
        } guard{fop, saved_ctx, std::move(saved_owner), saved_opts, std::move(saved_tuned)};

        try {
            auto temp_ctx = std::make_shared<FusedOpContext>();
            temp_ctx->setup(fop);
            fop->context = temp_ctx.get();
            fop->context_owner = temp_ctx;
            fop->do_prepare(get_jk());

            OpCompiler oc(fop);
            for (auto op_type : get_op_types()) op_type->post_pass(&oc);
            TunerManager tm(&oc);
            std::string src = tm.tune();
            // Relayed fusions keep runtime relay descriptors in FusedOpContext; skip bg snapshot.
            if (!temp_ctx->vrm.relay_groups.empty()) return false;
            fop->optimize_generated_source(src);

            std::string extra_flags = fop->codegen().extra_flags;
            std::unordered_set<std::string> seen;
            for (auto* member : fop->ops) {
                const auto& flags = member->codegen().extra_flags;
                if (!flags.empty() && seen.insert(flags).second)
                    extra_flags += " " + flags;
            }
            if (fop->loop_options) {
                for (const auto& kv : *fop->loop_options) {
                    if (kv.second && startswith(kv.first, "FLAGS:"))
                        extra_flags += " " + kv.first.substr(6) + " ";
                }
            }

            std::string new_key = fop->get_jit_key(get_jk());
            job.is_fused = true;
            job.is_nvrtc = false;
            job.is_cuda = fop->executes_on_accelerator();
            job.prev_jit_key = jit_key;
            job.new_jit_key = std::move(new_key);
            job.src = std::move(src);
            job.extra_flags = std::move(extra_flags);
            return true;
        } catch (const std::exception& e) {
            LOGv << "Tier-1 background snapshot skipped:" << e.what();
            return false;
        }
    } else {
        try {
            auto compile_fn = op->implementation().kernel.compile;
            if (compile_fn && compile_fn != compile_registered_source) return false;
            op->do_prepare(get_jk());
            OpCompiler oc(op);
            for (auto op_type : get_op_types()) op_type->post_pass(&oc);
            std::string src = std::move(oc.src);
            op->optimize_generated_source(src);
            std::string extra_flags = op->codegen().extra_flags;
            for (Var* v : op->outputs()) {
                if (v->loop_options) {
                    for (const auto& kv : v->loop_options.data()) {
                        if (kv.second && startswith(kv.first, "FLAGS:"))
                            extra_flags += " " + kv.first.substr(6) + " ";
                    }
                }
            }
            job.is_fused = false;
            job.is_nvrtc = false;
            job.is_cuda = op->executes_on_accelerator();
            job.prev_jit_key = jit_key;
            job.new_jit_key = jit_key;
            job.src = std::move(src);
            job.extra_flags = std::move(extra_flags);
            return true;
        } catch (const std::exception& e) {
            LOGv << "Tier-1 background snapshot skipped:" << e.what();
            return false;
        }
    }
}

static void maybe_schedule_tier1_promotion(Op* op, bool is_fused, const std::string& jit_key) {
    if (jit_tier_mode() != 2) return;
    int threshold = std::max(1, jit_tier1_threshold);
    bool should_build = false;
    {
        std::lock_guard<std::mutex> lock(bg_worker->mu);
        if (bg_worker->stopping.load(std::memory_order_acquire)) return;
        int hits = ++bg_worker->hit_counts[jit_key];
        if (hits < threshold) return;
        if (bg_worker->queue.size() >= 256) return;
        if (!bg_worker->enqueued.insert(jit_key).second) return;
        should_build = true;
    }
    if (!should_build) return;

    BgCompileJob job;
    if (!build_tier1_bg_job(op, is_fused, jit_key, job)) {
        std::lock_guard<std::mutex> lock(bg_worker->mu);
        bg_worker->enqueued.erase(jit_key);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(bg_worker->mu);
        if (bg_worker->stopping.load(std::memory_order_acquire)) return;
        bg_ensure_started_locked();
        bg_worker->queue.push_back(std::move(job));
        g_bg_enqueued.fetch_add(1, std::memory_order_relaxed);
    }
    bg_worker->cv.notify_one();
}

} // namespace

bool tier0_can_execute(Op* op, bool is_fused) {
    if (!op) return false;
    if (keep_graph || save_mem || profiler_enable || step_trace_active()
            || jit_search_kernel || graph_capture_recording)
        return false;
    BackendId backend = op->execution_backend();
    if (!backend_tier0_allowed(backend)) return false;

    if (!is_fused) {
        return subop_supported(op, backend);
    }
    auto* fop = static_cast<FusedOp*>(op);
    if (fop->ops.empty()) return false;
    for (Op* sub : fop->ops) {
        if (!subop_supported(sub, backend)) return false;
    }
    // Cap total intermediate scratch memory so very large fused graphs fall back
    // to fused Tier-1 / NVRTC kernels instead of materializing huge temporaries.
    constexpr size_t kMaxTier0TempBytes = 256ULL << 20;
    size_t temp_bytes = 0;
    for (const auto& vi : fop->vars) {
        if (vi.type == 1) {
            temp_bytes += (size_t)vi.var->storage_span_bytes();
            if (temp_bytes > kMaxTier0TempBytes) return false;
        }
    }
    return true;
}

bool tier0_should_skip_parallel_compile(Op* op, bool is_fused) {
    if (jit_tier_mode() == 1) return false;
    return tier0_can_execute(op, is_fused);
}

bool tier0_try_execute_op(Op* op, JK& jk) {
    int mode = jit_tier_mode();
    if (mode == 1) return false;
    if (!tier0_can_execute(op, false)) return false;

    std::string jit_key;
    if (mode == 2) jit_key = jk.to_string();

    execute_tier0_subop(op, op->execution_backend());
    g_tier0_op_execs.fetch_add(1, std::memory_order_relaxed);

    if (mode == 2) maybe_schedule_tier1_promotion(op, false, jit_key);
    return true;
}

bool tier0_try_execute_fused(FusedOp* fop, JK& jk) {
    int mode = jit_tier_mode();
    if (mode == 1) return false;
    if (!tier0_can_execute(fop, true)) return false;

    std::string jit_key;
    if (mode == 2) jit_key = jk.to_string();

    BackendId backend = fop->execution_backend();
    const bool is_cuda = (backend == BackendId::Cuda);

    // Collect external input addresses and allocation identities to guard against
    // reuse_dying_inputs aliasing (including strided/view overlap and reductions).
    static const OpId id_reduce = get_op_id("reduce");
    bool check_alias = fop->ops.size() > 1;
    if (!check_alias) {
        if (fop->ops[0]->type_id() == id_reduce) check_alias = true;
        else {
            for (const auto& vi : fop->vars) {
                if (vi.type == 0 && !vi.var->is_contiguous()) { check_alias = true; break; }
            }
        }
    }
    std::unordered_set<void*> input_ptrs;
    std::vector<std::pair<Allocator*, size_t>> input_allocs;
    if (check_alias) {
        for (const auto& vi : fop->vars) {
            if (vi.type == 0) {
                if (vi.var->mem_ptr) input_ptrs.insert(vi.var->mem_ptr);
                if (vi.var->allocator && vi.var->allocation)
                    input_allocs.emplace_back(vi.var->allocator, vi.var->allocation);
            }
        }
    }
    auto aliases_input = [&](Var* v) -> bool {
        if (!check_alias || !v->mem_ptr || v->size <= 0) return false;
        if (input_ptrs.count(v->mem_ptr)) return true;
        if (v->allocator && v->allocation) {
            for (const auto& ia : input_allocs) {
                if (ia.first == v->allocator && ia.second == v->allocation) return true;
            }
        }
        return false;
    };

    struct DeferredCopy {
        void* dst;
        void* src;
        size_t bytes;
    };
    std::vector<TempVarGuard> guards;
    std::vector<DeferredCopy> deferred_copies;
    guards.reserve(fop->vars.size());

    for (const auto& vi : fop->vars) {
        Var* v = vi.var;
        if (vi.type == 1) {
            // Fused intermediate: allocate temporary buffer, automatically freed on return.
            size_t bytes = (size_t)v->storage_span_bytes();
            guards.emplace_back(v, bytes);
        } else if (vi.type == 2 && aliases_input(v)) {
            // Output aliased a dying input: compute into temp buffer, copy at end.
            void* real_dst = v->mem_ptr;
            size_t bytes = (size_t)v->storage_span_bytes();
            guards.emplace_back(v, bytes);
            deferred_copies.push_back({real_dst, v->mem_ptr, bytes});
        }
    }

    for (Op* sub : fop->ops) {
        execute_tier0_subop(sub, backend);
    }

    if (!deferred_copies.empty()) {
        int dev = is_cuda ? current_device() : 0;
        Device device{backend, dev >= 0 ? dev : 0};
        for (const auto& c : deferred_copies) {
            if (!is_cuda) {
                std::memcpy(c.dst, c.src, c.bytes);
            } else {
                auto stream = backend_stream(device, BackendStreamKind::Compute);
                backend_copy_async(c.dst, device, c.src, device, c.bytes, stream);
            }
        }
    }

    // Release all temporary intermediate guards BEFORE snapshot/return so
    // Var::mem_ptr on intermediates is guaranteed restored to nullptr!
    guards.clear();
    g_tier0_fused_execs.fetch_add(1, std::memory_order_relaxed);

    if (mode == 2) maybe_schedule_tier1_promotion(fop, true, jit_key);
    return true;
}

std::vector<int64> jit_tier_stats() {
    int64 pending = 0;
    {
        std::lock_guard<std::mutex> lock(bg_worker->mu);
        pending = (int64)bg_worker->queue.size() + (bg_worker->active ? 1 : 0);
    }
    return {
        (int64)jit_tier_mode(),
        g_tier0_op_execs.load(std::memory_order_relaxed),
        g_tier0_fused_execs.load(std::memory_order_relaxed),
        g_bg_enqueued.load(std::memory_order_relaxed),
        g_bg_promoted.load(std::memory_order_relaxed),
        pending,
    };
}

void jit_tier_wait_bg() {
    PyThreadState* py_state = (Py_IsInitialized() && PyGILState_Check())
        ? PyEval_SaveThread() : nullptr;
    {
        std::unique_lock<std::mutex> lock(bg_worker->mu);
        bg_worker->idle_cv.wait(lock, []() {
            return (bg_worker->queue.empty() && !bg_worker->active) ||
                   bg_worker->stopping.load(std::memory_order_acquire);
        });
    }
    if (py_state) PyEval_RestoreThread(py_state);
}

} // namespace jittor
