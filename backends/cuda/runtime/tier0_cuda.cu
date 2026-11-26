// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
// Precompiled Tier-0 CUDA meta-operator kernels (strict IEEE math).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
#include <math_constants.h>
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include "core/common.h"
#include "helper_cuda.h"
#include "runtime/tier0_ops.h"

namespace jittor {
namespace {

__device__ __forceinline__ int32_t t0_offset(int32_t idx, int32_t rank,
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

__device__ __forceinline__ float t0_load_f32(const void* p, int32_t idx, T0DType dt) {
    switch (dt) {
        case T0DType::F32:  return static_cast<const float*>(p)[idx];
        case T0DType::F64:  return static_cast<float>(static_cast<const double*>(p)[idx]);
        case T0DType::F16:  return __half2float(static_cast<const __half*>(p)[idx]);
        case T0DType::BF16: return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[idx]);
        case T0DType::I32:  return static_cast<float>(static_cast<const int32_t*>(p)[idx]);
        case T0DType::I64:  return static_cast<float>(static_cast<const int64_t*>(p)[idx]);
        case T0DType::I8:   return static_cast<float>(static_cast<const int8_t*>(p)[idx]);
        case T0DType::U8:   return static_cast<float>(static_cast<const uint8_t*>(p)[idx]);
        case T0DType::BOOL: return static_cast<const uint8_t*>(p)[idx] ? 1.0f : 0.0f;
        default: return 0.0f;
    }
}

__device__ __forceinline__ double t0_load_f64(const void* p, int32_t idx, T0DType dt) {
    switch (dt) {
        case T0DType::F64:  return static_cast<const double*>(p)[idx];
        case T0DType::F32:  return static_cast<double>(static_cast<const float*>(p)[idx]);
        case T0DType::F16:  return static_cast<double>(__half2float(static_cast<const __half*>(p)[idx]));
        case T0DType::BF16: return static_cast<double>(__bfloat162float(static_cast<const __nv_bfloat16*>(p)[idx]));
        case T0DType::I32:  return static_cast<double>(static_cast<const int32_t*>(p)[idx]);
        case T0DType::I64:  return static_cast<double>(static_cast<const int64_t*>(p)[idx]);
        case T0DType::I8:   return static_cast<double>(static_cast<const int8_t*>(p)[idx]);
        case T0DType::U8:   return static_cast<double>(static_cast<const uint8_t*>(p)[idx]);
        case T0DType::BOOL: return static_cast<const uint8_t*>(p)[idx] ? 1.0 : 0.0;
        default: return 0.0;
    }
}

__device__ __forceinline__ int64_t t0_load_i64(const void* p, int32_t idx, T0DType dt) {
    switch (dt) {
        case T0DType::I64:  return static_cast<const int64_t*>(p)[idx];
        case T0DType::I32:  return static_cast<int64_t>(static_cast<const int32_t*>(p)[idx]);
        case T0DType::I8:   return static_cast<int64_t>(static_cast<const int8_t*>(p)[idx]);
        case T0DType::U8:   return static_cast<int64_t>(static_cast<const uint8_t*>(p)[idx]);
        case T0DType::BOOL: return static_cast<const uint8_t*>(p)[idx] ? 1LL : 0LL;
        case T0DType::F32:  return static_cast<int64_t>(static_cast<const float*>(p)[idx]);
        case T0DType::F64:  return static_cast<int64_t>(static_cast<const double*>(p)[idx]);
        case T0DType::F16:  return static_cast<int64_t>(__half2float(static_cast<const __half*>(p)[idx]));
        case T0DType::BF16: return static_cast<int64_t>(__bfloat162float(static_cast<const __nv_bfloat16*>(p)[idx]));
        default: return 0LL;
    }
}

__device__ __forceinline__ bool t0_load_bool(const void* p, int32_t idx, T0DType dt) {
    switch (dt) {
        case T0DType::BOOL: return static_cast<const uint8_t*>(p)[idx] != 0;
        case T0DType::I8:   return static_cast<const int8_t*>(p)[idx] != 0;
        case T0DType::U8:   return static_cast<const uint8_t*>(p)[idx] != 0;
        case T0DType::I32:  return static_cast<const int32_t*>(p)[idx] != 0;
        case T0DType::I64:  return static_cast<const int64_t*>(p)[idx] != 0;
        case T0DType::F32:  return static_cast<const float*>(p)[idx] != 0.0f;
        case T0DType::F64:  return static_cast<const double*>(p)[idx] != 0.0;
        case T0DType::F16:  return __half2float(static_cast<const __half*>(p)[idx]) != 0.0f;
        case T0DType::BF16: return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[idx]) != 0.0f;
        default: return false;
    }
}

__device__ __forceinline__ void t0_store_f32(void* p, int32_t idx, T0DType dt, float v) {
    switch (dt) {
        case T0DType::F32:  static_cast<float*>(p)[idx] = v; break;
        case T0DType::F64:  static_cast<double*>(p)[idx] = static_cast<double>(v); break;
        case T0DType::F16:  static_cast<__half*>(p)[idx] = __float2half_rn(v); break;
        case T0DType::BF16: static_cast<__nv_bfloat16*>(p)[idx] = __float2bfloat16_rn(v); break;
        case T0DType::I32:  static_cast<int32_t*>(p)[idx] = static_cast<int32_t>(v); break;
        case T0DType::I64:  static_cast<int64_t*>(p)[idx] = static_cast<int64_t>(v); break;
        case T0DType::I8:   static_cast<int8_t*>(p)[idx] = static_cast<int8_t>(v); break;
        case T0DType::U8:   static_cast<uint8_t*>(p)[idx] = static_cast<uint8_t>(v); break;
        case T0DType::BOOL: static_cast<uint8_t*>(p)[idx] = (v != 0.0f) ? 1 : 0; break;
        default: break;
    }
}

__device__ __forceinline__ void t0_store_f64(void* p, int32_t idx, T0DType dt, double v) {
    switch (dt) {
        case T0DType::F64:  static_cast<double*>(p)[idx] = v; break;
        case T0DType::F32:  static_cast<float*>(p)[idx] = static_cast<float>(v); break;
        case T0DType::F16:  static_cast<__half*>(p)[idx] = __float2half_rn(static_cast<float>(v)); break;
        case T0DType::BF16: static_cast<__nv_bfloat16*>(p)[idx] = __float2bfloat16_rn(static_cast<float>(v)); break;
        case T0DType::I32:  static_cast<int32_t*>(p)[idx] = static_cast<int32_t>(v); break;
        case T0DType::I64:  static_cast<int64_t*>(p)[idx] = static_cast<int64_t>(v); break;
        case T0DType::I8:   static_cast<int8_t*>(p)[idx] = static_cast<int8_t>(v); break;
        case T0DType::U8:   static_cast<uint8_t*>(p)[idx] = static_cast<uint8_t>(v); break;
        case T0DType::BOOL: static_cast<uint8_t*>(p)[idx] = (v != 0.0) ? 1 : 0; break;
        default: break;
    }
}

__device__ __forceinline__ void t0_store_i64(void* p, int32_t idx, T0DType dt, int64_t v) {
    switch (dt) {
        case T0DType::I64:  static_cast<int64_t*>(p)[idx] = v; break;
        case T0DType::I32:  static_cast<int32_t*>(p)[idx] = static_cast<int32_t>(v); break;
        case T0DType::I8:   static_cast<int8_t*>(p)[idx] = static_cast<int8_t>(v); break;
        case T0DType::U8:   static_cast<uint8_t*>(p)[idx] = static_cast<uint8_t>(v); break;
        case T0DType::BOOL: static_cast<uint8_t*>(p)[idx] = (v != 0) ? 1 : 0; break;
        case T0DType::F32:  static_cast<float*>(p)[idx] = static_cast<float>(v); break;
        case T0DType::F64:  static_cast<double*>(p)[idx] = static_cast<double>(v); break;
        case T0DType::F16:  static_cast<__half*>(p)[idx] = __float2half_rn(static_cast<float>(v)); break;
        case T0DType::BF16: static_cast<__nv_bfloat16*>(p)[idx] = __float2bfloat16_rn(static_cast<float>(v)); break;
        default: break;
    }
}

__device__ __forceinline__ float t0_max_f32(float a, float b) { return ((a > b) | (a != a)) ? a : b; }
__device__ __forceinline__ float t0_min_f32(float a, float b) { return ((a < b) | (a != a)) ? a : b; }
__device__ __forceinline__ double t0_max_f64(double a, double b) { return ((a > b) | (a != a)) ? a : b; }
__device__ __forceinline__ double t0_min_f64(double a, double b) { return ((a < b) | (a != a)) ? a : b; }

__device__ __forceinline__ float t0_signed_pow_f32(float x, float y) {
    if (x < 0.0f && ::floorf(y) == y)
        return ::powf(-x, y) * (::fmodf(y, 2.0f) != 0.0f ? -1.0f : 1.0f);
    return ::powf(x, y);
}
__device__ __forceinline__ double t0_signed_pow_f64(double x, double y) {
    if (x < 0.0 && ::floor(y) == y)
        return ::pow(-x, y) * (::fmod(y, 2.0) != 0.0 ? -1.0 : 1.0);
    return ::pow(x, y);
}

__device__ __forceinline__ float t0_floor_div_f32(float x, float y) {
    if (y == 0.0f) return x / y;
    float mod = ::fmodf(x, y);
    float div = (x - mod) / y;
    if (mod != 0.0f && ((y < 0.0f) != (mod < 0.0f))) div -= 1.0f;
    if (div == 0.0f) return ::copysignf(0.0f, x / y);
    float fd = ::floorf(div);
    if (div - fd > 0.5f) fd += 1.0f;
    return fd;
}
__device__ __forceinline__ double t0_floor_div_f64(double x, double y) {
    if (y == 0.0) return x / y;
    double mod = ::fmod(x, y);
    double div = (x - mod) / y;
    if (mod != 0.0 && ((y < 0.0) != (mod < 0.0))) div -= 1.0;
    if (div == 0.0) return ::copysign(0.0, x / y);
    double fd = ::floor(div);
    if (div - fd > 0.5) fd += 1.0;
    return fd;
}
__device__ __forceinline__ float t0_floor_mod_f32(float x, float y) {
    float mod = ::fmodf(x, y);
    if (y == 0.0f) return mod;
    if (mod != 0.0f) { if ((y < 0.0f) != (mod < 0.0f)) mod += y; }
    else mod = ::copysignf(0.0f, y);
    return mod;
}
__device__ __forceinline__ double t0_floor_mod_f64(double x, double y) {
    double mod = ::fmod(x, y);
    if (y == 0.0) return mod;
    if (mod != 0.0) { if ((y < 0.0) != (mod < 0.0)) mod += y; }
    else mod = ::copysign(0.0, y);
    return mod;
}

template <class T>
__device__ __forceinline__ T t0_floor_div_signed(T x, T y) {
    if (y == 0) return 0;
    if (y == T(-1)) {
        typedef typename std::make_unsigned<T>::type U;
        return T(U(0) - U(x));
    }
    T q = x / y, r = x % y;
    return q - T(r != 0 && ((r < 0) != (y < 0)));
}

template <class T>
__device__ __forceinline__ T t0_floor_mod_signed(T x, T y) {
    if (y == 0 || y == T(-1)) return 0;
    T r = x % y;
    return r + T(r != 0 && ((r < 0) != (y < 0))) * y;
}

__device__ __forceinline__ bool t0_is_float(T0DType dt) {
    return dt == T0DType::F32 || dt == T0DType::F64 || dt == T0DType::F16 || dt == T0DType::BF16;
}

__device__ __forceinline__ float t0_apply_unary_f32(T0UnaryOp op, float x, T0DType ty) {
    switch (op) {
        case T0UnaryOp::Cast:
        case T0UnaryOp::Conj:     return x;
        case T0UnaryOp::Abs:      return ::fabsf(x);
        case T0UnaryOp::Negative: return -x;
        case T0UnaryOp::Relu:     return x > 0.0f ? x : 0.0f;
        case T0UnaryOp::Exp:      return ::expf(x);
        case T0UnaryOp::Log:      return ::logf(x);
        case T0UnaryOp::Log1p:    return ::log1pf(x);
        case T0UnaryOp::Sqrt:     return ::sqrtf(x);
        case T0UnaryOp::Round:    return ::rintf(x);
        case T0UnaryOp::Floor:    return ::floorf(x);
        case T0UnaryOp::Ceil:     return ::ceilf(x);
        case T0UnaryOp::RoundInt: return ::roundf(x);
        case T0UnaryOp::FloorInt: return ::floorf(x);
        case T0UnaryOp::CeilInt:  return ::ceilf(x);
        case T0UnaryOp::Sin:      return ::sinf(x);
        case T0UnaryOp::Asin:     return ::asinf(x);
        case T0UnaryOp::Sinh:     return ::sinhf(x);
        case T0UnaryOp::Asinh:    return ::asinhf(x);
        case T0UnaryOp::Cos:      return ::cosf(x);
        case T0UnaryOp::Acos:     return ::acosf(x);
        case T0UnaryOp::Cosh:     return ::coshf(x);
        case T0UnaryOp::Acosh:    return ::acoshf(x);
        case T0UnaryOp::Tan:      return ::tanf(x);
        case T0UnaryOp::Atan:     return ::atanf(x);
        case T0UnaryOp::Tanh:     return ::tanhf(x);
        case T0UnaryOp::Atanh:    return ::atanhf(x);
        case T0UnaryOp::Sigmoid: {
            if (ty == T0DType::F16) return 1.0f / (1.0f + ::expf(::fminf(-x, 30.0f)));
            if (ty == T0DType::BF16) return 1.0f / (1.0f + ::expf(::fminf(-x, 300.0f)));
            return 1.0f / (1.0f + ::expf(-x));
        }
        case T0UnaryOp::Erf:      return ::erff(x);
        case T0UnaryOp::Erfinv:   return ::erfinvf(x);
        default: return x;
    }
}

__device__ __forceinline__ double t0_apply_unary_f64(T0UnaryOp op, double x) {
    switch (op) {
        case T0UnaryOp::Cast:
        case T0UnaryOp::Conj:     return x;
        case T0UnaryOp::Abs:      return ::fabs(x);
        case T0UnaryOp::Negative: return -x;
        case T0UnaryOp::Relu:     return x > 0.0 ? x : 0.0;
        case T0UnaryOp::Exp:      return ::exp(x);
        case T0UnaryOp::Log:      return ::log(x);
        case T0UnaryOp::Log1p:    return ::log1p(x);
        case T0UnaryOp::Sqrt:     return ::sqrt(x);
        case T0UnaryOp::Round:    return ::rint(x);
        case T0UnaryOp::Floor:    return ::floor(x);
        case T0UnaryOp::Ceil:     return ::ceil(x);
        case T0UnaryOp::RoundInt: return ::round(x);
        case T0UnaryOp::FloorInt: return ::floor(x);
        case T0UnaryOp::CeilInt:  return ::ceil(x);
        case T0UnaryOp::Sin:      return ::sin(x);
        case T0UnaryOp::Asin:     return ::asin(x);
        case T0UnaryOp::Sinh:     return ::sinh(x);
        case T0UnaryOp::Asinh:    return ::asinh(x);
        case T0UnaryOp::Cos:      return ::cos(x);
        case T0UnaryOp::Acos:     return ::acos(x);
        case T0UnaryOp::Cosh:     return ::cosh(x);
        case T0UnaryOp::Acosh:    return ::acosh(x);
        case T0UnaryOp::Tan:      return ::tan(x);
        case T0UnaryOp::Atan:     return ::atan(x);
        case T0UnaryOp::Tanh:     return ::tanh(x);
        case T0UnaryOp::Atanh:    return ::atanh(x);
        case T0UnaryOp::Sigmoid:  return 1.0 / (1.0 + ::exp(-x));
        case T0UnaryOp::Erf:      return ::erf(x);
        case T0UnaryOp::Erfinv:   return ::erfinv(x);
        default: return x;
    }
}

__global__ void k_fill_scalar(void* dst, int dsize, uint64_t raw_bytes, int32_t count) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;
    for (int32_t i = tid; i < count; i += stride) {
        switch (dsize) {
            case 1: static_cast<uint8_t*>(dst)[i] = static_cast<uint8_t>(raw_bytes); break;
            case 2: static_cast<uint16_t*>(dst)[i] = static_cast<uint16_t>(raw_bytes); break;
            case 4: static_cast<uint32_t*>(dst)[i] = static_cast<uint32_t>(raw_bytes); break;
            case 8: static_cast<uint64_t*>(dst)[i] = raw_bytes; break;
        }
    }
}

__global__ void k_unary_f32_contig(T0UnaryOp op, const float* x, float* y, int32_t num) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;
    for (int32_t i = tid; i < num; i += stride) {
        y[i] = t0_apply_unary_f32(op, x[i], T0DType::F32);
    }
}

__global__ void k_unary_generic(T0UnaryOp op, T0DType tx, T0DType ty,
                                const void* x, void* y, T0ShapeMeta meta) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;
    for (int32_t i = tid; i < meta.num; i += stride) {
        int32_t xi = meta.xstrided ? t0_offset(i, meta.rank, meta.shape, meta.above, meta.xstride) : i;
        if (op == T0UnaryOp::LogicalNot) {
            static_cast<uint8_t*>(y)[i] = t0_load_bool(x, xi, tx) ? 0 : 1;
        } else if (op == T0UnaryOp::BitwiseNot) {
            if (ty == T0DType::BOOL) {
                static_cast<uint8_t*>(y)[i] = t0_load_bool(x, xi, tx) ? 0 : 1;
            } else {
                t0_store_i64(y, i, ty, ~t0_load_i64(x, xi, tx));
            }
        } else if (op == T0UnaryOp::Cast && ty == T0DType::BOOL) {
            static_cast<uint8_t*>(y)[i] = t0_load_bool(x, xi, tx) ? 1 : 0;
        } else if (ty == T0DType::F64 || (op == T0UnaryOp::Cast && tx == T0DType::F64)) {
            t0_store_f64(y, i, ty, t0_apply_unary_f64(op, t0_load_f64(x, xi, tx)));
        } else if (t0_is_float(ty) || op == T0UnaryOp::RoundInt || op == T0UnaryOp::FloorInt || op == T0UnaryOp::CeilInt) {
            t0_store_f32(y, i, ty, t0_apply_unary_f32(op, t0_load_f32(x, xi, tx), ty));
        } else {
            int64_t v = t0_load_i64(x, xi, tx);
            if (op == T0UnaryOp::Abs) v = (v < 0) ? (int64_t)(0ULL - (uint64_t)v) : v;
            else if (op == T0UnaryOp::Negative) v = (int64_t)(0ULL - (uint64_t)v);
            else if (op == T0UnaryOp::Relu) v = (v > 0) ? v : 0;
            t0_store_i64(y, i, ty, v);
        }
    }
}

__global__ void k_binary_f32_contig(T0BinaryOp op, const float* x, const float* y, float* z, int32_t num) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;
    for (int32_t i = tid; i < num; i += stride) {
        float a = x[i], b = y[i], r = 0.0f;
        switch (op) {
            case T0BinaryOp::Add:         r = a + b; break;
            case T0BinaryOp::Subtract:    r = a - b; break;
            case T0BinaryOp::Multiply:    r = a * b; break;
            case T0BinaryOp::Divide:      r = a / b; break;
            case T0BinaryOp::FloorDivide: r = t0_floor_div_f32(a, b); break;
            case T0BinaryOp::Mod:         r = t0_floor_mod_f32(a, b); break;
            case T0BinaryOp::Fmod:        r = ::fmodf(a, b); break;
            case T0BinaryOp::Pow:         r = static_cast<float>(t0_signed_pow_f64((double)a, (double)b)); break;
            case T0BinaryOp::Maximum:     r = t0_max_f32(a, b); break;
            case T0BinaryOp::Minimum:     r = t0_min_f32(a, b); break;
            default: break;
        }
        z[i] = r;
    }
}

__global__ void k_binary_generic(T0BinaryOp op, T0DType tx, T0DType ty, T0DType tz,
                                 const void* x, const void* y, void* z, T0ShapeMeta meta) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;
    for (int32_t i = tid; i < meta.num; i += stride) {
        int32_t xi = meta.xstrided ? t0_offset(i, meta.rank, meta.shape, meta.above, meta.xstride) : i;
        int32_t yi = meta.ystrided ? t0_offset(i, meta.rank, meta.shape, meta.above, meta.ystride) : i;

        if (op == T0BinaryOp::LogicalAnd || op == T0BinaryOp::LogicalOr || op == T0BinaryOp::LogicalXor) {
            bool a = t0_load_bool(x, xi, tx);
            bool b = t0_load_bool(y, yi, ty);
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
                double a = t0_load_f64(x, xi, tx), b = t0_load_f64(y, yi, ty);
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
                float a = t0_load_f32(x, xi, tx), b = t0_load_f32(y, yi, ty);
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
                int64_t a = t0_load_i64(x, xi, tx), b = t0_load_i64(y, yi, ty);
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
            double a = t0_load_f64(x, xi, tx), b = t0_load_f64(y, yi, ty), r = 0.0;
            switch (op) {
                case T0BinaryOp::Add:         r = a + b; break;
                case T0BinaryOp::Subtract:    r = a - b; break;
                case T0BinaryOp::Multiply:    r = a * b; break;
                case T0BinaryOp::Divide:      r = a / b; break;
                case T0BinaryOp::FloorDivide: r = t0_floor_div_f64(a, b); break;
                case T0BinaryOp::Mod:         r = t0_floor_mod_f64(a, b); break;
                case T0BinaryOp::Fmod:        r = ::fmod(a, b); break;
                case T0BinaryOp::Pow:         r = t0_signed_pow_f64(a, b); break;
                case T0BinaryOp::Maximum:     r = t0_max_f64(a, b); break;
                case T0BinaryOp::Minimum:     r = t0_min_f64(a, b); break;
                default: break;
            }
            static_cast<double*>(z)[i] = r;
        } else if (t0_is_float(tz)) {
            float a = t0_load_f32(x, xi, tx), b = t0_load_f32(y, yi, ty), r = 0.0f;
            switch (op) {
                case T0BinaryOp::Add:         r = a + b; break;
                case T0BinaryOp::Subtract:    r = a - b; break;
                case T0BinaryOp::Multiply:    r = a * b; break;
                case T0BinaryOp::Divide:      r = a / b; break;
                case T0BinaryOp::FloorDivide: r = t0_floor_div_f32(a, b); break;
                case T0BinaryOp::Mod:         r = t0_floor_mod_f32(a, b); break;
                case T0BinaryOp::Fmod:        r = ::fmodf(a, b); break;
                case T0BinaryOp::Pow:
                    r = (tz == T0DType::F32)
                        ? static_cast<float>(t0_signed_pow_f64((double)a, (double)b))
                        : t0_signed_pow_f32(a, b);
                    break;
                case T0BinaryOp::Maximum:     r = t0_max_f32(a, b); break;
                case T0BinaryOp::Minimum:     r = t0_min_f32(a, b); break;
                default: break;
            }
            t0_store_f32(z, i, tz, r);
        } else if (tz == T0DType::I32) {
            int32_t a = static_cast<int32_t>(t0_load_i64(x, xi, tx));
            int32_t b = static_cast<int32_t>(t0_load_i64(y, yi, ty));
            int32_t r = 0;
            switch (op) {
                case T0BinaryOp::Add:         r = a + b; break;
                case T0BinaryOp::Subtract:    r = a - b; break;
                case T0BinaryOp::Multiply:    r = a * b; break;
                case T0BinaryOp::FloorDivide: r = t0_floor_div_signed<int32_t>(a, b); break;
                case T0BinaryOp::Mod:         r = t0_floor_mod_signed<int32_t>(a, b); break;
                case T0BinaryOp::Fmod:        r = (b == 0 || b == -1) ? 0 : (a % b); break;
                case T0BinaryOp::Pow:         r = static_cast<int32_t>(t0_signed_pow_f64((double)a, (double)b)); break;
                case T0BinaryOp::Maximum:     r = a > b ? a : b; break;
                case T0BinaryOp::Minimum:     r = a < b ? a : b; break;
                case T0BinaryOp::BitwiseAnd:  r = a & b; break;
                case T0BinaryOp::BitwiseOr:   r = a | b; break;
                case T0BinaryOp::BitwiseXor:  r = a ^ b; break;
                case T0BinaryOp::LeftShift:   r = a << b; break;
                case T0BinaryOp::RightShift:  r = a >> b; break;
                default: break;
            }
            static_cast<int32_t*>(z)[i] = r;
        } else if (tz == T0DType::I64) {
            int64_t a = t0_load_i64(x, xi, tx);
            int64_t b = t0_load_i64(y, yi, ty);
            int64_t r = 0;
            switch (op) {
                case T0BinaryOp::Add:         r = a + b; break;
                case T0BinaryOp::Subtract:    r = a - b; break;
                case T0BinaryOp::Multiply:    r = a * b; break;
                case T0BinaryOp::FloorDivide: r = t0_floor_div_signed<int64_t>(a, b); break;
                case T0BinaryOp::Mod:         r = t0_floor_mod_signed<int64_t>(a, b); break;
                case T0BinaryOp::Fmod:        r = (b == 0 || b == -1) ? 0 : (a % b); break;
                case T0BinaryOp::Pow:         r = static_cast<int64_t>(t0_signed_pow_f64((double)a, (double)b)); break;
                case T0BinaryOp::Maximum:     r = a > b ? a : b; break;
                case T0BinaryOp::Minimum:     r = a < b ? a : b; break;
                case T0BinaryOp::BitwiseAnd:  r = a & b; break;
                case T0BinaryOp::BitwiseOr:   r = a | b; break;
                case T0BinaryOp::BitwiseXor:  r = a ^ b; break;
                case T0BinaryOp::LeftShift:   r = a << b; break;
                case T0BinaryOp::RightShift:  r = a >> b; break;
                default: break;
            }
            static_cast<int64_t*>(z)[i] = r;
        } else if (tz == T0DType::BOOL) {
            bool a = t0_load_bool(x, xi, tx);
            bool b = t0_load_bool(y, yi, ty);
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
            int32_t a = static_cast<int32_t>(t0_load_i64(x, xi, tx));
            int32_t b = static_cast<int32_t>(t0_load_i64(y, yi, ty));
            int32_t r = 0;
            switch (op) {
                case T0BinaryOp::Add:         r = a + b; break;
                case T0BinaryOp::Subtract:    r = a - b; break;
                case T0BinaryOp::Multiply:    r = a * b; break;
                case T0BinaryOp::FloorDivide:
                    if (tz == T0DType::I8) r = t0_floor_div_signed<int8_t>((int8_t)a, (int8_t)b);
                    else r = (b == 0) ? 0 : ((uint8_t)a / (uint8_t)b);
                    break;
                case T0BinaryOp::Mod:
                    if (tz == T0DType::I8) r = t0_floor_mod_signed<int8_t>((int8_t)a, (int8_t)b);
                    else r = (b == 0) ? 0 : ((uint8_t)a % (uint8_t)b);
                    break;
                case T0BinaryOp::Fmod:
                    if (tz == T0DType::I8) r = (b == 0 || (int8_t)b == -1) ? 0 : ((int8_t)a % (int8_t)b);
                    else r = (b == 0) ? 0 : ((uint8_t)a % (uint8_t)b);
                    break;
                case T0BinaryOp::Pow:         r = static_cast<int32_t>(t0_signed_pow_f64((double)a, (double)b)); break;
                case T0BinaryOp::Maximum:     r = a > b ? a : b; break;
                case T0BinaryOp::Minimum:     r = a < b ? a : b; break;
                case T0BinaryOp::BitwiseAnd:  r = a & b; break;
                case T0BinaryOp::BitwiseOr:   r = a | b; break;
                case T0BinaryOp::BitwiseXor:  r = a ^ b; break;
                case T0BinaryOp::LeftShift:   r = a << b; break;
                case T0BinaryOp::RightShift:  r = (tz == T0DType::I8) ? ((int8_t)a >> b) : ((uint8_t)a >> b); break;
                default: break;
            }
            t0_store_i64(z, i, tz, r);
        }
    }
}

__global__ void k_ternary_generic(T0DType tc, T0DType tx, T0DType ty, T0DType tz,
                                  const void* cond, const void* x, const void* y, void* z,
                                  T0ShapeMeta meta) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;
    for (int32_t i = tid; i < meta.num; i += stride) {
        int32_t ci = meta.cstrided ? t0_offset(i, meta.rank, meta.shape, meta.above, meta.cstride) : i;
        int32_t xi = meta.xstrided ? t0_offset(i, meta.rank, meta.shape, meta.above, meta.xstride) : i;
        int32_t yi = meta.ystrided ? t0_offset(i, meta.rank, meta.shape, meta.above, meta.ystride) : i;
        bool c = t0_load_bool(cond, ci, tc);
        if (tz == T0DType::F64) {
            static_cast<double*>(z)[i] = c ? t0_load_f64(x, xi, tx) : t0_load_f64(y, yi, ty);
        } else if (t0_is_float(tz)) {
            t0_store_f32(z, i, tz, c ? t0_load_f32(x, xi, tx) : t0_load_f32(y, yi, ty));
        } else if (tz == T0DType::BOOL) {
            static_cast<uint8_t*>(z)[i] = (c ? t0_load_bool(x, xi, tx) : t0_load_bool(y, yi, ty)) ? 1 : 0;
        } else {
            t0_store_i64(z, i, tz, c ? t0_load_i64(x, xi, tx) : t0_load_i64(y, yi, ty));
        }
    }
}

__global__ void k_broadcast_generic(T0DType tx, const void* x, void* z, T0ShapeMeta meta) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;
    for (int32_t i = tid; i < meta.num; i += stride) {
        int32_t xi = meta.xstrided ? t0_offset(i, meta.rank, meta.shape, meta.above, meta.xstride) : 0;
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

__device__ __forceinline__ int32_t t0_red_base_offset(int32_t out_idx, const T0ReduceMeta& m) {
    int32_t off = 0;
    for (int32_t d = 0; d < m.out_rank; ++d) {
        if (m.out_xstride[d] == 0) continue;
        int32_t c;
        if (m.out_rank == 1) c = out_idx;
        else if (d == 0) c = out_idx / m.out_above[0];
        else if (d == m.out_rank - 1) c = out_idx % m.out_shape[d];
        else c = (out_idx / m.out_above[d]) % m.out_shape[d];
        off += c * m.out_xstride[d];
    }
    return off;
}

__device__ __forceinline__ int32_t t0_red_step_offset(int32_t red_idx, const T0ReduceMeta& m) {
    int32_t off = 0;
    for (int32_t d = 0; d < m.red_rank; ++d) {
        if (m.red_xstride[d] == 0) continue;
        int32_t c;
        if (m.red_rank == 1) c = red_idx;
        else if (d == 0) c = red_idx / m.red_above[0];
        else if (d == m.red_rank - 1) c = red_idx % m.red_shape[d];
        else c = (red_idx / m.red_above[d]) % m.red_shape[d];
        off += c * m.red_xstride[d];
    }
    return off;
}

__device__ __forceinline__ double t0_reduce_init_f64(T0ReduceOp op) {
    switch (op) {
        case T0ReduceOp::Add:
        case T0ReduceOp::Mean:     return 0.0;
        case T0ReduceOp::Multiply: return 1.0;
        case T0ReduceOp::Maximum:  return -CUDART_INF;
        case T0ReduceOp::Minimum:  return CUDART_INF;
        default: return 0.0;
    }
}

__device__ __forceinline__ double t0_reduce_combine_f64(T0ReduceOp op, double a, double b) {
    switch (op) {
        case T0ReduceOp::Add:
        case T0ReduceOp::Mean:     return a + b;
        case T0ReduceOp::Multiply: return a * b;
        case T0ReduceOp::Maximum:  return t0_max_f64(a, b);
        case T0ReduceOp::Minimum:  return t0_min_f64(a, b);
        default: return a;
    }
}

__device__ __forceinline__ int64_t t0_reduce_init_i64(T0ReduceOp op, T0DType ty) {
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
            if (ty == T0DType::I32) return -2147483648LL;
            return (-9223372036854775807LL - 1LL);
        case T0ReduceOp::Minimum:
            if (ty == T0DType::I8) return 127LL;
            if (ty == T0DType::U8) return 255LL;
            if (ty == T0DType::BOOL) return 1LL;
            if (ty == T0DType::I32) return 2147483647LL;
            return 9223372036854775807LL;
        default: return 0LL;
    }
}

__device__ __forceinline__ int64_t t0_reduce_combine_i64(T0ReduceOp op, int64_t a, int64_t b) {
    switch (op) {
        case T0ReduceOp::Add:        return (int64_t)((uint64_t)a + (uint64_t)b);
        case T0ReduceOp::Multiply:   return (int64_t)((uint64_t)a * (uint64_t)b);
        case T0ReduceOp::Maximum:    return a > b ? a : b;
        case T0ReduceOp::Minimum:    return a < b ? a : b;
        case T0ReduceOp::LogicalAnd: return (a && b) ? 1LL : 0LL;
        case T0ReduceOp::LogicalOr:  return (a || b) ? 1LL : 0LL;
        case T0ReduceOp::LogicalXor: return ((a != 0) != (b != 0)) ? 1LL : 0LL;
        case T0ReduceOp::BitwiseAnd: return a & b;
        case T0ReduceOp::BitwiseOr:  return a | b;
        case T0ReduceOp::BitwiseXor: return a ^ b;
        default: return a;
    }
}

// Block-parallel reduction: 1 block of 256 threads per output slot (loops over out_idx if > gridDim.x).
__global__ void k_reduce_block(T0ReduceOp op, T0DType tx, T0DType ty,
                               const void* x, void* y, T0ReduceMeta m) {
    const bool is_logical = (op == T0ReduceOp::LogicalAnd || op == T0ReduceOp::LogicalOr || op == T0ReduceOp::LogicalXor);
    const bool use_float = !is_logical && t0_is_float(ty);

    __shared__ double s_f64[256];
    __shared__ int64_t s_i64[256];
    const int tid = threadIdx.x;

    for (int32_t out_idx = blockIdx.x; out_idx < m.out_num; out_idx += gridDim.x) {
        if (m.red_num == 0) {
            if (tid == 0) {
                if (op == T0ReduceOp::Mean) {
                    t0_store_f64(y, out_idx, ty, (double)::nanf(""));
                } else if (use_float) {
                    t0_store_f64(y, out_idx, ty, t0_reduce_init_f64(op));
                } else {
                    t0_store_i64(y, out_idx, ty, t0_reduce_init_i64(op, ty));
                }
            }
            __syncthreads();
            continue;
        }
        const int32_t base = t0_red_base_offset(out_idx, m);
        if (use_float) {
            double acc = t0_reduce_init_f64(op);
            for (int32_t r = tid; r < m.red_num; r += blockDim.x) {
                double v = t0_load_f64(x, base + t0_red_step_offset(r, m), tx);
                acc = t0_reduce_combine_f64(op, acc, v);
                if (op == T0ReduceOp::Multiply && ty == T0DType::F16) {
                    acc = (double)__half2float(__float2half_rn((float)acc));
                } else if (op == T0ReduceOp::Multiply && ty == T0DType::BF16) {
                    acc = (double)__bfloat162float(__float2bfloat16_rn((float)acc));
                }
            }
            s_f64[tid] = acc;
            __syncthreads();
            for (int s = 128; s > 0; s >>= 1) {
                if (tid < s) {
                    double v = t0_reduce_combine_f64(op, s_f64[tid], s_f64[tid + s]);
                    if (op == T0ReduceOp::Multiply && ty == T0DType::F16)
                        v = (double)__half2float(__float2half_rn((float)v));
                    else if (op == T0ReduceOp::Multiply && ty == T0DType::BF16)
                        v = (double)__bfloat162float(__float2bfloat16_rn((float)v));
                    s_f64[tid] = v;
                }
                __syncthreads();
            }
            if (tid == 0) {
                double res = s_f64[0];
                if (op == T0ReduceOp::Mean) res /= (double)m.red_num;
                t0_store_f64(y, out_idx, ty, res);
            }
        } else {
            int64_t acc = t0_reduce_init_i64(op, ty);
            for (int32_t r = tid; r < m.red_num; r += blockDim.x) {
                int32_t off = base + t0_red_step_offset(r, m);
                int64_t v = is_logical ? (t0_load_bool(x, off, tx) ? 1LL : 0LL) : t0_load_i64(x, off, tx);
                acc = t0_reduce_combine_i64(op, acc, v);
            }
            s_i64[tid] = acc;
            __syncthreads();
            for (int s = 128; s > 0; s >>= 1) {
                if (tid < s) s_i64[tid] = t0_reduce_combine_i64(op, s_i64[tid], s_i64[tid + s]);
                __syncthreads();
            }
            if (tid == 0) {
                t0_store_i64(y, out_idx, ty, s_i64[0]);
            }
        }
        __syncthreads();
    }
}

// One-thread-per-output reduction for large out_num with small red_num.
__global__ void k_reduce_thread(T0ReduceOp op, T0DType tx, T0DType ty,
                                const void* x, void* y, T0ReduceMeta m) {
    const bool is_logical = (op == T0ReduceOp::LogicalAnd || op == T0ReduceOp::LogicalOr || op == T0ReduceOp::LogicalXor);
    const bool use_float = !is_logical && t0_is_float(ty);
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;
    for (int32_t out_idx = tid; out_idx < m.out_num; out_idx += stride) {
        if (m.red_num == 0) {
            if (op == T0ReduceOp::Mean) t0_store_f64(y, out_idx, ty, (double)::nanf(""));
            else if (use_float) t0_store_f64(y, out_idx, ty, t0_reduce_init_f64(op));
            else t0_store_i64(y, out_idx, ty, t0_reduce_init_i64(op, ty));
            continue;
        }
        int32_t base = t0_red_base_offset(out_idx, m);
        if (use_float) {
            double acc = t0_reduce_init_f64(op);
            for (int32_t r = 0; r < m.red_num; ++r) {
                double v = t0_load_f64(x, base + t0_red_step_offset(r, m), tx);
                acc = t0_reduce_combine_f64(op, acc, v);
                if (op == T0ReduceOp::Multiply && ty == T0DType::F16)
                    acc = (double)__half2float(__float2half_rn((float)acc));
                else if (op == T0ReduceOp::Multiply && ty == T0DType::BF16)
                    acc = (double)__bfloat162float(__float2bfloat16_rn((float)acc));
            }
            if (op == T0ReduceOp::Mean) acc /= (double)m.red_num;
            t0_store_f64(y, out_idx, ty, acc);
        } else {
            int64_t acc = t0_reduce_init_i64(op, ty);
            for (int32_t r = 0; r < m.red_num; ++r) {
                int32_t off = base + t0_red_step_offset(r, m);
                int64_t v = is_logical ? (t0_load_bool(x, off, tx) ? 1LL : 0LL) : t0_load_i64(x, off, tx);
                acc = t0_reduce_combine_i64(op, acc, v);
            }
            t0_store_i64(y, out_idx, ty, acc);
        }
    }
}

static inline int choose_blocks(int32_t num) {
    int b = (num + 255) / 256;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;
    return b;
}

} // namespace

void cuda_tier0_fill_scalar(void* dst, int dsize, uint64_t raw_bytes, int32_t count) {
    if (count <= 0) return;
    k_fill_scalar<<<choose_blocks(count), 256, 0, cudaStreamPerThread>>>(dst, dsize, raw_bytes, count);
    checkCudaErrors(cudaGetLastError());
}

void cuda_tier0_unary(T0UnaryOp op, T0DType tx, T0DType ty,
                      const void* x, void* y, const T0ShapeMeta& meta) {
    if (meta.num <= 0) return;
    int blocks = choose_blocks(meta.num);
    if (!meta.xstrided && tx == T0DType::F32 && ty == T0DType::F32 &&
        op != T0UnaryOp::LogicalNot && op != T0UnaryOp::BitwiseNot) {
        k_unary_f32_contig<<<blocks, 256, 0, cudaStreamPerThread>>>(
            op, static_cast<const float*>(x), static_cast<float*>(y), meta.num);
    } else {
        k_unary_generic<<<blocks, 256, 0, cudaStreamPerThread>>>(op, tx, ty, x, y, meta);
    }
    checkCudaErrors(cudaGetLastError());
}

void cuda_tier0_binary(T0BinaryOp op, T0DType tx, T0DType ty, T0DType tz,
                       const void* x, const void* y, void* z, const T0ShapeMeta& meta) {
    if (meta.num <= 0) return;
    int blocks = choose_blocks(meta.num);
    if (!meta.xstrided && !meta.ystrided &&
        tx == T0DType::F32 && ty == T0DType::F32 && tz == T0DType::F32 &&
        op >= T0BinaryOp::Add && op <= T0BinaryOp::Minimum) {
        k_binary_f32_contig<<<blocks, 256, 0, cudaStreamPerThread>>>(
            op, static_cast<const float*>(x), static_cast<const float*>(y),
            static_cast<float*>(z), meta.num);
    } else {
        k_binary_generic<<<blocks, 256, 0, cudaStreamPerThread>>>(op, tx, ty, tz, x, y, z, meta);
    }
    checkCudaErrors(cudaGetLastError());
}

void cuda_tier0_ternary(T0DType tc, T0DType tx, T0DType ty, T0DType tz,
                        const void* cond, const void* x, const void* y, void* z,
                        const T0ShapeMeta& meta) {
    if (meta.num <= 0) return;
    k_ternary_generic<<<choose_blocks(meta.num), 256, 0, cudaStreamPerThread>>>(
        tc, tx, ty, tz, cond, x, y, z, meta);
    checkCudaErrors(cudaGetLastError());
}

void cuda_tier0_broadcast(T0DType tx, const void* x, void* z, const T0ShapeMeta& meta) {
    if (meta.num <= 0) return;
    k_broadcast_generic<<<choose_blocks(meta.num), 256, 0, cudaStreamPerThread>>>(tx, x, z, meta);
    checkCudaErrors(cudaGetLastError());
}

void cuda_tier0_reduce(T0ReduceOp op, T0DType tx, T0DType ty,
                       const void* x, void* y, const T0ReduceMeta& meta) {
    if (meta.out_num <= 0) return;
    if (meta.red_num >= 16 || meta.out_num < 1024) {
        int blocks = std::min<int>(meta.out_num, 65535);
        k_reduce_block<<<blocks, 256, 0, cudaStreamPerThread>>>(op, tx, ty, x, y, meta);
    } else {
        k_reduce_thread<<<choose_blocks(meta.out_num), 256, 0, cudaStreamPerThread>>>(op, tx, ty, x, y, meta);
    }
    checkCudaErrors(cudaGetLastError());
}

} // namespace jittor
