// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cmath>
#include <cstring>
#include "core/common.h"

namespace jittor {

namespace fp8_detail {

inline uint32 f32_to_bits(float32 f) {
    uint32 u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

inline float32 bits_to_f32(uint32 u) {
    float32 f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// OCP FP8 E4M3FN: 1 sign, 4 exponent (bias 7), 3 mantissa.
// No infinities; 0x7F and 0xFF are NaN; max finite magnitude is 448.0.
// Out-of-range finite values and +/-inf saturate to +/-448.0 (matches PyTorch).
inline float32 e4m3fn_to_f32(uint8 b) {
    uint32 sign = (uint32(b & 0x80u)) << 24;
    uint32 exp = (b >> 3) & 0xFu;
    uint32 mant = b & 0x7u;
    if (exp == 0xFu && mant == 0x7u) {
        return bits_to_f32(sign | 0x7FC00000u);
    }
    if (exp == 0u) {
        if (mant == 0u) return bits_to_f32(sign);
        // Subnormal: (-1)^s * mant * 2^-9.
        float32 val = float32(mant) * (1.0f / 512.0f);
        return (b & 0x80u) ? -val : val;
    }
    uint32 f_exp = (exp + 120u) << 23;
    uint32 f_mant = mant << 20;
    return bits_to_f32(sign | f_exp | f_mant);
}

inline uint8 f32_to_e4m3fn(float32 f) {
    uint32 u = f32_to_bits(f);
    uint8 sign = uint8((u >> 24) & 0x80u);
    uint32 abs_u = u & 0x7FFFFFFFu;
    if (abs_u > 0x7F800000u) {
        return uint8(sign | 0x7Fu);
    }
    if (abs_u >= 0x43E00000u) {
        // >= 448.0 or +/-inf saturates to +/-448.0 (0x7E / 0xFE).
        return uint8(sign | 0x7Eu);
    }
    if (abs_u == 0u) {
        return sign;
    }
    uint32 exp_f = (abs_u >> 23) & 0xFFu;
    if (exp_f < 121u) {
        // Subnormal or underflow (< 2^-6).
        uint32 mant_f = (abs_u & 0x7FFFFFu) | 0x800000u;
        uint32 shift = 141u - exp_f;
        if (shift > 24u) return sign;
        uint32 q = mant_f >> shift;
        uint32 half = 1u << (shift - 1u);
        uint32 rem = mant_f & ((1u << shift) - 1u);
        if (rem > half || (rem == half && (q & 1u))) q += 1u;
        return uint8(sign | q);
    }
    uint32 new_exp = exp_f - 120u;
    uint32 mant3 = (abs_u >> 20) & 0x7u;
    uint32 round_bit = (abs_u >> 19) & 1u;
    uint32 sticky = (abs_u & 0x7FFFFu) != 0u;
    if (round_bit && (sticky || (mant3 & 1u))) {
        mant3 += 1u;
        if (mant3 == 8u) {
            mant3 = 0u;
            new_exp += 1u;
        }
    }
    if (new_exp > 15u || (new_exp == 15u && mant3 > 6u)) {
        return uint8(sign | 0x7Eu);
    }
    return uint8(sign | (new_exp << 3) | mant3);
}

// IEEE / OCP FP8 E5M2: 1 sign, 5 exponent (bias 15), 2 mantissa.
// +/-inf = 0x7C/0xFC; NaN = exp==31 && mant!=0; max finite = 57344.0.
inline float32 e5m2_to_f32(uint8 b) {
    uint32 sign = (uint32(b & 0x80u)) << 24;
    uint32 exp = (b >> 2) & 0x1Fu;
    uint32 mant = b & 0x3u;
    if (exp == 0x1Fu) {
        if (mant == 0u) return bits_to_f32(sign | 0x7F800000u);
        return bits_to_f32(sign | 0x7FC00000u | (mant << 21));
    }
    if (exp == 0u) {
        if (mant == 0u) return bits_to_f32(sign);
        // Subnormal: (-1)^s * mant * 2^-16.
        float32 val = float32(mant) * (1.0f / 65536.0f);
        return (b & 0x80u) ? -val : val;
    }
    uint32 f_exp = (exp + 112u) << 23;
    uint32 f_mant = mant << 21;
    return bits_to_f32(sign | f_exp | f_mant);
}

inline uint8 f32_to_e5m2(float32 f) {
    uint32 u = f32_to_bits(f);
    uint8 sign = uint8((u >> 24) & 0x80u);
    uint32 abs_u = u & 0x7FFFFFFFu;
    if (abs_u > 0x7F800000u) {
        return uint8(sign | 0x7Fu);
    }
    if (abs_u == 0x7F800000u) {
        return uint8(sign | 0x7Cu);
    }
    if (abs_u == 0u) {
        return sign;
    }
    uint32 exp_f = (abs_u >> 23) & 0xFFu;
    if (exp_f < 113u) {
        // Subnormal or underflow (< 2^-14).
        uint32 mant_f = (abs_u & 0x7FFFFFu) | 0x800000u;
        uint32 shift = 134u - exp_f;
        if (shift > 24u) return sign;
        uint32 q = mant_f >> shift;
        uint32 half = 1u << (shift - 1u);
        uint32 rem = mant_f & ((1u << shift) - 1u);
        if (rem > half || (rem == half && (q & 1u))) q += 1u;
        return uint8(sign | q);
    }
    uint32 new_exp = exp_f - 112u;
    uint32 mant2 = (abs_u >> 21) & 0x3u;
    uint32 round_bit = (abs_u >> 20) & 1u;
    uint32 sticky = (abs_u & 0xFFFFFu) != 0u;
    if (round_bit && (sticky || (mant2 & 1u))) {
        mant2 += 1u;
        if (mant2 == 4u) {
            mant2 = 0u;
            new_exp += 1u;
        }
    }
    if (new_exp >= 31u) {
        return uint8(sign | 0x7Cu);
    }
    return uint8(sign | (new_exp << 2) | mant2);
}

// OCP MX scale E8M0FNU: 0 sign, 8 exponent (bias 127), 0 mantissa.
// 0..254 represent 2^(x - 127); 255 (0xFF) is NaN.
inline float32 e8m0fnu_to_f32(uint8 b) {
    if (b == 0xFFu) return bits_to_f32(0x7FC00000u);
    if (b == 0u) return bits_to_f32(0x00400000u); // 2^-127
    return bits_to_f32(uint32(b) << 23);
}

inline uint8 f32_to_e8m0fnu(float32 f) {
    uint32 u = f32_to_bits(f);
    uint32 abs_u = u & 0x7FFFFFFFu;
    if (abs_u >= 0x7F800000u) return 0xFFu; // NaN / Inf -> NaN
    uint32 exp_f = (abs_u >> 23) & 0xFFu;
    if (exp_f == 0u) return 0u;
    return uint8(exp_f);
}

} // namespace fp8_detail

#define JT_DEFINE_FP8_TYPE(NAME, TO_F32, FROM_F32) \
struct NAME { \
    uint8 x; \
    inline NAME() : x(0) {} \
    inline NAME(float32 f) : x(fp8_detail::FROM_F32(f)) {} \
    inline NAME(float64 f) : x(fp8_detail::FROM_F32(static_cast<float32>(f))) {} \
    inline NAME(int32 v) : x(fp8_detail::FROM_F32(static_cast<float32>(v))) {} \
    inline NAME(int64 v) : x(fp8_detail::FROM_F32(static_cast<float32>(v))) {} \
    inline NAME(uint32 v) : x(fp8_detail::FROM_F32(static_cast<float32>(v))) {} \
    inline NAME(uint64 v) : x(fp8_detail::FROM_F32(static_cast<float32>(v))) {} \
    inline NAME(int16 v) : x(fp8_detail::FROM_F32(static_cast<float32>(v))) {} \
    inline NAME(uint16 v) : x(fp8_detail::FROM_F32(static_cast<float32>(v))) {} \
    inline NAME(int8 v) : x(fp8_detail::FROM_F32(static_cast<float32>(v))) {} \
    inline NAME(uint8 v) : x(fp8_detail::FROM_F32(static_cast<float32>(v))) {} \
    inline NAME(bool v) : x(fp8_detail::FROM_F32(v ? 1.0f : 0.0f)) {} \
    inline static NAME from_bits(uint8 bits) { NAME r; r.x = bits; return r; } \
    inline operator float32() const { return fp8_detail::TO_F32(x); } \
}; \
static_assert(sizeof(NAME) == 1, #NAME " must be 1 byte"); \
inline NAME operator+(NAME a, NAME b) { return NAME(float32(a) + float32(b)); } \
inline NAME operator-(NAME a, NAME b) { return NAME(float32(a) - float32(b)); } \
inline NAME operator*(NAME a, NAME b) { return NAME(float32(a) * float32(b)); } \
inline NAME operator/(NAME a, NAME b) { return NAME(float32(a) / float32(b)); } \
inline NAME operator-(NAME a) { return NAME(-float32(a)); } \
inline bool operator<(NAME a, NAME b) { return float32(a) < float32(b); } \
inline bool operator<=(NAME a, NAME b) { return float32(a) <= float32(b); } \
inline bool operator>(NAME a, NAME b) { return float32(a) > float32(b); } \
inline bool operator>=(NAME a, NAME b) { return float32(a) >= float32(b); } \
inline bool operator==(NAME a, NAME b) { return float32(a) == float32(b); } \
inline bool operator!=(NAME a, NAME b) { return float32(a) != float32(b); } \
inline NAME max(NAME a, NAME b) { \
    float32 fa = float32(a), fb = float32(b); \
    if (std::isnan(fa)) return a; \
    if (std::isnan(fb)) return b; \
    return fa >= fb ? a : b; \
} \
inline NAME min(NAME a, NAME b) { \
    float32 fa = float32(a), fb = float32(b); \
    if (std::isnan(fa)) return a; \
    if (std::isnan(fb)) return b; \
    return fa <= fb ? a : b; \
} \
inline NAME abs(NAME a) { return NAME(std::fabs(float32(a))); }

JT_DEFINE_FP8_TYPE(float8_e4m3fn, e4m3fn_to_f32, f32_to_e4m3fn)
JT_DEFINE_FP8_TYPE(float8_e5m2, e5m2_to_f32, f32_to_e5m2)
JT_DEFINE_FP8_TYPE(float8_e8m0fnu, e8m0fnu_to_f32, f32_to_e8m0fnu)

#undef JT_DEFINE_FP8_TYPE

} // namespace jittor
