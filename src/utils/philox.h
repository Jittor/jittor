// ***************************************************************
// Copyright (c) 2026 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define JITTOR_PHILOX_HD __host__ __device__
#else
#define JITTOR_PHILOX_HD
#endif

namespace jittor {

struct Philox4x32Result {
    uint32_t x0, x1, x2, x3;
};

struct PhiloxUint64Product {
    uint64_t low, high;
};

JITTOR_PHILOX_HD inline uint32_t philox_mul_hi(uint32_t a, uint32_t b) {
#ifdef __CUDA_ARCH__
    return __umulhi(a, b);
#else
    return uint32_t((uint64_t(a) * uint64_t(b)) >> 32);
#endif
}

// Random123 Philox4x32-10. The seed is the key and the monotonically
// increasing block index is the counter, so changing a seed does not merely
// shift into a neighboring seed's stream.
JITTOR_PHILOX_HD inline Philox4x32Result philox4x32_10(
        uint64_t seed, uint64_t counter_low, uint64_t counter_high) {
    uint32_t c0 = uint32_t(counter_low);
    uint32_t c1 = uint32_t(counter_low >> 32);
    uint32_t c2 = uint32_t(counter_high);
    uint32_t c3 = uint32_t(counter_high >> 32);
    uint32_t k0 = uint32_t(seed);
    uint32_t k1 = uint32_t(seed >> 32);
    for (int round = 0; round < 10; ++round) {
        const uint32_t hi0 = philox_mul_hi(0xD2511F53u, c0);
        const uint32_t lo0 = 0xD2511F53u * c0;
        const uint32_t hi1 = philox_mul_hi(0xCD9E8D57u, c2);
        const uint32_t lo1 = 0xCD9E8D57u * c2;
        const uint32_t next0 = hi1 ^ c1 ^ k0;
        const uint32_t next2 = hi0 ^ c3 ^ k1;
        c0 = next0;
        c1 = lo1;
        c2 = next2;
        c3 = lo0;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return {c0, c1, c2, c3};
}

JITTOR_PHILOX_HD inline Philox4x32Result philox4x32_10(
        uint64_t seed, uint64_t counter) {
    return philox4x32_10(seed, counter, 0);
}

JITTOR_PHILOX_HD inline uint64_t philox_uint64(
        const Philox4x32Result& value, int pair) {
    return pair == 0
        ? uint64_t(value.x0) | (uint64_t(value.x1) << 32)
        : uint64_t(value.x2) | (uint64_t(value.x3) << 32);
}

JITTOR_PHILOX_HD inline PhiloxUint64Product philox_mul_wide(
        uint64_t a, uint64_t b) {
#ifdef __CUDA_ARCH__
    return {a * b, __umul64hi(a, b)};
#else
    const unsigned __int128 product =
        static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
    return {uint64_t(product), uint64_t(product >> 64)};
#endif
}

// Open on zero so Box-Muller never evaluates log(0); like cuRAND's uniform
// API the upper endpoint may be one.
JITTOR_PHILOX_HD inline float philox_uniform_float(uint32_t value) {
    return (float(value) + 1.0f) * 2.3283064365386962890625e-10f;
}

JITTOR_PHILOX_HD inline double philox_uniform_double(
        uint32_t high, uint32_t low) {
    const uint64_t bits = (uint64_t(high >> 5) << 26) | uint64_t(low >> 6);
    return (double(bits) + 1.0) * 1.1102230246251565404236316680908e-16;
}

} // namespace jittor

#undef JITTOR_PHILOX_HD
