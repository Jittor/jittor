// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

uint constexpr const_hash(const char *input) {
    return *input ?
        static_cast<uint>(*input) + 55 * const_hash(input + 1) :
        0;
}

/* simple hash function */
// @pyjt(hash)
inline uint hash(const char* input) {
    uint v=0, mul=1;
    while (*input) {
        v += mul * (uint)*input;
        mul *= 55;
        input++;
    }
    return v;
}


/* Polynomial hash, kept only for anything that still wants a cheap 64-bit
   digest of a short string. It is linear -- v = sum(c_i * 257^i) mod 2^64 --
   so collisions can be constructed by hand, and it must not be used to decide
   whether a build product is up to date. See content_hash below. */
inline uint64 hash64(const string& input) {
    uint64 v=0, mul=1;
    for (char c : input) {
        v += mul * (uint64)c;
        mul *= 257;
    }
    return v;
}

/* SHA-256 of a file's contents, hex, as the compile cache's freshness test.

   This decides whether an object file may be reused, and it used to be the
   polynomial above: weaker than the MD5 the same repository already used for
   downloads, and linear, so two sources with the same digest can be produced
   deliberately. The cost is not interesting here -- a warm import hashes a few
   tens of MB, which is milliseconds -- and being able to check the answer
   against any other SHA-256 implementation is worth more than the speed.

   Self-contained on purpose: this header is compiled into jit_utils_core,
   which is built before anything else and cannot depend on a library. */
namespace sha256_detail {

inline uint32 rotr(uint32 x, int n) { return (x >> n) | (x << (32 - n)); }

static const uint32 K[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,
    0x923f82a4u,0xab1c5ed5u,0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
    0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,0xe49b69c1u,0xefbe4786u,
    0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,
    0x06ca6351u,0x14292967u,0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
    0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,0xa2bfe8a1u,0xa81a664bu,
    0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,
    0x5b9cca4fu,0x682e6ff3u,0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
    0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};

inline void round_block(uint32* h, const unsigned char* p) {
    uint32 w[64];
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32)p[i*4] << 24) | ((uint32)p[i*4+1] << 16)
             | ((uint32)p[i*4+2] << 8) | (uint32)p[i*4+3];
    for (int i = 16; i < 64; i++) {
        uint32 s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
        uint32 s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (int i = 0; i < 64; i++) {
        uint32 S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        uint32 ch = (e & f) ^ ((~e) & g);
        uint32 t1 = hh + S1 + ch + K[i] + w[i];
        uint32 S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        uint32 maj = (a & b) ^ (a & c) ^ (b & c);
        uint32 t2 = S0 + maj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d;
    h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
}

} // sha256_detail

inline string content_hash(const string& input) {
    uint32 h[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                   0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    const unsigned char* p = (const unsigned char*)input.data();
    size_t n = input.size(), i = 0;
    for (; i + 64 <= n; i += 64)
        sha256_detail::round_block(h, p + i);
    unsigned char tail[128] = {0};
    size_t rest = n - i;
    for (size_t j = 0; j < rest; j++) tail[j] = p[i+j];
    tail[rest] = 0x80;
    size_t total = (rest + 9 <= 64) ? 64 : 128;
    uint64 bits = (uint64)n * 8;
    for (int j = 0; j < 8; j++)
        tail[total-1-j] = (unsigned char)((bits >> (8*j)) & 0xff);
    for (size_t off = 0; off < total; off += 64)
        sha256_detail::round_block(h, tail + off);
    static const char* digits = "0123456789abcdef";
    string out(64, '0');
    for (int j = 0; j < 8; j++)
        for (int k = 0; k < 4; k++) {
            unsigned char byte = (unsigned char)((h[j] >> (24 - 8*k)) & 0xff);
            out[j*8 + k*2]     = digits[byte >> 4];
            out[j*8 + k*2 + 1] = digits[byte & 0xf];
        }
    return out;
}

} // jittor
