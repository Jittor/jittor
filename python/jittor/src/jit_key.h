// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cstring>
#include "common.h"
#include "misc/nano_string.h"
#include "misc/nano_vector.h"

namespace jittor {

struct JitKey {
    static constexpr size_t buffer_size = 2*1024*1024;
    static constexpr const char
        *key = "«",
        val = ':',
        hex_val = '=';
    int64 size=0;
    uint64 flags=0;
    char buffer[buffer_size];

    JitKey();
    ~JitKey();

    inline void clear() {size = flags = 0;}
    inline void finilize() { buffer[size] = 0; }
    inline bool empty() { return !size; }
    inline const char* to_cstring() {
        return &buffer[0];
    }
    inline string to_string() {
        return string(&buffer[0], size);
    }

    struct hex {
        uint64 data;
        explicit hex(uint64 data) : data(data) {}
    };

    struct hex1 {
        uint data;
        explicit hex1(uint data) : data(data) {}
    };

    struct shex1 {
        int data;
        explicit shex1(int data) : data(data) {}
    };

    struct hex2 {
        uint data;
        explicit hex2(uint data) : data(data) {}
    };

    struct Oxhex {
        uint64 data;
        explicit Oxhex(uint64 data) : data(data) {}
    };

    struct Oxhex1 {
        uint data;
        explicit Oxhex1(uint data) : data(data) {}
    };

    struct Oxhex2 {
        uint data;
        explicit Oxhex2(uint data) : data(data) {}
    };
};

struct __jk_int128 {
    int64 a,b;
};
struct __jk_int256 {
    int64 a,b,c,d;
};

typedef JitKey JK;
EXTERN_LIB JK& get_jk();

inline void jk_put_str_with_len(JK& jk, const char* a, int n) {
    char* xx = &jk.buffer[jk.size];
    int i=0;
    while (i+32<=n) {
        ((__jk_int256*)(xx+i))[0] = ((const __jk_int256*)(a+i))[0];
        i+=32;
    }
    while (i+16<=n) {
        ((__jk_int128*)(xx+i))[0] = ((const __jk_int128*)(a+i))[0];
        i+=16;
    }
    while (i+8<=n) {
        ((long long*)(xx+i))[0] = ((const long long*)(a+i))[0];
        i+=8;
    }
    while (i+4<=n) {
        ((int*)(xx+i))[0] = ((const int*)(a+i))[0];
        i+=4;
    }
    while (i+2<=n) {
        ((int16_t*)(xx+i))[0] = ((const int16_t*)(a+i))[0];
        i+=2;
    }
    while (i+1<=n) {
        ((char*)(xx+i))[0] = ((const char*)(a+i))[0];
        i+=1;
    }
    jk.size += n;
}

inline JK& operator<<(JK& jk, const char* s) {
    jk_put_str_with_len(jk, s, strlen(s));
    return jk;
}

inline JK& operator<<(JK& jk, const string& s) {
    auto a = (__jk_int256*)(jk.buffer+jk.size);
    auto b = (__jk_int256*)(&s[0]);
    auto len = s.size();
    uint64 i=0;
    for (; i+32<=len; i+=32)
        a[i/32] = b[i/32];
        
    for (; i<len; i++)
        jk.buffer[jk.size+i] = s[i];
    jk.size += len;
    return jk;
}

inline JK& operator<<(JK& jk, const char c) {
    jk.buffer[jk.size++] = c;
    return jk;
}

inline JK& operator<<(JK& jk, const JK::hex1& h) {
    uint8 data = h.data % 16;
    return jk << (char)((data<10) ? data+'0' : data-10+'a');
}

inline JK& operator<<(JK& jk, const JK::shex1& h) {
    if (h.data<0)
        return jk << '-' << JK::hex1(-h.data);
    else
        return jk << JK::hex1(h.data);
}

inline JK& operator<<(JK& jk, const JK::hex2& h) {
    return jk << JK::hex1(h.data>>4) << JK::hex1(h.data);
}

inline JK& operator<<(JK& jk, const JK::hex& h) {
    auto a = h.data;
    uint nbits = 64 - lzcnt(a);
    nbits = a ? nbits-1 : 0;
    int i=nbits/4;
    for (; i>=0; i--)
        jk << JK::hex1(a >> (i*4));
    return jk;
}

inline JK& operator<<(JK& jk, const JK::Oxhex& h) {
    return jk << "0x" << JK::hex(h.data);
}

inline JK& operator<<(JK& jk, const JK::Oxhex1& h) {
    return jk << "0x" << JK::hex1(h.data);
}

inline JK& operator<<(JK& jk, const JK::Oxhex2& h) {
    return jk << "0x" << JK::hex2(h.data);
}

inline JK& operator<<(JK& jk, int c) {
    if (c<0) {
        c = -c;
        jk << '-';
    }
    return jk << JK::hex(c);
}

inline JK& operator<<(JK& jk, uint c) {
    return jk << JK::hex(c);
}

inline JK& operator<<(JK& jk, int64 c) {
    if (c<0) {
        c = -c;
        jk << '-';
    }
    return jk << JK::hex(c);
}

#ifdef __linux__
inline JK& operator<<(JK& jk, int64_t c) {
    return jk << (int64)c;
}
#endif

inline JK& operator<<(JK& jk, uint64 c) {
    return jk << JK::hex(c);
}

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif
static inline uint64 ftoi(float64 a) { return *(uint64*)&a; }
static inline float64 itof(uint64 a) { return *(float64*)&a; }
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

inline JK& operator<<(JK& jk, const NanoString& ns) {
    auto a = (__jk_int128*)(jk.buffer+jk.size);
    auto b = (__jk_int128*)(ns.to_cstring());
    auto len = ns.len();
    a[0] = b[0];
    jk.size += len;
    return jk;
}

vector<pair<string,string>> parse_jit_keys(const string& s);

template <class Ta, class Tb>
void add_jit_define(JK& jk, const Ta& key, const Tb& val) {
    jk << JK::key << key << JK::val << val;
}

template <class Ta, class Tb, class Tc>
void add_jit_define(JK& jk, const Ta& key, const Tb& i, const Tc& val) {
    jk << JK::key << key << i << JK::val << val;
}


template <class Ta>
void add_jit_define(JK& jk, const Ta& key, const JK::hex& val) {
    jk << JK::key << key << JK::hex_val << val;
}

template <class Ta, class Tb>
void add_jit_define(JK& jk, const Ta& key, const Tb& i, const JK::hex& val) {
    jk << JK::key << key << i << JK::hex_val << val;
}

template <class Ta>
void add_jit_define(JK& jk, const Ta& key, const JK::hex1& val) {
    jk << JK::key << key << JK::hex_val << val;
}

template <class Ta, class Tb>
void add_jit_define(JK& jk, const Ta& key, const Tb& i, const JK::hex1& val) {
    jk << JK::key << key << i << JK::hex_val << val;
}

template <class Ta>
void add_jit_define(JK& jk, const Ta& key, const JK::hex2& val) {
    jk << JK::key << key << JK::hex_val << val;
}

template <class Ta, class Tb>
void add_jit_define(JK& jk, const Ta& key, const Tb& i, const JK::hex2& val) {
    jk << JK::key << key << i << JK::hex_val << val;
}

#define _CS(x) x

inline JK& operator<<(JK& jk, float64 f) {
    return jk << "itof(0x" << JK::hex(ftoi(f)) << ')';
}

} // jittor