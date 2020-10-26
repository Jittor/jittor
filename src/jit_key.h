// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "misc/nano_string.h"
#include "misc/nano_vector.h"

namespace jittor {

struct JitKey {
    static constexpr size_t buffer_size = 2*1024*1024;
    static constexpr char
        key = '[',
        val = ':',
        hex_val = '=',
        end = ']';
    int64 size=0;
    uint64 flags=0;
    char buffer[buffer_size];

    JitKey();
    ~JitKey();

    inline void clear() {size = flags = 0;}
    inline void finilize() { buffer[size] = 0; }
    inline bool empty() { return buffer[size-1] != end; }
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

extern thread_local JitKey jk;
typedef JitKey JK;

inline JK& operator<<(JK& jk, const char* s) {
    while (*s) jk.buffer[jk.size++] = *s, s++;
    return jk;
}

inline JK& operator<<(JK& jk, const string& s) {
    for (uint64 i=0; i<s.size(); i++)
        jk.buffer[jk.size+i] = s[i];
    jk.size += s.size();
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

inline JK& operator<<(JK& jk, long long c) {
    return jk << (int64)c;
}

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

inline JK& operator<<(JK& jk, float64 f) {
    return jk << "itof(0x" << JK::hex(ftoi(f)) << ')';
}

inline JK& operator<<(JK& jk, const NanoString& ns) {
    return jk << ns.to_cstring();
}

vector<pair<string,string>> parse_jit_keys(const string& s);

template <class Ta, class Tb>
void add_jit_define(const Ta& key, const Tb& val) {
    jk << JK::key << key << JK::val << val << JK::end;
}

template <class Ta, class Tb, class Tc>
void add_jit_define(const Ta& key, const Tb& i, const Tc& val) {
    jk << JK::key << key << i << JK::val << val << JK::end;
}


template <class Ta>
void add_jit_define(const Ta& key, const JK::hex& val) {
    jk << JK::key << key << JK::hex_val << val << JK::end;
}

template <class Ta, class Tb>
void add_jit_define(const Ta& key, const Tb& i, const JK::hex& val) {
    jk << JK::key << key << i << JK::hex_val << val << JK::end;
}

template <class Ta>
void add_jit_define(const Ta& key, const JK::hex1& val) {
    jk << JK::key << key << JK::hex_val << val << JK::end;
}

template <class Ta, class Tb>
void add_jit_define(const Ta& key, const Tb& i, const JK::hex1& val) {
    jk << JK::key << key << i << JK::hex_val << val << JK::end;
}

template <class Ta>
void add_jit_define(const Ta& key, const JK::hex2& val) {
    jk << JK::key << key << JK::hex_val << val << JK::end;
}

template <class Ta, class Tb>
void add_jit_define(const Ta& key, const Tb& i, const JK::hex2& val) {
    jk << JK::key << key << i << JK::hex_val << val << JK::end;
}

} // jittor