// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
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

struct __jk_int128 {
    int64 a,b;
};
struct __jk_int256 {
    int64 a,b,c,d;
};

extern thread_local JitKey jk;
typedef JitKey JK;

inline JK& operator<<(JK& jk, const char* s) {
    int i;
    for (i=0; s[i]; i++)
        jk.buffer[jk.size+i] = s[i];
    jk.size += i;
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

inline JK& operator<<(JK& jk, long long int c) {
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
    jk << JK::key << key << JK::val << val << JK::end;
}

template <class Ta, class Tb, class Tc>
void add_jit_define(JK& jk, const Ta& key, const Tb& i, const Tc& val) {
    jk << JK::key << key << i << JK::val << val << JK::end;
}


template <class Ta>
void add_jit_define(JK& jk, const Ta& key, const JK::hex& val) {
    jk << JK::key << key << JK::hex_val << val << JK::end;
}

template <class Ta, class Tb>
void add_jit_define(JK& jk, const Ta& key, const Tb& i, const JK::hex& val) {
    jk << JK::key << key << i << JK::hex_val << val << JK::end;
}

template <class Ta>
void add_jit_define(JK& jk, const Ta& key, const JK::hex1& val) {
    jk << JK::key << key << JK::hex_val << val << JK::end;
}

template <class Ta, class Tb>
void add_jit_define(JK& jk, const Ta& key, const Tb& i, const JK::hex1& val) {
    jk << JK::key << key << i << JK::hex_val << val << JK::end;
}

template <class Ta>
void add_jit_define(JK& jk, const Ta& key, const JK::hex2& val) {
    jk << JK::key << key << JK::hex_val << val << JK::end;
}

template <class Ta, class Tb>
void add_jit_define(JK& jk, const Ta& key, const Tb& i, const JK::hex2& val) {
    jk << JK::key << key << i << JK::hex_val << val << JK::end;
}


// begin of const string
#define MAX_CONST_CHAR 32

#define _CS_MIN(a,b) (a)<(b)?(a):(b)

#define _CS_T(s)\
getChr(s,0),\
getChr(s,1),\
getChr(s,2),\
getChr(s,3),\
getChr(s,4),\
getChr(s,5),\
getChr(s,6),\
getChr(s,7),\
getChr(s,8),\
getChr(s,9),\
getChr(s,10),\
getChr(s,11),\
getChr(s,12),\
getChr(s,13),\
getChr(s,14),\
getChr(s,15),\
getChr(s,16),\
getChr(s,17),\
getChr(s,18),\
getChr(s,19),\
getChr(s,20),\
getChr(s,21),\
getChr(s,22),\
getChr(s,23),\
getChr(s,24),\
getChr(s,25),\
getChr(s,26),\
getChr(s,27),\
getChr(s,28),\
getChr(s,29),\
getChr(s,30),\
getChr(s,31),\
getChr(s,32),\
getChr(s,33),\
getChr(s,34),\
getChr(s,35)

#define getChr(name, ii) ((_CS_MIN(ii,MAX_CONST_CHAR))<sizeof(name)/sizeof(*name)?name[ii]:0)

#define _CS(str) _CS_G<_CS_T(str)>()

template <char c1, char c2, char c3, char c4, char... Chars_> struct _CS_G {
 };

template<> struct _CS_G<0,0,0,0> {};

template <char c1, char c2, char c3, char c4, char... Chars_>
inline JK& operator<<(JK& jk, const _CS_G<c1,c2,c3,c4,Chars_...>& _) {
    ((int*)(jk.buffer+jk.size))[0] = c4*(1<<24)+c3*(1<<16)+c2*(1<<8)+c1;
    if (c4) {
        jk.size += 4;
        jk << _CS_G<Chars_...>();
    } else
    if (c3) {
        jk.size += 3;
    } else
    if (c2) {
        jk.size += 2;
    } else
    if (c1) {
        jk.size += 1;
    }
    return jk;
}

template <>
inline JK& operator<<(JK& jk, const _CS_G<0,0,0,0>& _) {
    return jk;
}


inline JK& operator<<(JK& jk, float64 f) {
    return jk << _CS("itof(0x") << JK::hex(ftoi(f)) << ')';
}

} // jittor