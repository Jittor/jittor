// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cstring>
#include "common.h"
#include "type/nano_string.h"
#include "type/nano_vector.h"

namespace jittor {

// The buffer a jit key is assembled in.
//
// Every writer below goes through `reserve()` before it stores anything, and a
// key that would grow past `max_size` throws a `UserError` -- which reaches
// Python as a catchable RuntimeError.
//
// What this replaces: `buffer` used to be `char buffer[2*1024*1024]` inline in
// this object, with no length check in any `operator<<`. They all stored at
// `buffer[size]` and trusted the caller. Standing in for the check was an
// mprotect'ed guard page at the tail of the array, installed by the
// constructor, so an over-long key hit PROT_NONE and raised SIGSEGV. That is
// not a bounds check:
//
//   * the condition it reports ("this graph's key does not fit") is
//     recoverable, and it turned it into a process kill delivered from a
//     signal handler -- from which nothing can be caught, because throwing out
//     of a handler is undefined behaviour (see utils/log.cc);
//   * it only protected the *last page*. `jk_put_str_with_len` copies in 32
//     byte blocks, so a store can begin below the guard page and end above it
//     -- and a `<<` that starts more than a page past the end skips the guard
//     entirely and corrupts whatever follows;
//   * `jk` is thread_local, so the 2 MB was per thread (16 compile workers =
//     32 MB of untouched pages), and the guard page mprotect'ed a page of the
//     thread's own storage.
//
// The default limit is the old capacity, so nothing that fits today stops
// fitting; the differences are that exceeding it is reported instead of fatal,
// that the memory is grown on demand rather than reserved, and that the limit
// is a flag.
DECLARE_FLAG(int, jit_key_max_size);

struct JitKey {
    // Allocated eagerly by the constructor: `to_cstring()` must never return
    // null, and the first key of a run is usually well under this.
    static constexpr size_t init_capacity = 8*1024;
    // Bytes past `size` that a single store may touch: `operator<<(const
    // NanoString&)` always stores 16 bytes however short the string is, and
    // `finilize()` writes a terminator at `buffer[size]`. `reserve` keeps this
    // much room above the logical end at all times, so those two need no
    // special case.
    static constexpr size_t tail_slack = 64;
    static constexpr const char
        *key = "«",
        val = ':',
        hex_val = '=';
    int64 size=0;
    uint64 flags=0;
    char* buffer=nullptr;
    size_t capacity=0;
    // What `reserve` compares against: the smaller of what is allocated and
    // what `jit_key_max_size` allows. One number, so the check under every
    // `<<` below is a single comparison rather than two.
    //
    // Recomputed by `clear()`, i.e. once per key, which is what makes the flag
    // a runtime flag: recomputing it only in `grow()` would leave a *lowered*
    // limit unenforced until the buffer -- which only ever grows -- happened
    // to need extending again.
    size_t check_at=0;

    JitKey();
    ~JitKey();
    JitKey(const JitKey&) = delete;
    JitKey& operator=(const JitKey&) = delete;

    inline static size_t size_limit() {
        return jit_key_max_size > 0 ? (size_t)jit_key_max_size : 1;
    }
    inline size_t effective_check_at() const {
        size_t allowed = size_limit() + tail_slack;
        return capacity < allowed ? capacity : allowed;
    }

    // Make room for `n` more bytes, or throw. Out of line: this is the cold
    // half of `reserve`, which sits under every `<<` in this file.
    void grow(size_t n);
    inline void reserve(size_t n) {
        if (PREDICT_BRANCH_NOT_TAKEN(
                (size_t)size + n + tail_slack > check_at))
            grow(n);
    }

    inline void clear() {
        size = flags = 0;
        check_at = effective_check_at();
    }
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

    struct dec1 {
        uint data;
        explicit dec1(uint data) : data(data) {}
    };

    struct dec2 {
        uint data;
        explicit dec2(uint data) : data(data) {}
    };

    struct dec3 {
        uint data;
        explicit dec3(uint data) : data(data) {}
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
    jk.reserve(n);
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
    auto len = s.size();
    jk.reserve(len);
    auto a = (__jk_int256*)(jk.buffer+jk.size);
    auto b = (__jk_int256*)(&s[0]);
    uint64 i=0;
    for (; i+32<=len; i+=32)
        a[i/32] = b[i/32];
        
    for (; i<len; i++)
        jk.buffer[jk.size+i] = s[i];
    jk.size += len;
    return jk;
}

inline JK& operator<<(JK& jk, const char c) {
    jk.reserve(1);
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

inline JK& operator<<(JK& jk, const JK::dec3& h) {
    uint8 a = h.data % 10;
    uint8 b = h.data / 10 % 10;
    uint8 c = h.data / 100;
    if (c) jk << (char)(c+'0'), jk << (char)(b+'0');
    else if (b) jk << (char)(b+'0');
    return jk << (char)(a+'0');
}

inline JK& operator<<(JK& jk, const JK::dec2& h) {
    uint8 a = h.data % 10;
    uint8 b = h.data / 10;
    if (b) jk << (char)(b+'0');
    return jk << (char)(a+'0');
}

inline JK& operator<<(JK& jk, const JK::dec1& h) {
    uint8 a = h.data % 10;
    return jk << (char)(a+'0');
}

inline std::ostream& operator<<(std::ostream& os, const JK::dec3& h) {
    uint8 a = h.data % 10;
    uint8 b = h.data / 10 %10;
    uint8 c = h.data / 100;
    if (c) os << (char)(c+'0'), os << (char)(b+'0');
    else if (b) os << (char)(b+'0');
    return os << (char)(a+'0');
}

inline std::ostream& operator<<(std::ostream& os, const JK::dec2& h) {
    uint8 a = h.data % 10;
    uint8 b = h.data / 10;
    if (b) os << (char)(b+'0');
    return os << (char)(a+'0');
}

inline std::ostream& operator<<(std::ostream& os, const JK::dec1& h) {
    uint8 a = h.data % 10;
    return os << (char)(a+'0');
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
    auto len = ns.len();
    // The store below is 16 bytes wide whatever `len` is; `tail_slack` covers
    // the difference, but the request has to be for what is written.
    jk.reserve(sizeof(__jk_int128));
    auto a = (__jk_int128*)(jk.buffer+jk.size);
    auto b = (__jk_int128*)(ns.to_cstring());
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
