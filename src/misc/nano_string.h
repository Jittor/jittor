// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {


#define FOR_ALL_NS(m) \
\
    m(void) \
    m(bool) \
    m(int8) \
    m(int16) \
    m(int32) \
    m(int64) \
    m(uint8) \
    m(uint16) \
    m(uint32) \
    m(uint64) \
    m(float32) \
    m(float64) \
\
    m(pow) \
    m(maximum) \
    m(minimum) \
    m(add) \
    m(subtract) \
    m(multiply) \
    m(divide) \
    m(floor_divide) \
    m(mod) \
    m(less) \
    m(less_equal) \
    m(greater) \
    m(greater_equal) \
    m(equal) \
    m(not_equal) \
    m(left_shift) \
    m(right_shift) \
    m(logical_and) \
    m(logical_or) \
    m(logical_xor) \
    m(bitwise_and) \
    m(bitwise_or) \
    m(bitwise_xor) \
    m(mean) \
\
    m(abs) \
    m(negative) \
    m(logical_not) \
    m(bitwise_not) \
    m(log) \
    m(exp) \
    m(sqrt) \
    m(round) \
    m(floor) \
    m(ceil) \
    m(cast) \
    \
    m(sin) \
    m(asin) \
    m(sinh) \
    m(asinh) \
    m(tan) \
    m(atan) \
    m(tanh) \
    m(atanh) \
    m(cos) \
    m(acos) \
    m(cosh) \
    m(acosh) \
    m(sigmoid) \
    \
    m(uniform) \
    m(normal) \

struct NanoString;
#define DECLEAR_NS(T) extern NanoString ns_##T;
FOR_ALL_NS(DECLEAR_NS);

// @pyjt(NanoString)
struct NanoString {
    typedef uint16 ns_t;
    enum Flags {
        // bit0~7: index
        _index=0, _index_nbits=8,
        _n=_index_nbits,

        // bit0-1: type
        _type=_n, _type_nbits=2,
        _other=0, _dtype=1, _unary=2, _binary=3,
        // bit2: is bool
        _bool=_n+2,
        // bit3: is int
        _int=_n+3,
        // bit4: is unsigned
        _unsigned=_n+4,
        // bit5: is float
        _float=_n+5,
        // bit6-7: dsize(1,2,4,8 byte)
        _dsize=_n+6, _dsize_nbits=2,
    };
    ns_t data=0;

    static unordered_map<string, NanoString> __string_to_ns;
    static vector<const char*> __ns_to_string;

    inline void set(Flags f, ns_t a=1, ns_t nbits=1) {
        ns_t mask = (((1u<<nbits)-1)<<f);
        data = (data & ~mask) | ((a<<f)&mask);
    }

    inline ns_t get(Flags f, ns_t nbits=1) const {
        return (data>>f) & ((1u<<nbits)-1);
    }
    inline ns_t index() const { return get(_index, _index_nbits); }
    inline ns_t type() const { return get(_type, _type_nbits); }
    inline ns_t is_bool() const { return get(_bool); }
    inline ns_t is_int() const { return get(_int); }
    inline ns_t is_unsigned() const { return get(_unsigned); }
    inline ns_t is_float() const { return get(_float); }
    inline ns_t dsize() const { return 1<<get(_dsize, _dsize_nbits); }
    inline ns_t is_dtype() const { return get(_type, _type_nbits)==_dtype; }
    inline ns_t is_binary() const { return get(_type, _type_nbits)==_binary; }
    inline ns_t is_unary() const { return get(_type, _type_nbits)==_unary; }

    inline NanoString() {}
    // @pyjt(__init__)
    inline NanoString(const char* s) {
        auto iter = __string_to_ns.find(s);
        ASSERT(iter != __string_to_ns.end()) << s;
        data = iter->second.data;
    }
    // @pyjt(__init__)
    inline NanoString(const NanoString& other) : data(other.data) {}
    inline NanoString(const string& s) : NanoString(s.c_str()) {}
    // @pyjt(__repr__)
    inline const char* to_cstring() const
        { return __ns_to_string[index()]; }
};

// force_type = 1 for int, 2 for float
inline 
NanoString dtype_infer(NanoString v1, NanoString v2, int force_type=0) {
    bool is_float = v1.is_float() || v2.is_float();
    int dsize = std::max(v1.dsize(), v2.dsize());
    if (force_type == 1)
        is_float = false;
    else if (force_type == 2)
        is_float = true;
    if (is_float) {
        if (dsize==4) return ns_float32;
        return ns_float64;
    } else {
        if (dsize==8) return ns_int64;
        if (dsize==4) return ns_int32;
        if (dsize==2) return ns_int16;
        return v1;
    }
}

// @pyjt(NanoString.__eq__)
inline bool eq(const NanoString& a, const NanoString& b) {
    return a.data == b.data;
}

// @pyjt(NanoString.__ne__)
inline bool ne(const NanoString& a, const NanoString& b) {
    return a.data != b.data;
}

inline bool operator==(const NanoString& a, const NanoString& b) {
    return a.data == b.data;
}
inline bool operator!=(const NanoString& a, const NanoString& b) {
    return a.data != b.data;
}

inline std::ostream& operator<<(std::ostream& os, const NanoString& v) {
    return os << v.to_cstring();
}

}
