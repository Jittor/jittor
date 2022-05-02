// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

constexpr int ns_max_size = 256;
constexpr int ns_max_len = 16;

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
    m(float16) \
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
    m(round_int) \
    m(floor_int) \
    m(ceil_int) \
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
    m(erf) \
    m(erfinv) \
    m(sigmoid) \
    \
    m(uniform) \
    m(normal) \

struct NanoString;
#define DECLEAR_NS(T) EXTERN_LIB NanoString ns_##T;
FOR_ALL_NS(DECLEAR_NS);


EXTERN_LIB unordered_map<string, NanoString> __string_to_ns;
EXTERN_LIB char __ns_to_string[];
EXTERN_LIB int __ns_len[];

// @pyjt(NanoString)
struct NanoString {
    typedef uint32 ns_t;
    enum Flags {
        // bit0~7: index
        _index=0, _index_nbits=7,
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
        // bit8: white list
        _white_list=_n+8,
        // bit9: backward opt
        _no_need_back_in=_n+9,
        _no_need_back_out=_n+10,
    };
    ns_t data=0;

    inline void set(Flags f, ns_t a=1, ns_t nbits=1) {
        ns_t mask = (((1u<<nbits)-1)<<f);
        data = (data & ~mask) | ((a<<f)&mask);
    }

    inline ns_t get(Flags f, ns_t nbits=1) const {
        return (data>>f) & ((1u<<nbits)-1);
    }
    inline ns_t index() const { return get(_index, _index_nbits); }
    inline int len() const { return __ns_len[index()]; }
    inline ns_t type() const { return get(_type, _type_nbits); }
    // @pyjt(is_bool)
    inline bool is_bool() const { return get(_bool); }
    // @pyjt(is_int)
    inline bool is_int() const { return get(_int); }
    inline bool is_unsigned() const { return get(_unsigned); }
    // @pyjt(is_float)
    inline bool is_float() const { return get(_float); }
    inline ns_t is_white() const { return get(_white_list); }
    inline ns_t dsize() const { return 1<<get(_dsize, _dsize_nbits); }
    inline ns_t dsize_() const { return get(_dsize, _dsize_nbits); }
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
        { return __ns_to_string+index()*ns_max_len; }
    inline char* to_cstring()
        { return __ns_to_string+index()*ns_max_len; }
};

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

EXTERN_LIB int amp_reg;
constexpr int amp_prefer32 = 1;
constexpr int amp_prefer16 = 2;
constexpr int amp_keep_reduce = 4;
constexpr int amp_keep_white = 8;
constexpr int amp_array_prefer = 16;

inline NanoString float_dtype(int dsize_) {
    if (amp_reg & amp_prefer32) return ns_float32;
    if (amp_reg & amp_prefer16) return ns_float16;
    return (dsize_ == 3) ? ns_float64 : 
        (dsize_ == 2 ) ? ns_float32 : ns_float16;
}

inline NanoString int_dtype(int dsize_) {
    return (dsize_ == 3) ? ns_int64 : 
        (dsize_ == 2) ? ns_int32 :
        (dsize_ == 1) ? ns_int16 : ns_int8;
}

inline  NanoString dtype_infer(NanoString x, NanoString y) {
    int dsize_ = std::max(x.dsize_(), y.dsize_());
    bool is_float = x.is_float() || y.is_float();
    if (is_float)
        return float_dtype(dsize_);
    else {
        return int_dtype(dsize_);
    }
}

inline NanoString binary_dtype_infer(NanoString op, NanoString x, NanoString y) {
    if (op.is_bool()) return ns_bool;
    int dsize_ = std::max(x.dsize_(), y.dsize_());
    bool is_float = !op.is_int() && 
        (x.is_float() || y.is_float() || op.is_float());
    if (is_float) {
        if (op.is_white() && !(amp_reg & amp_keep_white))
            return (dsize_ == 3) ? ns_float64 : ns_float32;
        return float_dtype(dsize_);
    } else {
        if (x.is_bool() && y.is_bool()) return ns_bool;
        return int_dtype(dsize_);
    }
}

inline NanoString unary_dtype_infer(NanoString op, NanoString x) {
    if (op.is_bool()) return ns_bool;
    int dsize_ = x.dsize_();
    if (op.is_float()) {
        if (op.is_white() && !(amp_reg & amp_keep_white))
            return (dsize_ == 3) ? ns_float64 : ns_float32;
        return float_dtype(dsize_);
    }
    if (op.is_int()) return int_dtype(dsize_);
    return x;
}

inline NanoString reduce_dtype_infer(NanoString op, NanoString x) {
    bool is_float = x.is_float() || op.is_float();
    int dsize_ = x.dsize_();
    if (is_float) {
        if (amp_reg & amp_keep_reduce)
            return float_dtype(dsize_);
        return (dsize_ == 3) ? ns_float64 : ns_float32;
    } else {
        return x;
    }
}

}
