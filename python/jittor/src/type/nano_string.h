// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

//: How many NanoString entries exist, and how many bytes each name gets.
//:
//: Both are hard limits on a table that is written by index with no bounds
//: check of its own, so they are asserted at registration time
//: (`ns_check_registration`) rather than trusted. `ns_max_size` must stay equal
//: to `1 << NanoString::_index_nbits`; a static_assert below holds the two
//: together, because they disagreed for years -- the table had room for 256
//: entries and the index field could only address 128 of them.
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
    m(bfloat16) \
    m(complex64) \
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
    m(conj) \
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
        //
        // 8 bits, not 7. `set()` masks the value it is given, so an index the
        // field cannot hold does not overflow into the next field -- it wraps,
        // and entry 128 silently becomes an alias of entry 0. The table it
        // indexes (`__ns_to_string`, `__ns_len`) has always been 256 entries
        // long, so the two disagreed by a factor of two with 71 of the 128
        // slots already spent. See `ns_check_registration`.
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
        // bit8: white list
        _white_list=_n+8,
        // bit9: backward opt
        _no_need_back_in=_n+9,
        _no_need_back_out=_n+10,
        // bit11: is complex (real/imag pair; _float/_int both 0)
        _complex=_n+11,
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
    // @pyjt(is_floating_point)
    inline bool is_floating_point() const { return get(_float); }
    // @pyjt(is_complex)
    inline bool is_complex() const { return get(_complex); }
    // @pyjt(is_float)
    inline bool is_float() const { return get(_float); }
    inline ns_t is_white() const { return get(_white_list); }
    // @pyjt(dsize)
    inline int dsize() const { return 1<<get(_dsize, _dsize_nbits); }
    inline ns_t dsize_() const { return get(_dsize, _dsize_nbits); }
    inline ns_t is_dtype() const { return get(_type, _type_nbits)==_dtype; }
    inline ns_t is_binary() const { return get(_type, _type_nbits)==_binary; }
    inline ns_t is_unary() const { return get(_type, _type_nbits)==_unary; }

    inline NanoString() {}
    // @pyjt(__init__)
    inline NanoString(const char* s) {
        auto iter = __string_to_ns.find(s);
        if (iter == __string_to_ns.end() && s &&
            s[0]=='t'&&s[1]=='o'&&s[2]=='r'&&s[3]=='c'&&s[4]=='h'&&s[5]=='.') {
            // Tolerate torch-style dtype names that the torch-compat shim
            // (`import jittor as torch`) leaks into jittor internals via
            // str(Var.dtype), e.g. "torch.bfloat16" -> "bfloat16". Only a
            // subset resolved before; strip the prefix uniformly.
            iter = __string_to_ns.find(s+6);
        }
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
    operator uint32() const { return data; }
};

static_assert(ns_max_size == (1 << NanoString::_index_nbits),
    "the name table and the index field must address the same number of "
    "entries: a table larger than the field wraps silently, a field larger "
    "than the table writes past it");

/** Reject a NanoString registration that the tables cannot hold.
 *
 * Both limits used to be unchecked at the point where they are exceeded. An
 * index past the end of the field wrapped (see `_index_nbits`); a name of
 * `ns_max_len` characters or more ran off the end of its 16-byte slot and
 * overwrote the *next* entry's name, which reads as "some unrelated operator
 * is suddenly called something else". The longest name in the table today is
 * 13 characters, so the second one is two characters of headroom away, and
 * whoever spends it will be adding an operator or a dtype and looking
 * somewhere else entirely.
 *
 * Throws (via ASSERT) rather than returning a code: it runs during static
 * initialisation, where there is nobody to return to.
 */
void ns_check_registration(uint32 index, const char* name);

/** Does `s` name a NanoString -- a dtype or an operator name?
 *
 * Mirrors the `NanoString(const char*)` constructor above, including its
 * tolerance for the torch-compat shim's "torch.<dtype>" spelling.  It exists so
 * that `is_type<NanoString>` in pyjt/py_converter.h can ask the same question
 * the conversion will answer: when the two disagreed, the type check claimed a
 * NanoString overload for something that was not one, and the failure surfaced
 * from inside the conversion as an operator error instead of a bad argument.
 */
inline bool ns_valid_name(const char* s) {
    if (!s) return false;
    if (__string_to_ns.count(s)) return true;
    if (s[0]=='t'&&s[1]=='o'&&s[2]=='r'&&s[3]=='c'&&s[4]=='h'&&s[5]=='.')
        return __string_to_ns.count(s+6) != 0;
    return false;
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

EXTERN_LIB int amp_reg;
constexpr int amp_prefer32 = 1;
constexpr int amp_prefer16 = 2;
constexpr int amp_keep_reduce = 4;
constexpr int amp_keep_white = 8;
constexpr int amp_array_prefer = 16;

inline NanoString float_dtype(int dsize_, bool has_scalar=false, bool has_bf16=false) {
    if (!has_scalar) {
        if (amp_reg & amp_prefer32)
            return ns_float32;
        if (amp_reg & amp_prefer16)
            return has_bf16 ? ns_bfloat16 : ns_float16;
    } 
    return (dsize_ == 3) ? ns_float64 : 
        (dsize_ == 2 ) ? ns_float32 : 
        has_bf16 ? ns_bfloat16 : ns_float16;
}

inline NanoString int_dtype(int dsize_, bool is_unsigned=false) {
    if (is_unsigned) {
        return (dsize_ == 3) ? ns_uint64 :
            (dsize_ == 2) ? ns_uint32 :
            (dsize_ == 1) ? ns_uint16 : ns_uint8;
    }
    return (dsize_ == 3) ? ns_int64 :
        (dsize_ == 2) ? ns_int32 :
        (dsize_ == 1) ? ns_int16 : ns_int8;
}

//: `dsize_` of the widest integer that exists: 3, i.e. 2^3 = 8 bytes.
constexpr int ns_max_int_dsize = 3;

/** Promote two integer dtypes the way NumPy and Torch do.
 *
 * The rule this replaces was "widest byte count wins, and the result is
 * unsigned only if both operands are": not a lattice, and it drops the sign of
 * the signed operand whenever the unsigned one is at least as wide. So
 * `uint8 + int8` came out `int8`, and `uint8(200) + int8(1)` -- which needs 201
 * -- came out **-55**, with nothing said. `uint32 + int32` came out `int32`,
 * losing the top half of the uint32 range the same way.
 *
 * The lattice: within one signedness the wider type wins. Across signedness the
 * result must hold every value of *both*, and a signed type only covers an
 * unsigned one that is strictly narrower -- so it takes one extra doubling
 * beyond the unsigned operand. When that runs past int64 (any signed type mixed
 * with uint64) no integer type is wide enough and the common type is float64,
 * which is what NumPy does too.
 *
 * bool is a kind, not a width: it does not force a widening, it adopts the
 * other operand's type. (`is_unsigned` is true for bool -- `std::is_unsigned`
 * says so -- which is exactly why it has to be handled before the signedness
 * test rather than falling into it.)
 */
inline NanoString int_dtype_promote(NanoString x, NanoString y) {
    if (x.is_bool()) return y;
    if (y.is_bool()) return x;
    bool xu = x.is_unsigned(), yu = y.is_unsigned();
    if (xu == yu)
        return int_dtype(std::max(x.dsize_(), y.dsize_()), xu);
    int unsigned_size = (int)(xu ? x.dsize_() : y.dsize_());
    int signed_size = (int)(xu ? y.dsize_() : x.dsize_());
    int size = std::max(signed_size, unsigned_size + 1);
    if (size > ns_max_int_dsize) return ns_float64;
    return int_dtype(size, false);
}

/** Does `op` refuse floating-point operands outright?
 *
 * The uint64 fallback above would hand these a float64 output, and the
 * generated kernel (`x & y`, `x << y`) does not compile for doubles. They keep
 * the older widest-type answer instead: there is no float meaning to fall back
 * to, and `BinaryOp` already rejects float operands for them with a message.
 */
inline bool ns_is_integral_only(NanoString op) {
    return op==ns_bitwise_and || op==ns_bitwise_or || op==ns_bitwise_xor ||
        op==ns_left_shift || op==ns_right_shift;
}

/** The integer half of binary promotion, scalars included.
 *
 * A Python scalar is a "wrapped number": it never widens the tensor, it adopts
 * the tensor's dtype. That is what the callers' `xscalar`/`yscalar` flags mean,
 * and it is why the promotion lattice is only consulted when both sides are
 * real operands.
 */
inline NanoString int_binary_dtype(NanoString op, NanoString x, NanoString y,
                                   bool xscalar, bool yscalar) {
    // Exactly one scalar: the tensor decides. Two scalars: neither is "the
    // tensor", so fall through to the lattice -- overriding on one of them
    // would make the answer depend on which side it was written, and `a+b`
    // must agree with `b+a`.
    if (xscalar && !yscalar) return int_dtype(y.dsize_(), y.is_unsigned());
    if (yscalar && !xscalar) return int_dtype(x.dsize_(), x.is_unsigned());
    auto promoted = int_dtype_promote(x, y);
    if (promoted.is_float() && ns_is_integral_only(op))
        return int_dtype(std::max(x.dsize_(), y.dsize_()),
                         x.is_unsigned() && y.is_unsigned());
    return promoted;
}

/** The float half: what a python float scalar does to an integer operand.
 *
 * `float_dtype(dsize_, ...)` picks the float type by *byte width*, and with a
 * scalar involved `dsize_` is the tensor's. For a float tensor that is right --
 * float16 stays float16. For an *integer* tensor it is not: the scalar carries
 * no width to promote to, so `uint8 * (1/255.)` became float16 (one byte ->
 * float16) and `int64 * 2.0` became float64. Both are the default float dtype
 * in Torch, which is what a wrapped number lifts an integer to.
 */
inline bool ns_scalar_lifts_int_to_default_float(
        NanoString x, NanoString y, bool xscalar, bool yscalar) {
    // The scalar must itself be a float -- that is what lifts the category --
    // and the tensor must be an integer or a bool, i.e. carry no float width
    // for the result to inherit. `int64 / 2` is deliberately not this case:
    // there the float comes from the *operator*, and jittor follows numpy in
    // giving it float64.
    if (xscalar && !yscalar)
        return x.is_float() && !y.is_float() && !y.is_complex();
    if (yscalar && !xscalar)
        return y.is_float() && !x.is_float() && !x.is_complex();
    return false;
}

inline  NanoString dtype_infer(NanoString x, NanoString y, bool xscalar=false, bool yscalar=false) {
    if (x.is_complex() || y.is_complex()) return ns_complex64;  // complex propagates
    if (x.is_bool() && y.is_bool()) return ns_bool;
    int dsize_ = std::max(x.dsize_(), y.dsize_());
    if (xscalar) dsize_ = y.dsize_();
    if (yscalar) dsize_ = x.dsize_();
    bool is_float = x.is_float() || y.is_float();
    bool has_bf16 = x==ns_bfloat16 || y==ns_bfloat16;
    if (is_float) {
        if (ns_scalar_lifts_int_to_default_float(x, y, xscalar, yscalar))
            return ns_float32;
        return float_dtype(dsize_, xscalar||yscalar, has_bf16);
    }
    return int_binary_dtype(ns_add, x, y, xscalar, yscalar);
}

// @pyjt(binary_dtype_infer)
inline NanoString binary_dtype_infer(NanoString op, NanoString x, NanoString y, bool xscalar=false, bool yscalar=false) {
    if (op.is_bool()) return ns_bool;   // comparisons -> bool even for complex
    if (x.is_complex() || y.is_complex()) return ns_complex64;  // complex arithmetic
    int dsize_ = std::max(x.dsize_(), y.dsize_());
    if (xscalar) dsize_ = y.dsize_();
    if (yscalar) dsize_ = x.dsize_();
    bool is_float = !op.is_int() && 
        (x.is_float() || y.is_float() || op.is_float());
    bool has_bf16 = x==ns_bfloat16 || y==ns_bfloat16;
    if (is_float) {
        if (op.is_white() && !(amp_reg & amp_keep_white))
            return (dsize_ == 3) ? ns_float64 : ns_float32;
        if (ns_scalar_lifts_int_to_default_float(x, y, xscalar, yscalar))
            return ns_float32;
        return float_dtype(dsize_, xscalar||yscalar, has_bf16);
    } else {
        if (x.is_bool() && y.is_bool()) return ns_bool;
        return int_binary_dtype(op, x, y, xscalar, yscalar);
    }
}

inline NanoString unary_dtype_infer(NanoString op, NanoString x) {
    if (op.is_bool()) return ns_bool;
    if (x.is_complex()) return (op==ns_abs) ? ns_float32 : ns_complex64;  // |z|->float
    int dsize_ = x.dsize_();
    if (op.is_float()) {
        if (op.is_white() && !(amp_reg & amp_keep_white))
            return (dsize_ == 3) ? ns_float64 : ns_float32;
        return float_dtype(dsize_, false, x==ns_bfloat16);
    }
    if (op.is_int()) return int_dtype(dsize_, x.is_unsigned());
    return x;
}

inline NanoString reduce_dtype_infer(NanoString op, NanoString x) {
    // complex reductions stay complex (sum/mean/prod). Without this, mean -- which is in
    // float_ops -- forces a float output dtype, so the kernel tries to assign a complex64
    // accumulator into a double and fails to compile. (sum works already because 'add' is
    // not a float_op.)
    if (x.is_complex()) return ns_complex64;
    bool is_float = x.is_float() || op.is_float();
    int dsize_ = x.dsize_();
    if (is_float) {
        if (amp_reg & amp_keep_reduce)
            return float_dtype(dsize_, false, x==ns_bfloat16);
        return (dsize_ == 3) ? ns_float64 : ns_float32;
    } else {
        return x;
    }
}

}
