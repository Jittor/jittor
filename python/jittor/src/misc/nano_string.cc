// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cstring>
#include "misc/nano_string.h"

namespace jittor {

#define FOR_ALL_TYPES(m) \
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
    m(float64)

#ifdef _MSC_VER
inline int ffs(int i) {
    int j=0;
    while (i) j++,i/=2;
    return j;
}
#define map_size(T) {#T, ffs(sizeof(T))-1},
#else
#define map_size(T) {#T, __builtin_ffs(sizeof(T))-1},
#endif

unordered_map<string, size_t> dsize_map = {FOR_ALL_TYPES(map_size)};

// TODO: make all static

#define map_is_float(T) {#T, std::is_floating_point<T>::value},
static unordered_map<string, bool> is_float_map = {FOR_ALL_TYPES(map_is_float)};

#define map_is_unsigned(T) {#T, std::is_unsigned<T>::value},
static unordered_map<string, bool> is_unsigned = {FOR_ALL_TYPES(map_is_unsigned)};

static unordered_set<string> is_bool = {
    "bool",
    "logical_not",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "equal",
    "not_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
};

static unordered_set<string> unary_ops = {
    "abs",
    "negative",
    "logical_not",
    "bitwise_not",
    "log",
    "exp",
    "sqrt",
    "round",
    "floor",
    "ceil",
    "round_int",
    "floor_int",
    "ceil_int",
    "cast",
    "sin",
    "asin",
    "sinh",
    "asinh",
    "tan",
    "atan",
    "tanh",
    "atanh",
    "cos",
    "acos",
    "cosh",
    "acosh",
    "sigmoid",
    "erf",
    "erfinv"
};

static unordered_set<string> float_ops = {
    "log",
    "exp",
    "sqrt",
    "mean",
    "divide",
};
static unordered_set<string> int_ops = {
    "round_int",
    "floor_int",
    "ceil_int",
    "floor_divide",
};

static unordered_set<string> binary_ops = {
    "pow",
    "maximum",
    "minimum",
    "add",
    "subtract",
    "multiply",
    "divide",
    "floor_divide",
    "mod",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "equal",
    "not_equal",
    "left_shift",
    "right_shift",
    "logical_and",
    "logical_or",
    "logical_xor",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "mean",
};


static unordered_set<string> white_ops = {
    // "log",
    "exp",
    "pow",
};

static unordered_set<string> no_need_back_in = {
    "void",
    "cast",
    "negative",
    "add",
    "subtract",
    "mean",
};

static unordered_set<string> no_need_back_out = {
    "void",
    "cast",
    "negative",
    "add",
    "subtract",
    "multiply",
};

#define DEFINE_NS(T) NanoString ns_##T;
FOR_ALL_NS(DEFINE_NS);

unordered_map<string, NanoString> __string_to_ns;
char __ns_to_string[ns_max_size*ns_max_len];
int __ns_len[ns_max_size];

static void init_ns() {
    dsize_map["float16"] = 1;
    is_float_map["float16"] = 1;
    is_unsigned["float16"] = 0;
    NanoString::ns_t i=0;
    auto func = [&](const char* name, NanoString& ns) {
        ns.set(NanoString::_index, i++, NanoString::_index_nbits);
        if (dsize_map.count(name)) {
            ns.set(NanoString::_type, NanoString::_dtype, NanoString::_type_nbits);
            ns.set(NanoString::_bool, is_bool.count(name));
            ns.set(NanoString::_int, !is_float_map.at(name));
            ns.set(NanoString::_unsigned, is_unsigned.count(name));
            ns.set(NanoString::_float, is_float_map.at(name));
            ns.set(NanoString::_dsize, dsize_map.at(name), NanoString::_dsize_nbits);
        } else
        if (unary_ops.count(name)) {
            ns.set(NanoString::_type, NanoString::_unary, NanoString::_type_nbits);
            ns.set(NanoString::_bool, is_bool.count(name));
            ns.set(NanoString::_int, int_ops.count(name));
            ns.set(NanoString::_float, float_ops.count(name));
        } else
        if (binary_ops.count(name)) {
            ns.set(NanoString::_type, NanoString::_binary, NanoString::_type_nbits);
            ns.set(NanoString::_bool, is_bool.count(name));
            ns.set(NanoString::_int, int_ops.count(name));
            ns.set(NanoString::_float, float_ops.count(name));
        }
        ns.set(NanoString::_white_list, white_ops.count(name));
        ns.set(NanoString::_no_need_back_in, no_need_back_in.count(name));
        ns.set(NanoString::_no_need_back_out, no_need_back_out.count(name));
        __string_to_ns[name] = ns;
        auto name2 = ns.to_cstring();
        int len=0;
        for (;;len++) {
            name2[len] = name[len];
            if (!name[len]) break;
        }
        __ns_len[i-1] = len;
    };
    #define INIT_NS(T) func(#T, ns_##T);
    FOR_ALL_NS(INIT_NS);
    ASSERT(i<=(1<<NanoString::_index_nbits));
    __string_to_ns["sum"] = ns_add;
    __string_to_ns["min"] = ns_minimum;
    __string_to_ns["max"] = ns_maximum;
    __string_to_ns["half"] = ns_float16;
    __string_to_ns["float"] = ns_float32;
    __string_to_ns["double"] = ns_float64;
    __string_to_ns["int"] = ns_int32;
    __string_to_ns["uint"] = ns_uint32;
    LOGvv << "init __string_to_ns" << __string_to_ns;
    LOGvv << "init __ns_to_string" << __ns_to_string;
}

int __init_ns = (init_ns(), 0);

}
