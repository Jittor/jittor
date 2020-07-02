// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <bits/stdc++.h>
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

#define map_size(T) {#T, ffs(sizeof(T))-1},
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
};

static unordered_set<string> unary_float_ops = {
    "log",
    "exp",
    "sqrt",
};
static unordered_set<string> unary_int_ops = {
    "round",
    "floor",
    "ceil",
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

#define DEFINE_NS(T) NanoString ns_##T;
FOR_ALL_NS(DEFINE_NS);

unordered_map<string, NanoString> NanoString::__string_to_ns;
vector<const char*> NanoString::__ns_to_string;

static void init_ns() {
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
            ns.set(NanoString::_int, unary_int_ops.count(name));
            ns.set(NanoString::_float, unary_float_ops.count(name));
        } else
        if (binary_ops.count(name)) {
            ns.set(NanoString::_type, NanoString::_binary, NanoString::_type_nbits);
            ns.set(NanoString::_bool, is_bool.count(name));
        }
        NanoString::__string_to_ns[name] = ns;
        NanoString::__ns_to_string.push_back(name);
    };
    #define INIT_NS(T) func(#T, ns_##T);
    FOR_ALL_NS(INIT_NS);
    ASSERT(NanoString::__ns_to_string.size()<=(1<<NanoString::_index_nbits));
    NanoString::__string_to_ns["sum"] = ns_add;
    NanoString::__string_to_ns["min"] = ns_minimum;
    NanoString::__string_to_ns["max"] = ns_maximum;
    NanoString::__string_to_ns["float"] = ns_float32;
    NanoString::__string_to_ns["double"] = ns_float64;
    NanoString::__string_to_ns["int"] = ns_int32;
    NanoString::__string_to_ns["uint"] = ns_uint32;
    LOGvv << "init __string_to_ns" << NanoString::__string_to_ns;
    LOGvv << "init __ns_to_string" << NanoString::__ns_to_string;
}

int __init_ns = (init_ns(), 0);

}
