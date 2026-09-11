// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/common.h"
#include "runtime/device_state.h"
#include "utils/str_utils.h"
#include "ops/op_register.h"
#include "codegen/op_compiler.h"

namespace jittor {


unordered_map<string,string> common_op_type_cuda_map = {
    {"logical_not", "(!($2))"},
    {"bitwise_not", "(~($2))"},
    {"negative", "(-($2))"},
    {"abs", "::abs($2)"},
    {"conj", "($2)"},   // conj(real) is identity (torch parity)
    {"log", "@if(@strcmp($1,float32)==0,::logf(($1)($2)),::log(($1)($2)))"},
    {"exp", "@if(@strcmp($1,float32)==0,::expf(($1)($2)),::exp(($1)($2)))"},
    {"sqrt", "@if(@strcmp($1,float32)==0,::sqrtf(($1)($2)),::sqrt(($1)($2)))"},
    // numpy and torch round halves to even (round(0.5)==0, round(2.5)==2);
    // ::roundf rounds halves away from zero. ::rint follows the current
    // rounding mode, which is to-nearest-even and is never changed here.
    // The width is dispatched too: the old entry was the *float* spelling,
    // so a float64 input was narrowed to float before rounding and every
    // bit past the 24th was lost (round(12345678901234.5) came back as
    // 12345679020032). Anything wider than float32 goes through ::rint.
    {"round", "@if(@strcmp($1,float32)==0,(($1) ::rintf(($1)($2))),(($1) ::rint((double)($2))))"},
    {"floor", "@if(@strcmp($1,float32)==0,(($1) ::floorf(($2))),(($1) ::floor(($2))))"},
    {"ceil", "@if(@strcmp($1,float32)==0,(($1) ::ceilf(($2))),(($1) ::ceil(($2))))"},
    {"round_int", "(($1) ::roundf(($2)))"},
    {"floor_int", "(($1) ::floorf(($2)))"},
    {"ceil_int", "(($1) ::ceilf(($2)))"},
    {"sin", "@if(@strcmp($1,float32)==0,(($1) ::sinf(($2))),(($1) ::sin(($2))))"},
    {"asin", "@if(@strcmp($1,float32)==0,(($1) ::asinf(($2))),(($1) ::asin(($2))))"},
    {"sinh", "@if(@strcmp($1,float32)==0,(($1) ::sinhf(($2))),(($1) ::sinh(($2))))"},
    {"asinh", "@if(@strcmp($1,float32)==0,(($1) ::asinhf(($2))),(($1) ::asinh(($2))))"},
    {"cos", "@if(@strcmp($1,float32)==0,(($1) ::cosf(($2))),(($1) ::cos(($2))))"},
    {"acos", "@if(@strcmp($1,float32)==0,(($1) ::acosf(($2))),(($1) ::acos(($2))))"},
    {"cosh", "@if(@strcmp($1,float32)==0,(($1) ::coshf(($2))),(($1) ::cosh(($2))))"},
    {"acosh", "@if(@strcmp($1,float32)==0,(($1) ::acoshf(($2))),(($1) ::acosh(($2))))"},
    {"tan", "@if(@strcmp($1,float32)==0,(($1) ::tanf(($2))),(($1) ::tan(($2))))"},
    {"atan", "@if(@strcmp($1,float32)==0,(($1) ::atanf(($2))),(($1) ::atan(($2))))"},
    {"tanh", "@if(@strcmp($1,float32)==0,(($1) ::tanhf(($2))),(($1) ::tanh(($2))))"},
    {"atanh", "@if(@strcmp($1,float32)==0,(($1) ::atanhf(($2))),(($1) ::atanh(($2))))"},
    {"sigmoid", "(($1) (1.0f/(1.0f+::expf((::min($1(-($2)), $1(@if(@strcmp($1,float32)==0,30,300))))))))"},
    {"erf", "@if(@strcmp($1,float32)==0,(($1) ::erff(($2))),(($1) ::erf(($2))))"},
    {"erfinv", "@if(@strcmp($1,float32)==0,(($1) ::erfinvf(($1)($2))),(($1) ::erfinv(($1)($2))))"},
    {"cast", "(($1)($2))"},
    // sign-aware pow (see type/pow_compute.h): CUDA ::pow returns NaN for a
    // negative base even with an integer exponent (transformers' tanh-GELU
    // does pow(x, 3.0)); route through jittor::_signed_pow to match std::pow.
    {"pow", "(($1)jittor::_signed_pow(($2),($4)))"},
    {"maximum", "jittor::_max($1($2), $1($4))"},
    {"minimum", "jittor::_min($1($2), $1($4))"},
    {"mod", "@if(@strcmp($1,float32)==0,(($2)-::floorf(($2)/($4))*($4)),@if(@strcmp(@Tx,float64)==0,(($2)-::floor(($2)/($4))*($4)),jittor::_floor_mod($1($2), $1($4))))"},
    {"init_maximum", "::numeric_min<$1>()"},
    {"init_minimum", "::numeric_max<$1>()"},
};

struct CommonOpType : OpByType {
    CommonOpType() {
        types = {
            "bool",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float32",
            "float64",
        };
    }

    string expand_op(const vector<string>& args) {
        for (int i=1; i<args.size(); i+=2) {
            if (!types.count(args[i]))
                return "";
        }
        auto& cuda_map = common_op_type_cuda_map;

        static unordered_map<string,string> cpu_map = {
            {"logical_not", "(!($2))"},
            {"bitwise_not", "(~($2))"},
            {"negative", "(-($2))"},
            {"abs", "std::abs($2)"},
            {"conj", "($2)"},   // conj(real) is identity (torch parity)
            {"log", "std::log(($1)($2))"},
            {"exp", "std::exp(($1)($2))"},
            {"sqrt", "std::sqrt(($1)($2))"},
            // half-to-even, matching numpy/torch -- see the CUDA table.
            // std::nearbyint keeps std::round's integral overload, so an
            // integer input still widens to double instead of going
            // ambiguous between the float and double spellings.
            {"round", "(($1)std::nearbyint(($2)))"},
            {"floor", "(($1)std::floor(($2)))"},
            {"ceil", "(($1)std::ceil(($2)))"},
            {"round_int", "(($1)std::round(($2)))"},
            {"floor_int", "(($1)std::floor(($2)))"},
            {"ceil_int", "(($1)std::ceil(($2)))"},
            {"sin", "(($1) std::sin(($2)))"},
            {"asin", "(($1) std::asin(($2)))"},
            {"sinh", "(($1) std::sinh(($2)))"},
            {"asinh", "(($1) std::asinh(($2)))"},
            {"cos", "(($1) std::cos(($2)))"},
            {"acos", "(($1) std::acos(($2)))"},
            {"cosh", "(($1) std::cosh(($2)))"},
            {"acosh", "(($1) std::acosh(($2)))"},
            {"tan", "(($1) std::tan(($2)))"},
            {"atan", "(($1) std::atan(($2)))"},
            {"tanh", "(($1) std::tanh(($2)))"},
            {"atanh", "(($1) std::atanh(($2)))"},
            {"sigmoid", "(($1) (1.0f/(1.0f+std::exp(std::min($1(-($2)), $1(@if(@strcmp($1,float32)==0,30,300)))))))"},
            {"erf", "(($1) std::erf(($2)))"},
            {"erfinv", "(jittor::_erfinv($2))"},
            {"cast", "(($1)($2))"},
            {"pow", "std::pow(($2),($4))"},
            {"maximum", "jittor::_max($1($2), $1($4))"},
            {"minimum", "jittor::_min($1($2), $1($4))"},
            {"mod", "@if(@strcmp($1,float32)==0,(($2)-std::floor(($2)/($4))*($4)),@if(@strcmp(@Tx,float64)==0,(($2)-std::floor(($2)/($4))*($4)),jittor::_floor_mod($1($2), $1($4))))"},
            {"init_maximum", "std::numeric_limits<$1>::lowest()"},
            {"init_minimum", "std::numeric_limits<$1>::max()"},
        };

        static unordered_map<string,string> both_map {
            {"void", "($4)"},
            {"add", "(($2)+($4))"},
            {"subtract", "(($2)-($4))"},
            {"multiply", "(($2)*($4))"},
            {"divide", "($1(($1($2))/($1($4))))"},
            {"floor_divide", "jittor::_floor_divide($1($2), $1($4))"},
            {"less", "(($2)<($4))"},
            {"less_equal", "(($2)<=($4))"},
            {"greater", "(($2)>($4))"},
            {"greater_equal", "(($2)>=($4))"},
            {"equal", "(($2)==($4))"},
            {"not_equal", "(($2)!=($4))"},
            {"left_shift", "(($2)<<($4))"},
            {"right_shift", "(($2)>>($4))"},
            {"logical_and", "(($2)&&($4))"},
            {"logical_or", "(($2)||($4))"},
            {"logical_xor", "((bool($2))!=(bool($4)))"},
            {"bitwise_and", "(($2)&($4))"},
            {"bitwise_or", "(($2)|($4))"},
            {"bitwise_xor", "(($2)^($4))"},
            {"mean", "(($2)+$1($4)*($1(rcount)))"},
            {"init_void", "$1(0)"},
            {"init_add", "$1(0)"},
            {"init_multiply", "$1(1)"},
            {"init_logical_and", "true"},
            {"init_logical_or", "false"},
            {"init_logical_xor", "false"},
            {"init_bitwise_and", "$1(-1)"},
            {"init_bitwise_or", "$1(0)"},
            {"init_bitwise_xor", "$1(0)"},
            {"init_mean", "$1(0)"},
        };

        string ret;
        if (both_map.count(args.at(0)))
            ret = both_map[args.at(0)];
        else if (runtime_flag_use_cuda())
            ret = cuda_map[args.at(0)];
        else
            ret = cpu_map[args.at(0)];
        // `~` on a C++ bool promotes to int first, so ~true is -2 and
        // ~false is -1 -- both non-zero, and the cast below turns either
        // back into true. NumPy and Torch define bitwise_not on bool as
        // logical negation, so the bool case needs `!`, not `~`.
        if (args.at(1) == "bool" && args.at(0) == "bitwise_not")
            ret = "(!($2))";
        if (args.at(1) == "bool") ret = "((bool)"+ret+")";
        return format(ret, args);
    }

    void post_pass(OpCompiler* oc) {
        string& src = oc->src;
        string includes;
        if ((src.find("_floor_divide") != string::npos ||
             src.find("_floor_mod") != string::npos) &&
            src.find("type/floor_divide_compute.h") == string::npos)
            includes += "#include \"type/floor_divide_compute.h\"\n";
        if (src.find("_signed_pow") != string::npos &&
            src.find("type/pow_compute.h") == string::npos)
            includes += "#include \"type/pow_compute.h\"\n";
        if ((src.find("jittor::_max") != string::npos ||
             src.find("jittor::_min") != string::npos) &&
            src.find("type/minmax_compute.h") == string::npos)
            includes += "#include \"type/minmax_compute.h\"\n";
        if (includes.empty()) return;
        int i = src.rfind("#include");
        if (i<0) i=0;
        i = src.find('\n', i) + 1;
        src = src.substr(0, i) + includes + src.substr(i);
        return;
    }
};


static int _ = registe_op_type(new CommonOpType());

}
