// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "common.h"
#include "utils/str_utils.h"
#include "ops/op_register.h"
#include "op_compiler.h"

namespace jittor {

extern int use_cuda;

extern unordered_map<string,string> common_op_type_cuda_map;

static bool isvar(char x) { return isalnum(x) || x == '_' || x == ':'; }

struct FP16OpType : OpByType {
    FP16OpType() {
        types = {
            "float16",
        };
    }

    string expand_op(const vector<string>& args) {
        bool found_fp16 = 0;
        for (int i=1; i<args.size(); i+=2) {
            if (types.count(args[i]))
                found_fp16 = 1;
        }
        if (!found_fp16) return "";
        static unordered_map<string,string> cuda_map = {
            {"logical_not", "(!($2))"},
            {"bitwise_not", "(~($2))"},
            {"negative", "(-($2))"},
            {"abs", "::abs($2)"},
            {"log", "::hlog(($1)($2))"},
            {"exp", "::hexp(($1)($2))"},
            {"sqrt", "::hsqrt(($1)($2))"},
            {"round", "(($1) ::roundf(($2)))"},
            {"floor", "(($1) ::floorf(($2)))"},
            {"ceil", "(($1) ::ceilf(($2)))"},
            {"round_int", "(($1) ::roundf(($2)))"},
            {"floor_int", "(($1) ::floorf(($2)))"},
            {"ceil_int", "(($1) ::ceilf(($2)))"},
            {"sin", "(($1) ::sinf(($2)))"},
            {"asin", "(($1) ::asinf(($2)))"},
            {"sinh", "(($1) ::sinhf(($2)))"},
            {"asinh", "(($1) ::asinhf(($2)))"},
            {"cos", "(($1) ::cosf(($2)))"},
            {"acos", "(($1) ::acosf(($2)))"},
            {"cosh", "(($1) ::coshf(($2)))"},
            {"acosh", "(($1) ::acoshf(($2)))"},
            {"tan", "(($1) ::tanf(($2)))"},
            {"atan", "(($1) ::atanf(($2)))"},
            {"tanh", "(($1) ::tanhf(($2)))"},
            {"atanh", "(($1) ::atanhf(($2)))"},
            {"sigmoid", "(($1) (1.0f/(1.0f+::expf((::min($1(-($2)), $1(@if(@strcmp($1,float16)==0,30,300))))))))"},
            {"erf", "(($1) ::erff(($2)))"},
            {"erfinv", "(($1) ::erfinvf(($1)($2)))"},
            {"cast", "(($1)($2))"},
            {"pow", "::pow(($2),($4))"},
            {"maximum", "::max($1($2), $1($4))"},
            {"minimum", "::min($1($2), $1($4))"},
            {"mod", "$1(($2)-::hfloor(($2)/($4))*($4))"},
            {"init_maximum", "-32768.0f"},
            {"init_minimum", "32768.0f"},
        };

        static unordered_map<string,string> cpu_map = {
            {"logical_not", "(!($2))"},
            {"bitwise_not", "(~($2))"},
            {"negative", "(-($2))"},
            {"abs", "std::abs($2)"},
            {"log", "std::log(($1)($2))"},
            {"exp", "std::exp(($1)($2))"},
            {"sqrt", "std::sqrt(($1)($2))"},
            {"round", "(($1)std::round(($2)))"},
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
            {"maximum", "std::max($1($2), $1($4))"},
            {"minimum", "std::min($1($2), $1($4))"},
            {"mod", "$1(($2)-std::floor(($2)/($4))*($4))"},
            {"init_maximum", "-32768.0f"},
            {"init_minimum", "32768.0f"},
        };

        static unordered_map<string,string> both_map {
            {"add", "(($2)+($4))"},
            {"subtract", "(($2)-($4))"},
            {"multiply", "(($2)*($4))"},
            {"divide", "($1(($1($2))/($1($4))))"},
            {"floor_divide", "($1(($1($2))/($1($4))))"},
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
            {"mean", "(($2)+($4)*($1(rcount)))"},
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
        else if (use_cuda)
            ret = cuda_map[args.at(0)];
        else
            ret = cpu_map[args.at(0)];
        if (use_cuda) {
            if (args[1] == "float32" && !both_map.count(args.at(0))) {
                ret = common_op_type_cuda_map[args.at(0)];
            }
            if (args[1] == "float16" || args[1] == "float32") {
                for (int i=3; i<args.size(); i+=2) {
                    if (args[i] != args[1]) {
                        ret = replace(ret, "$"+S(i-1),
                            args[1]+"($"+S(i-1)+")");
                    }
                }
            } else {
                for (int i=3; i<args.size(); i+=2) {
                    if (args[i] != "float16") {
                        ret = replace(ret, "$"+S(i-1),
                            "float16($"+S(i-1)+")");
                    }
                }
            }
        }
        return format(ret, args);
    }

    void post_pass(OpCompiler* oc) {
        string& src = oc->src;
        if (src.find("float16") == string::npos)
            return;
        int i = src.rfind("#include");
        if (i<0) i=0;
        i = src.find('\n', i) + 1;
        src = src.substr(0, i) + "#include \"type/fp16_compute.h\"\n" + 
            src.substr(i);
        return;
    }
};


static int _ = registe_op_type(new FP16OpType());

}