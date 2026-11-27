// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/common.h"
#include "utils/str_utils.h"
#include "ops/op_register.h"
#include "codegen/op_compiler.h"

namespace jittor {

// Native FP8 (float8_e4m3fn, float8_e5m2, float8_e8m0fnu) codegen.
// Mirrors ComplexOpType / FP16OpType: dispatches elementwise ops via float32
// widening in type/fp8_compute.h (injected by post_pass).
struct FP8OpType : OpByType {
    FP8OpType() {
        types = {
            "float8_e4m3fn",
            "float8_e5m2",
            "float8_e8m0fnu",
        };
    }

    string expand_op(const vector<string>& args, bool /*is_cuda*/) {
        bool found = false;
        for (size_t i = 1; i < args.size(); i += 2)
            if (types.count(args[i])) found = true;
        if (!found) return "";
        const string& op = args.at(0);
        if (op == "cast") {
            // Route through float32 so cross-FP8 and FP8<->FP16/BF16 casts
            // never depend on direct pairwise constructors.
            if (args.size() >= 4 && args[1] == args[3])
                return format("($2)", args);
            if (args.size() >= 4 && args[1] == "bool")
                return format("(float32($2) != 0.0f)", args);
            return format("(($1)(float32($2)))", args);
        }
        static unordered_map<string, string> m = {
            {"void", "($4)"},
            {"add", "(($1)(float32($2) + float32($4)))"},
            {"subtract", "(($1)(float32($2) - float32($4)))"},
            {"multiply", "(($1)(float32($2) * float32($4)))"},
            {"divide", "(($1)(float32($2) / float32($4)))"},
            {"maximum", "(($1)(std::isnan(float32($2)) ? float32($2) : std::isnan(float32($4)) ? float32($4) : (float32($2) >= float32($4) ? float32($2) : float32($4))))"},
            {"minimum", "(($1)(std::isnan(float32($2)) ? float32($2) : std::isnan(float32($4)) ? float32($4) : (float32($2) <= float32($4) ? float32($2) : float32($4))))"},
            {"negative", "(($1)(-float32($2)))"},
            {"abs", "(($1)(std::fabs(float32($2))))"},
            {"less", "(float32($2) < float32($4))"},
            {"less_equal", "(float32($2) <= float32($4))"},
            {"greater", "(float32($2) > float32($4))"},
            {"greater_equal", "(float32($2) >= float32($4))"},
            {"equal", "(float32($2) == float32($4))"},
            {"not_equal", "(float32($2) != float32($4))"},
            {"isnan", "(std::isnan(float32($2)))"},
            {"isinf", "(std::isinf(float32($2)))"},
            {"isfinite", "(std::isfinite(float32($2)))"},
            {"exp", "(($1)(std::exp(float32($2))))"},
            {"log", "(($1)(std::log(float32($2))))"},
            {"sqrt", "(($1)(std::sqrt(float32($2))))"},
            {"mean", "(($1)(float32($2) + float32($4) * float32(rcount)))"},
            {"init_void", "($1)(0.0f)"},
            {"init_add", "($1)(0.0f)"},
            {"init_multiply", "($1)(1.0f)"},
            {"init_mean", "($1)(0.0f)"},
            {"init_maximum", "($1)(-INFINITY)"},
            {"init_minimum", "($1)(INFINITY)"},
        };
        if (!m.count(op))
            return "";
        return format(m[op], args);
    }

    void post_pass(OpCompiler* oc) {
        string& src = oc->src;
        if (src.find("float8_e4m3fn") == string::npos &&
            src.find("float8_e5m2") == string::npos &&
            src.find("float8_e8m0fnu") == string::npos)
            return;
        int i = src.rfind("#include");
        if (i < 0) i = 0;
        i = src.find('\n', i) + 1;
        src = src.substr(0, i) + "#include \"type/fp8_compute.h\"\n" + src.substr(i);
    }
};

static int _ = registe_op_type(new FP8OpType());

} // namespace jittor
