// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "common.h"
#include "utils/str_utils.h"
#include "ops/op_register.h"
#include "op_compiler.h"

namespace jittor {

// Native complex64 codegen. Mirrors FP16OpType: dispatches the elementwise ops on
// complex64 to the operators in type/complex_compute.h (injected by post_pass).
// Unsupported ops return "" (the op then fails loudly rather than silent-wrong).
struct ComplexOpType : OpByType {
    ComplexOpType() {
        types = {
            "complex64",
        };
    }

    string expand_op(const vector<string>& args) {
        bool found = 0;
        for (int i=1; i<args.size(); i+=2)
            if (types.count(args[i])) found = 1;
        if (!found) return "";
        static unordered_map<string,string> m = {
            {"void", "($4)"},
            {"add", "(($2)+($4))"},
            {"subtract", "(($2)-($4))"},
            {"multiply", "(($2)*($4))"},
            {"divide", "(($2)/($4))"},
            {"negative", "(-($2))"},
            {"cast", "(($1)($2))"},
            {"equal", "(($2)==($4))"},
            {"not_equal", "(($2)!=($4))"},
            {"mean", "(($2)+($4)*(($1)(rcount)))"},
            {"init_void", "($1)(0)"},
            {"init_add", "($1)(0)"},
            {"init_multiply", "($1)(1)"},
            {"init_mean", "($1)(0)"},
        };
        if (!m.count(args.at(0)))
            return "";
        return format(m[args.at(0)], args);
    }

    void post_pass(OpCompiler* oc) {
        string& src = oc->src;
        if (src.find("complex64") == string::npos)
            return;
        int i = src.rfind("#include");
        if (i<0) i=0;
        i = src.find('\n', i) + 1;
        src = src.substr(0, i) + "#include \"type/complex_compute.h\"\n" + src.substr(i);
        return;
    }
};

static int _ = registe_op_type(new ComplexOpType());

}
