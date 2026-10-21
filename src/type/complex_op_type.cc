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

// Native complex64 codegen. Mirrors FP16OpType: dispatches the elementwise ops on
// complex64 to the operators in type/complex_compute.h (injected by post_pass).
// Unsupported ops return "" (the op then fails loudly rather than silent-wrong).
struct ComplexOpType : OpByType {
    ComplexOpType() {
        types = {
            "complex64",
            "complex128",
        };
    }

    // This backend has a single table; the target only matters to the others.
    string expand_op(const vector<string>& args, bool /*is_cuda*/) {
        bool found = 0;
        for (int i=1; i<args.size(); i+=2)
            if (types.count(args[i])) found = 1;
        if (!found) return "";
        if (args.size() >= 4 && args[0] == "cast") {
            bool src_is_complex = args[3] == "complex64" || args[3] == "complex128";
            bool dst_is_complex = args[1] == "complex64" || args[1] == "complex128";
            if (src_is_complex && !dst_is_complex) {
                // Casting complex to real discards the imaginary component.
                // This is also the adjoint needed by real->complex cast
                // backward.
                if (args[1] == "bool")
                    return format("((($2).real != 0) || (($2).imag != 0))", args);
                return format("(($1)(jittor::jt_creal($2)))", args);
            }
            if (args[3] == "complex64" && args[1] == "complex128")
                return format("jittor::jt_c64_to_c128($2)", args);
            if (args[3] == "complex128" && args[1] == "complex64")
                return format("jittor::jt_c128_to_c64($2)", args);
            // else: real->complex (uses the target struct's single-arg
            // converting constructor) or same-width cast fall through to the
            // generic "cast" entry below.
        }
        static unordered_map<string,string> m = {
            {"void", "($4)"},
            {"add", "(($2)+($4))"},
            {"subtract", "(($2)-($4))"},
            {"multiply", "(($2)*($4))"},
            {"divide", "(($2)/($4))"},
            {"negative", "(-($2))"},
            {"abs", "jittor::jt_cabs($2)"},  // |z| -> float (Tz inferred float32); qualified
                                             // name so codegen doesn't rename it op0_jt_cabs
            {"conj", "jittor::jt_conj($2)"}, // conj(a+bi) = a-bi; qualified for same reason
            {"exp", "jittor::jt_cexp($2)"},  // complex transcendentals (qualified names so
            {"log", "jittor::jt_clog($2)"},  // codegen keeps them; complex->complex)
            {"sin", "jittor::jt_csin($2)"},
            {"cos", "jittor::jt_ccos($2)"},
            {"sqrt", "jittor::jt_csqrt($2)"},
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
        if (src.find("complex64") == string::npos && src.find("complex128") == string::npos)
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
