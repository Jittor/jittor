// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "opt/expr.h"
#include "opt/pass_manager.h"
#include "opt/pass/float_atomic_fix_pass.h"
#include "utils/str_utils.h"

namespace jittor {

void FloatAtomicFixPass::run() {
    auto choice = op->get_loop_option("parallel");
    bool is_cuda = op->flag(OpFlags::_cuda);
    if (is_cuda) choice=1;
    if (!choice) return;

    unordered_map<string,int> fixed;
    auto fix_float_atomic = [&](string name, Var* v) {
        if (fixed.count(name)) return;
        fixed[name] = 1;
        string namep = name+"p";
        ir->dfs([&](unique_ptr<KernelIR>& i) {
            if (!i->has_attr(kir::code)) return;
            auto& code = i->attrs[kir::code];
            if (!startswith(code, namep)) return;
            LOGvvvv << "find code" << code;
            auto src = expr::make(code);
            auto target = expr::make(namep+"[b]=c");
            vector<unique_ptr<expr::Expr>> results;
            if (!expr::match(src.get(), target.get(), {"b","c"}, {}, results))
                return;
            // fix code a[b] = c -->
            // a[b] = __int_as_float(floatToOrderedInt(c))
            string new_code;
            if (v->dtype() == ns_float32)
                new_code = namep+'['+results.at(0)->to_string(true)+
                    "] = __int_as_float(floatToOrderedInt(" +
                    results.at(1)->to_string(true) + "));";
            else
                new_code = namep+'['+results.at(0)->to_string(true)+
                    "] = __longlong_as_double(floatToOrderedInt(" +
                    results.at(1)->to_string(true) + "));";
            LOGvvvv << "prev code" << code >> "\nreplace:" << new_code;
            code = new_code;
        });
        ir->push_back("fix_float("+namep+", "+name+"->num);");
    };

    ir->dfs([&](unique_ptr<KernelIR>& i) {
        if (!i->has_attr(kir::code)) return;
        auto& code = i->attrs[kir::code];
        // The exact call, not the prefix: cuda_atomic_max_rmw is a different
        // function (a raw-IEEE CAS loop, used by setitem, which has no
        // ordered-int pass) and this must not claim it.
        const char* m = nullptr;
        if (startswith(code, "cuda_atomic_min("))
            m = "cuda_atomic_min";
        else if (startswith(code, "cuda_atomic_max("))
            m = "cuda_atomic_max";
        if (!m) return;
        LOGvvvv << "find match" << m << i;
        // Everything below has to succeed. cuda_atomic_max(float*) is
        // atomicMax over the buffer reinterpreted as ordered ints, so it is
        // only correct if this pass also rewrites the initialisation into that
        // representation and appends fix_float() to convert back. Declining a
        // statement it does not understand leaves an integer atomicMax running
        // over raw float bit patterns, never converted back: wrong numbers, no
        // diagnostic. That is what these steps used to do -- two silent
        // returns and a catch(...).
        // LOGf throws, so cannot() does not return -- the statements after
        // each call are unreachable, not a fallthrough.
        auto cannot = [&](const string& why) {
            LOGf << "Jit error: FloatAtomicFixPass cannot identify the buffer of"
                << why >> ":\n    " >> code >>
                "\nThis pass has to convert that buffer to the ordered-int form"
                " cuda_atomic_max/min need, and it cannot do that without"
                " knowing which var it is. Leaving the statement alone would"
                " run an integer atomicMax over raw float bits and never"
                " convert them back.";
        };
        vector<unique_ptr<expr::Expr>> results;
        auto target = expr::make(string(m)+"(&x[y], z)");
        auto src = expr::make(code);
        if (!expr::match(src.get(), target.get(), {"x","y","z"}, {}, results))
            cannot("a call this pass does not recognise the shape of");
        LOGvvvv << "match results" << results;
        uint op_id; uint opvar_id; Op* op; Var* var;
        string s = results.at(0)->to_string();
        if (s.rbegin()[0] != 'p')
            cannot("a target that is not an op var pointer (a pass that renamed"
                   " it, such as restride, has to run after this one)");
        s = s.substr(0, s.size()-1);
        if (!pm->oc->try_get_op_var_by_name(s, op_id, opvar_id, op, var))
            cannot("an unknown name " + s);
        // an integer atomicMax is already correct on integers
        if (!var->dtype().is_float()) return;
        if (var->dtype() == ns_float16 || var->dtype() == ns_bfloat16)
            // float16 use atomicCAS, because no float16 atomicMax
            return;
        LOGvvvv << "find var" << var << "op" << op;
        fix_float_atomic(s, var);
    });
}

} // jittor