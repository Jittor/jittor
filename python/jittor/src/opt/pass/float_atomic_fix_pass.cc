// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
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
    bool is_cuda = op->flags.get(NodeFlags::_cuda);
    if (is_cuda) choice=1;
    if (!choice) return;

    unordered_map<string,int> fixed;
    auto fix_float_atomic = [&](string name, Var* v) {
        if (fixed.count(name)) return;
        fixed[name] = 1;
        string namep = name+"p";
        ir->dfs([&](unique_ptr<KernelIR>& i) {
            if (!i->has_attr("code")) return;
            auto& code = i->attrs["code"];
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
        if (!i->has_attr("code")) return;
        auto& code = i->attrs["code"];
        const char* m = nullptr;
        if (startswith(code, "cuda_atomic_min"))
            m = "cuda_atomic_min";
        else if (startswith(code, "cuda_atomic_max"))
            m = "cuda_atomic_max";
        if (!m) return;
        LOGvvvv << "find match" << m << i;
        vector<unique_ptr<expr::Expr>> results;
        auto target = expr::make(string(m)+"(&x[y], z)");
        auto src = expr::make(code);
        if (!expr::match(src.get(), target.get(), {"x","y","z"}, {}, results))
            return;
        LOGvvvv << "match results" << results;
        uint op_id; uint opvar_id; Op* op; Var* var;
        string s = results.at(0)->to_string();
        if (s.rbegin()[0] != 'p') return;
        s = s.substr(0, s.size()-1);
        try {
            pm->oc->get_op_var_by_name(s, op_id, opvar_id, op, var);
        } catch (...) {
            return;
        }
        if (!var->dtype().is_float()) return;
        LOGvvvv << "find var" << var << "op" << op;
        fix_float_atomic(s, var);
    });
}

} // jittor