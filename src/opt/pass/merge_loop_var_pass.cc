// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "opt/expr.h"
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/merge_loop_var_pass.h"

namespace jittor {

using namespace expr;

static unique_ptr<expr::Expr> trace_and_expand(KernelIR* ir, expr::Expr* e) {
    auto a = e->clone();
    std::function<void(expr::Expr*)> func =
    [&](expr::Expr* c) {
        if (!c->is_sym()) return;
        if (startswith(c->str, "range") && c->str.size() == 6)
            // dont expand range
            return;
        if (endswith(c->str, "outputd"))
            return;
        auto def = ir->find_define(c->str);
        if (!def) return;
        if (def->type!="define")
            return;
        if (!def->has_attr("rvalue")) return;
        auto& rvalue = def->attrs["rvalue"];
        LOGvvvv << *c << "->" << rvalue;
        if (def->father && def->flist==&def->father->inner) {
            // dont expand loop or func
            return;
        }
        c->swap(expr::make(rvalue).get());
        if (!c->children.size()) func(c);
    };
    a->dfs(func);
    return a;
}

void MergeLoopVarPass::run() {
    // LOGir << ir->to_string();
    auto choice = op->get_loop_option("merge_loop_var", 1);
    if (!choice) return;
    for (int ci=0; ci<ir->children.size(); ci++) {
        auto& c = ir->children[ci];
        if (c->type != "loop")
            continue;
        vector<KernelIR*> to_opt;
        c->dfs([&](unique_ptr<KernelIR>& i) {
            if (i->type == "loop" && i->father && i->father->type == "loop"
                && i->father->children.size() == 1 &&
                i->before.size() == 0 && i->after.size() == 0) {
                    to_opt.push_back(i.get());
                }
        });
        for (int ii=0; ii<to_opt.size(); ii++) {
            auto i = to_opt[to_opt.size()-1-ii];
            auto fa = i->father;
            LOGvvvv << "check opt" << i->attrs["rvalue"] << fa->attrs["rvalue"];
            auto range_b = i->attrs["rvalue"];
            auto id_b = i->attrs["lvalue"];
            auto range_a = fa->attrs["rvalue"];
            auto id_a = fa->attrs["lvalue"];
            if (!(i->type == "loop" && i->father && i->father->type == "loop"
                && i->father->children.size() == 1 && i->father->inner.size() == 3 &&
                i->before.size() == 0 && i->after.size() == 0)) {
                continue;
            }
            if (range_b.size() > 6) {
                // range23 -> range2*range3
                string tmp = range_b.substr(0, 6);
                for (int i=6; i<range_b.size(); i++) {
                    tmp += "*range";
                    tmp += range_b[i];
                }
                range_b = tmp;
            }
            /*
                for (id_a : range_a)
                    for (id_b : range_b)
                        match(id_a * range_b * d + id_b * d + c)
            */
            auto te = expr::make(id_a+"*"+range_b+"*d+"+id_b+"*d+c");
            vector<unique_ptr<Expr>> results;
            vector<string> solve_symbols = {"d", "c"};
            vector<string> exclude_symbols = {id_a, id_b};

            bool can_opt = true;
            i->dfs([&](unique_ptr<KernelIR>& c) {
                if (!can_opt) return;
                if (c->type == "if") {
                    // don't optimize reindex like op yet
                    can_opt = false;
                    return;
                }
                if (c->type == "define" && c->has_attr("rvalue")) {
                    auto& s = c->attrs["rvalue"];
                    auto& lv = c->attrs["lvalue"];
                    if (!(endswith(lv, "id") || endswith(lv, "_i")))
                        return;
                    auto se = expr::make(s);
                    se = trace_and_expand(c.get(), se.get())->simplify();
                    LOGvvvv << "expand" << s << "->" << se;
                    // LOGir << "expand" << s << "->" << se;
                    results.clear();
                    auto ret = expr::match(se.get(), te.get(), solve_symbols, exclude_symbols, results);
                    if (ret) {
                        LOGvvvv << "check rvalue" << se << '\n' << 
                            te << '\n' << 
                            ret << results;
                    } else {
                        can_opt = false;
                        LOGvvvv << "cannot match" << se << '\n' << 
                            te;
                    }
                }
            });
            if (!can_opt)
                continue;
            auto ni = i->clone();
            auto aid = fa->attrs["loop_id"];
            auto bid = i->attrs["loop_id"];
            auto newid = aid+bid;
            auto new_range = "range" + newid;
            auto x = i->find_define(new_range);
            if (!x) {
                ir->push_back(i->attrs["dtype"]+" "+new_range+" = "+range_b+" * "+range_a+";");
            }
            ni->replace({{"range"+bid, new_range}, {"id"+aid, "0"}}, true, true);
            ni->attrs["loop_id"] = newid;
            ni->attrs["rvalue"] = new_range;
            // simplify 0 * x -> 0
            // ni->dfs([&](unique_ptr<KernelIR>& c) {
            //     if (!can_opt) return;
            //     if (c->type == "define" && c->has_attr("rvalue")) {
            //         auto& s = c->attrs["rvalue"];
            //         auto se = expr::make(s)->simplify();
            //         s = se->to_string();
            //     }
            // });
            LOGvvvv << "new merged loop" << ni;
            ni->swap(*fa, true);
        }
    }
    ir->move_loop_back();
    ir->remove_all_unused();
    // LOGir << ir->to_string();
}

} // jittor