// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
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
        if (is_single_range_name(c->str))
            // a base range stands for itself; don't expand it into the shape
            // expression it is read from.  A merged loop's range is a product
            // of base ranges and does get expanded, because the index
            // expressions this is matched against are written in base ranges.
            // (The test used to be `size()==6`, i.e. "range" plus one
            // character, which reads range10 as a merged range.)
            return;
        if (endswith(c->str, "outputd"))
            return;
        auto def = ir->find_define(c->str);
        if (!def) return;
        if (def->type != KernelIRType::define)
            return;
        if (!def->has_attr(kir::rvalue)) return;
        auto& rvalue = def->attrs[kir::rvalue];
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
        if (c->type != KernelIRType::loop)
            continue;
        vector<KernelIR*> to_opt;
        c->dfs([&](unique_ptr<KernelIR>& i) {
            if (i->type == KernelIRType::loop && i->father && i->father->type == KernelIRType::loop
                && i->father->children.size() == 1 &&
                i->before.size() == 0 && i->after.size() == 0) {
                    to_opt.push_back(i.get());
                }
        });
        for (int ii=0; ii<to_opt.size(); ii++) {
            auto i = to_opt[to_opt.size()-1-ii];
            auto fa = i->father;
            LOGvvvv << "check opt" << i->attrs[kir::rvalue] << fa->attrs[kir::rvalue];
            auto range_b = i->attrs[kir::rvalue];
            auto id_b = i->attrs[kir::lvalue];
            auto range_a = fa->attrs[kir::rvalue];
            auto id_a = fa->attrs[kir::lvalue];
            if (!(i->type == KernelIRType::loop && i->father && i->father->type == KernelIRType::loop
                && i->father->children.size() == 1 && i->father->inner.size() == 3 &&
                i->before.size() == 0 && i->after.size() == 0)) {
                continue;
            }
            auto aid_ranges = parse_loop_id(fa->attrs[kir::loop_id]);
            auto bid_ranges = parse_loop_id(i->attrs[kir::loop_id]);
            if (!aid_ranges.size() || !bid_ranges.size())
                // not a loop we know how to name; leave it alone
                continue;
            // The template below is matched against index expressions, which
            // are written in base ranges, so the inner loop's range has to be
            // written that way too: a merged loop's range is the product of the
            // ranges it covers, a single loop's range is itself.  This used to
            // be a per-character split of the name, which turns range10 -- one
            // range -- into range1*range0.
            range_b = "range" + S(bid_ranges[0]);
            for (uint k=1; k<bid_ranges.size(); k++)
                range_b += "*range" + S(bid_ranges[k]);
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
                if (c->type == KernelIRType::branch) {
                    // don't optimize reindex like op yet
                    can_opt = false;
                    return;
                }
                if (c->type == KernelIRType::define && c->has_attr(kir::rvalue)) {
                    auto& s = c->attrs[kir::rvalue];
                    auto& lv = c->attrs[kir::lvalue];
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
            auto aid = fa->attrs[kir::loop_id];
            auto bid = i->attrs[kir::loop_id];
            auto merged_ranges = aid_ranges;
            merged_ranges.insert(merged_ranges.end(),
                bid_ranges.begin(), bid_ranges.end());
            auto newid = format_loop_id(merged_ranges);
            auto new_range = "range" + newid;
            // If this name could also be read as one range, the lookup below
            // could find somebody else's definition, skip defining ours, and
            // give the merged loop the wrong trip count -- silently, and it
            // would still compile.  format_loop_id is what rules that out.
            ASSERT(!is_single_range_name(new_range))
                << "merged loop id" << newid << "reads as a single range";
            auto x = i->find_define(new_range);
            if (!x) {
                ir->push_back(i->attrs[kir::dtype]+" "+new_range+" = "+range_b+" * "+range_a+";");
            }
            ni->replace({{"range"+bid, new_range}, {"id"+aid, "0"}}, true, true);
            ni->attrs[kir::loop_id] = newid;
            ni->attrs[kir::rvalue] = new_range;
            // simplify 0 * x -> 0
            // ni->dfs([&](unique_ptr<KernelIR>& c) {
            //     if (!can_opt) return;
            //     if (c->type == KernelIRType::define && c->has_attr(kir::rvalue)) {
            //         auto& s = c->attrs[kir::rvalue];
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