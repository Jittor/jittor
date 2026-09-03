// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers:
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>.
//
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include <omp.h>
#include "var.h"
#include "opt/expr.h"
#include "opt/pass_manager.h"
#include "opt/pass/atomic_tuner_pass.h"
#include "opt/pass/loop_var_analyze_pass.h"

namespace jittor {

extern int para_opt_level;

// Move `def` -- and, transitively, every definition in `scope` that `def`'s
// rvalue reads -- to the front of dst->before. Used to hoist the address
// computation of an atomic out of the loop that the accumulation was lifted
// past, so that the address is still in scope where the flush is emitted.
static void move_definition_chain(KernelIR* scope, KernelIR* dst, KernelIR* def) {
    vector<KernelIR*> pending{def};
    map<KernelIR*, int> moved;
    moved[def] = 1;
    dst->push_front(def->move_out(), &dst->before);
    // `pending` grows inside the loop, so index rather than iterate
    for (uint i = 0; i < pending.size(); i++) {
        auto rvalue = expr::make(pending[i]->attrs[kir::rvalue]);
        LOGvvvv << "move_rely" << rvalue->to_string();
        rvalue->dfs([&](expr::Expr* sym) {
            if (!sym->is_sym()) return;
            auto ir = scope->find_define(sym->str);
            if (ir == nullptr) return;
            if (!ir->father) return;
            // only definitions owned directly by `scope` move; anything
            // defined further out is already visible at the destination
            if (ir->father != scope) return;
            if (!moved.count(ir)) {
                dst->push_front(ir->move_out(), &dst->before);
                pending.push_back(ir);
                moved[ir] = 1;
            }
        });
    }
}

// Source of the identity element for `op`'s reduction, in `var`'s dtype, e.g.
// "0" for add or "-inf" for max. `op->name_ex()` looks like "reduce.add".
string get_reduce_init_code(Op* op, Var* var, bool is_cuda) {
    auto name_parts = split(op->name_ex(), ".");
    ASSERT(name_parts.size() == 2) << name_parts;
    auto code = OpCompiler::precompile(
        {
            {"OP", name_parts.back()},
            {"T", var->dtype().to_cstring()},
            {is_cuda ? "JIT_cuda" : "JIT_cpu", "1"},
        },
        "@expand_op(init_@OP, @T)");
    return code;
}

// Lift atomics out of the innermost loops of one kernel function.
//
// The input is a perfectly nested loop whose body contains one or more
// `atomicAdd(&op0_xp[i], v)`-style statements, one per output. Each such
// statement carries a "rely" attribute listing the loop variables its
// destination address depends on. Loops the address does *not* depend on can
// be moved inward, and the atomic can then be replaced by
//
//     T tmp0 = <identity>;        // before the innermost dependent loop
//     ... tmp0 = tmp0 + v; ...    // plain accumulation inside it
//     atomicAdd(&op0_xp[i], tmp0) // after it
//
// which turns one atomic per iteration into one atomic per output address.
//
// Outputs, consumed by AtomicTunerPass::run to reorder the matching thread
// range declarations at the call site:
//   loop_orders     -- for each rewritten kernel, the new loop order
//   loop_func_names -- the name of the kernel function it belongs to
static void tune_atomic(Pass* pass, KernelIR* func, bool is_cuda,
                        vector<vector<int>>& loop_orders, vector<string>& loop_func_names) {
    LOGvvvv << "tune_atomic" << func->children;
    // loop_lvalues[i]: induction variable of the i-th loop of the nest,
    // counted from the outside in, in the *original* order
    vector<string> loop_lvalues;
    vector<KernelIR*> atomic_stmts;
    vector<KernelIR*> loops;
    // order[p] is the original loop index that should end up at nesting
    // position p; order_owner[p] records which atomic statement demanded it
    vector<int> order;
    vector<int> order_owner;
    int tmp_id = 0;
    for (uint i = 0; i < func->children.size(); i++) {
        auto& child = func->children[i];
        if (child->type != "loop") continue;
        loop_lvalues.clear();
        atomic_stmts.clear();
        loops.clear();
        order.clear();
        order_owner.clear();
        child->dfs([&](unique_ptr<KernelIR>& c) {
            auto& code = c->attrs[kir::code];
            if (code.find("atomic") != string::npos && c->has_attr(kir::rely)) {
                atomic_stmts.push_back(c.get());
            }
        });
        if (atomic_stmts.size() == 0) continue;

        // collect the loop nest; give up unless it is perfectly nested
        KernelIR* loop = child.get();
        loops.push_back(loop);
        loop_lvalues.push_back(loop->attrs[kir::lvalue]);
        order.push_back(loops.size() - 1);
        order_owner.push_back(-1);
        bool is_perfect_nest = true;
        while (1) {
            loop = loops.back();
            KernelIR* inner = nullptr;
            for (auto& c : loop->children) {
                if (c->type != "loop") continue;
                if (inner != nullptr) is_perfect_nest = false;
                inner = c.get();
            }
            if (inner == nullptr) break;
            if (loop->children.size() != 1) is_perfect_nest = false;
            if (!is_perfect_nest) break;
            ASSERT(loop->children.size() == 1);
            loops.push_back(inner);
            loop_lvalues.push_back(inner->attrs[kir::lvalue]);
            order.push_back(loops.size() - 1);
            order_owner.push_back(-1);
        }
        if (!is_perfect_nest) continue;

        // Every loop an atomic's address depends on has to sit outside the
        // loops it does not depend on, so pull them to the front of `order`.
        // The last "rely" entry is the trailing empty string of the list.
        for (uint ai = 0; ai < atomic_stmts.size(); ai++) {
            KernelIR* stmt = atomic_stmts[ai];
            auto rely = split(stmt->get_attr(kir::rely), ",");
            for (int ri = (int)rely.size() - 2; ri >= 0; ri--) {
                if (!rely[ri].size()) continue;
                int loop_idx = -1;
                int order_pos = -1;
                for (uint k = 0; k < loop_lvalues.size(); k++)
                    if (loop_lvalues[k] == rely[ri]) loop_idx = k;
                ASSERT(loop_idx != -1);
                for (uint k = 0; k < order.size(); k++)
                    if (order[k] == loop_idx) order_pos = k;
                ASSERT(order_pos != -1);
                for (int k = order_pos; k; k--) {
                    order[k] = order[k - 1];
                    order_owner[k] = order_owner[k - 1];
                }
                order[0] = loop_idx;
                order_owner[0] = ai;
            }
        }
        LOGvvvv << "atomic tuner order" << order;

        // The loops claimed by the first atomic statement stay outermost but
        // are reversed; everything after them is reversed too. `split_pos` is
        // at least 1 because order_owner[0] equals itself.
        vector<int> new_order;
        uint split_pos;
        for (split_pos = 0; split_pos < order.size(); split_pos++)
            if (order_owner[split_pos] != order_owner[0]) break;
        for (int j = (int)split_pos - 1; j >= 0; j--) new_order.push_back(order[j]);
        for (int j = (int)order.size() - 1; j >= (int)split_pos; j--) new_order.push_back(order[j]);
        loop_orders.push_back(new_order);
        loop_func_names.push_back(func->attrs[kir::lvalue]);

        // Interchange the loop headers so that nesting position p holds the
        // loop named loop_lvalues[order[p]].
        int count = 0;
        for (auto j : order) {
            uint k;
            for (k = count; k < loops.size(); k++)
                if (loops[k]->check_attr(kir::lvalue, loop_lvalues[j])) break;
            if (k < loops.size()) {
                loops[k]->swap(*loops[count++]);
            }
        }

        for (uint ai = 0; ai < atomic_stmts.size(); ai++) {
            KernelIR* stmt = atomic_stmts[ai];
            auto rely = split(stmt->get_attr(kir::rely), ",");
            // innermost nesting position this atomic's address depends on
            int last_rely_pos = -1;
            for (int ri = (int)rely.size() - 2; ri >= 0; ri--)
                for (uint k = 0; k < order.size(); k++)
                    if (loop_lvalues[order[k]] == rely[ri] && (int)k > last_rely_pos) last_rely_pos = k;
            vector<unique_ptr<expr::Expr>> matches;
            string tmp_name = "tmp" + std::to_string(tmp_id++);
            auto& code = stmt->attrs[kir::code];
            LOGvvvv << "atomic code" << code;
            auto e = expr::make(code.substr(0, code.size() - 1));

            // Try to recognise the atomic as one particular reduction:
            //   acc_code   what the statement becomes (accumulate into tmp)
            //   syms       symbols solved by the match, "a" is the destination
            //   cpu_pat / cuda_pat    the shape of the original statement
            //   cpu_flush / cuda_flush   the single atomic emitted afterwards
            auto try_atomic = [&](const string& acc_code, const vector<string>& syms,
                                  const string& cpu_pat, const string& cuda_pat,
                                  const string& cpu_flush, const string& cuda_flush) -> bool {
                auto pattern = is_cuda ? expr::make(cuda_pat) : expr::make(cpu_pat);
                if (!expr::match(e.get(), pattern.get(), syms, {}, matches)) return false;
                unordered_map<string, string> smap;
                for (uint si = 0; si < syms.size(); si++) smap[syms[si]] = matches[si]->to_string();
                string dst = smap["a"];
                // the destination must be a plain "pointer[index]"
                if (!expr::match(expr::make(dst).get(), expr::make("(c[d])").get(), {"c", "d"}, {}, matches))
                    return false;
                string ptr = matches[0]->to_string();
                string index = matches[1]->to_string();
                auto index_def = stmt->father->find_define(index);
                ASSERT(index_def != nullptr);
                // the address already changes in the innermost dependent loop,
                // so there is nothing to hoist
                if (last_rely_pos >= 0 && index_def->father == loops[last_rely_pos]) return true;
                auto& target = loops.at(last_rely_pos + 1);
                target->push_back(OpCompiler::precompile(smap, is_cuda ? cuda_flush : cpu_flush) + ";",
                                  &target->after);
                uint op_id, opvar_id;
                Op* op;
                Var* var;
                // ptr is "op{i}_{name}p"; drop the trailing 'p' to get the var
                pass->pm->oc->get_op_var_by_name(ptr.substr(0, ptr.length() - 1), op_id, opvar_id, op, var);
                auto init_code = get_reduce_init_code(op, var, is_cuda);
                if (var->dtype() != op->input(0)->dtype()) {
                    // accumulate in the output dtype, not the input one
                    code = OpCompiler::precompile(
                        smap, replace(acc_code, "@b", var->dtype().to_cstring() + string("(@b)"))) + ";";
                } else
                    code = OpCompiler::precompile(smap, acc_code) + ";";
                target->push_back(string(var->dtype().to_cstring()) + " " + tmp_name + "=" + init_code + ";",
                                  &target->before);
                string pat = is_cuda ? cuda_pat : cpu_pat;
                LOGvvv << "atomictuner: move " + pat.substr(0, pat.find("(")) + " to loop " +
                              std::to_string(last_rely_pos);
                move_definition_chain(index_def->father, target, index_def);
                return true;
            };

            // ::max / ::min on CUDA, std::max / std::min on CPU
            string max_ns = is_cuda ? "" : "std";
            if (try_atomic(tmp_name + "=" + tmp_name + "+@b", {"a", "b"},
                    "cpu_atomic_add(&a,b)", "atomicAdd(&a,b)",
                    "cpu_atomic_add(&@a," + tmp_name + ")", "atomicAdd(&@a," + tmp_name + ")") ||
                try_atomic(tmp_name + "=" + tmp_name + "-@b", {"a", "b"},
                    "cpu_atomic_sub(&a,b)", "atomicSub(&a,b)",
                    "cpu_atomic_sub(&@a," + tmp_name + ")", "atomicSub(&@a," + tmp_name + ")") ||
                try_atomic(tmp_name + "=" + tmp_name + "*@b", {"a", "b"},
                    "cpu_atomic_mul(&a,b)", "cuda_atomic_mul(&a,b)",
                    "cpu_atomic_mul(&@a," + tmp_name + ")", "cuda_atomic_mul(&@a," + tmp_name + ")") ||
                try_atomic(tmp_name + "=" + max_ns + "::max(@T@@(" + tmp_name + "),@T@@(@b))", {"a", "b", "T"},
                    "cpu_atomic_max(&a,T(b))", "cuda_atomic_max(&a,T(b))",
                    "cpu_atomic_max(&@a,@T@@(" + tmp_name + "))", "cuda_atomic_max(&@a,@T@@(" + tmp_name + "))") ||
                try_atomic(tmp_name + "=" + max_ns + "::min(@T@@(" + tmp_name + "),@T@@(@b))", {"a", "b", "T"},
                    "cpu_atomic_min(&a,T(b))", "cuda_atomic_min(&a,T(b))",
                    "cpu_atomic_min(&@a,@T@@(" + tmp_name + "))", "cuda_atomic_min(&@a,@T@@(" + tmp_name + "))") ||
                try_atomic(tmp_name + "=" + tmp_name + "&@b", {"a", "b"},
                    "cpu_atomic_and(&a,b)", "atomicAnd(&a,b)",
                    "cpu_atomic_and(&@a," + tmp_name + ")", "atomicAnd(&@a," + tmp_name + ")") ||
                try_atomic(tmp_name + "=" + tmp_name + "|@b", {"a", "b"},
                    "cpu_atomic_or(&a,b)", "atomicOr(&a,b)",
                    "cpu_atomic_or(&@a," + tmp_name + ")", "atomicOr(&@a," + tmp_name + ")") ||
                try_atomic(tmp_name + "=" + tmp_name + "^@b", {"a", "b"},
                    "cpu_atomic_xor(&a,b)", "atomicXor(&a,b)",
                    "cpu_atomic_xor(&@a," + tmp_name + ")", "atomicXor(&@a," + tmp_name + ")") ||
                try_atomic(tmp_name + "=" + tmp_name + "&&@b", {"a", "b"},
                    "cpu_atomic_and(&a,bool(b))", "atomicAnd(&a,bool(b))",
                    "cpu_atomic_and(&@a,bool(" + tmp_name + "))", "atomicAnd(&@a,bool(" + tmp_name + "))") ||
                try_atomic(tmp_name + "=" + tmp_name + "||@b", {"a", "b"},
                    "cpu_atomic_or(&a,bool(b))", "atomicOr(&a,bool(b))",
                    "cpu_atomic_or(&@a,bool(" + tmp_name + "))", "atomicOr(&@a,bool(" + tmp_name + "))") ||
                try_atomic(tmp_name + "=((bool(" + tmp_name + "))!=(bool(@b)))", {"a", "b"},
                    "cpu_atomic_xor(&@a,bool(@b))", "atomicXor(&@a,bool(@b))",
                    "cpu_atomic_xor(&@a,bool(@b))", "atomicXor(&@a,bool(@b))"))
                continue;
            LOGf << "Atomic not match" << e;
        }
    }
}

void AtomicTunerPass::run() {
    auto parallel = op->get_loop_option("parallel");
    bool is_cuda = op->flag(OpFlags::_cuda);
    if (is_cuda) parallel = 1;
    if (!parallel) return;
    vector<vector<int>> loop_orders;
    vector<string> loop_func_names;
    for (uint i = 0; i < ir->before.size(); i++) {
        auto& kernel = ir->before[i];
        if (kernel->get_attr(kir::dtype).find("__global__ void") == string::npos) continue;
        tune_atomic(this, kernel.get(), is_cuda, loop_orders, loop_func_names);
    }
    // tune_atomic reordered the loops inside each kernel; the thread range
    // declarations at the call site (tn0, tn1, ...) have to follow, because
    // tn{i} is consumed positionally by the kernel.
    for (uint fi = 0; fi < loop_func_names.size(); fi++)
        for (uint i = 0; i < ir->children.size(); i++) {
            auto& call = ir->children[i];
            int found = 0;
            for (uint k = 0; k < call->children.size(); k++) {
                auto& stmt = call->children[k];
                if (stmt->has_attr(kir::loop_func) && stmt->attrs[kir::loop_func] == loop_func_names[fi]) {
                    found = 1;
                    break;
                }
            }
            if (!found) continue;
            // first tn declaration; the reordered ones are swapped into place
            // starting here
            uint pos;
            for (pos = 0; pos < call->children.size(); pos++) {
                auto& stmt = call->children[pos];
                if (stmt->has_attr(kir::lvalue) && stmt->attrs[kir::lvalue].find("tn") == 0) break;
            }
            for (uint k = 0; k < loop_orders[fi].size(); k++) {
                for (uint ci = 0; ci < call->children.size(); ci++) {
                    auto& stmt = call->children[ci];
                    if (stmt->has_attr(kir::lvalue) &&
                        stmt->attrs[kir::lvalue].find("tn" + S(loop_orders[fi][k])) == 0) {
                        call->children[ci]->swap(*call->children[pos++]);
                        break;
                    }
                }
            }
            call->rebuild_scope();
            if (para_opt_level <= 3) {
                // The innermost loop after reordering is the one carrying the
                // atomic. Giving it every remaining thread makes the atomic
                // contend with itself, so cap it at a quarter of its range and
                // hand the threads back to the outer dimensions.
                int max_tn = 0;
                int max_pos = 0;
                for (uint j = 0; j < loop_orders[fi].size(); j++)
                    if (loop_orders[fi][j] > max_tn) {
                        max_tn = loop_orders[fi][j];
                        max_pos = j;
                    }
                if (max_pos > 0 && loop_orders[fi][max_pos - 1] < max_tn) {
                    vector<string> prev_tns;
                    for (uint ci = 0; ci < call->children.size(); ci++) {
                        auto& stmt = call->children[ci];
                        if (stmt->has_attr(kir::lvalue) && startswith(stmt->attrs[kir::lvalue], "tn" + S(max_tn))) {
                            auto& rvalue = stmt->attrs[kir::rvalue];
                            ASSERT(startswith(rvalue, "get_thread_range_log"));
                            LOGvvvv << "change rvalue from" << rvalue;
                            auto range = split(rvalue, ",").at(1);
                            range = split(range, ")").at(0);
                            rvalue = "get_thread_range_log(thread_num_left, std::min<int64>(" + range +
                                     ", std::max<int64>(" + range + "/4,32)))";
                            LOGvvvv << "change rvalue to" << rvalue;
                            if (para_opt_level >= 2) {
                                // keep at least 32 threads (5 bits) on the
                                // atomic dimension by borrowing from the
                                // dimensions declared before it
                                string tn_name = "tn" + S(max_tn);
                                for (auto& prev : prev_tns) {
                                    call->insert(ci + 1, "if (" + tn_name + "<5) {int _ = std::min(5-" +
                                                 tn_name + "," + prev + "); " + tn_name + "+=_; " + prev + "-=_;}");
                                }
                            }
                            break;
                        }
                        if (stmt->has_attr(kir::lvalue) && startswith(stmt->attrs[kir::lvalue], "tn")) {
                            prev_tns.push_back(stmt->attrs[kir::lvalue]);
                        }
                    }
                    if (para_opt_level >= 3) {
                        // What it does, read off the emitted expression: when
                        // the ranges of every thread dimension declared before
                        // the atomic one multiply out to 256 or less, and the
                        // kernel is reduction-dominated (reduce ops are at
                        // least a third of the fused op), the thread budget
                        // becomes max(thread_num/2, 65536) instead of
                        // thread_num -- a halving with a floor, so it only
                        // bites above 128k threads.
                        //
                        // NOTE: behaviour checked line by line against the
                        // original; the intent is not fully understood. Halving
                        // when there is little outer parallelism presumably
                        // trades threads for less atomic contention, but why
                        // the cutoffs are 256, 65536 and n_reduce*3 is not
                        // recoverable from the code. It also sits under the
                        // same "max_pos > 0" guard as the block above, which
                        // borrows threads *for* the atomic dimension; the two
                        // pull in opposite directions and nothing here says how
                        // they are meant to combine.
                        string range_product;
                        for (auto& prev : prev_tns) {
                            auto* def = call->find_define(prev);
                            ASSERT(def);
                            auto& rvalue = def->attrs[kir::rvalue];
                            auto range = split(rvalue, ",").at(1);
                            range = split(range, ")").at(0);
                            if (range_product.size()) range_product += '*';
                            range_product += '(';
                            range_product += range;
                            range_product += ')';
                        }
                        if (range_product.size()) {
                            auto* def = call->find_define("thread_num");
                            ASSERT(def);
                            auto& rvalue = def->attrs[kir::rvalue];
                            int n_reduce = 0;
                            for (auto o : op->ops)
                                if (o->type() == OpType::reduce) n_reduce++;
                            if ((int)op->ops.size() <= n_reduce * 3)
                                rvalue = "((" + range_product + ")<=256)?std::max(" + rvalue + "/2,65536):" + rvalue;
                        }
                    }
                }
            }
        }
    ir->remove_all_unused();
}

} // jittor
