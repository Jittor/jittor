// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include <functional>
#include <omp.h>
#include "var.h"
#include "opt/expr.h"
#include "opt/pass_manager.h"
#include "opt/pass/parallel_pass.h"
#include "opt/pass/loop_var_analyze_pass.h"

namespace jittor {

EXTERN_LIB vector<int> cuda_archs;

// Host mirror of the device helper emitted below. Kept next to the emitted
// source so the two stay in sync; the compiler only sees it through the
// (void) reference in ParallelPass::run.
inline static int get_thread_range_log(int& thread_num, int64 range) {
    int nbits = NanoVector::get_nbits(std::min((int64)thread_num, range)) - 2;
    thread_num >>= nbits;
    return nbits;
}

// Splits the thread budget over the loop nest: takes the largest power of two
// that is no larger than both the remaining budget and this loop's range,
// returns its log2 and subtracts it from the budget.
const char* get_thread_range_log_src =
    "inline static int get_thread_range_log(int& thread_num, int64 range) {"
    "int nbits = NanoVector::get_nbits(std::min((int64)thread_num, range)) - 2;"
    "thread_num >>= nbits;"
    "return nbits;}";

extern int para_opt_level;

// Rewrite `e` -- an index expression of a store -- in terms of the enclosing
// loop variables, replacing every symbol by its definition. A loop induction
// variable becomes "init + loop_cnt*stride", where loop_cnt stands for the
// iteration number, so that the caller can ask whether the address depends on
// a given thread dimension.
//
// Side effect: ir->attrs[kir::rely] is set to a comma separated list of the loop
// variables the expression turned out to depend on. AtomicTunerPass reads it.
unique_ptr<expr::Expr> expand_offset_expr(KernelIR* ir, expr::Expr* e) {
    auto ret = e->clone();
    string rely = ",";
    std::function<void(expr::Expr*)> func = [&](expr::Expr* node) {
        if (!node->is_sym()) return;
        auto def = ir->find_define(node->str);
        if (!def) return;
        ASSERT(def->type == "define");
        if (!def->has_attr(kir::rvalue)) return;
        auto& rvalue = def->attrs[kir::rvalue];
        // defined in the header of an enclosing statement
        if (def->father && def->flist == &def->father->inner) {
            if (def->father->type == "func") return;
            if (def->father->type != "loop") return;
            LOGvvvv << "expand loop expr" << def->father->inner;
            // the loop must look like "for (T i=init; i<range; i+=stride)"
            vector<unique_ptr<expr::Expr>> matches;
            if (!expr::match(expr::make(def->father->inner.at(1)->require_attr(kir::code)).get(),
                             expr::make(node->str + "<range").get(), {"range"}, {}, matches))
                return;
            rely += node->str + ",";
            vector<unique_ptr<expr::Expr>> stride_match;
            if (expr::match(expr::make(def->father->inner.at(2)->require_attr(kir::code)).get(),
                            expr::make(node->str + "++").get())) {
                stride_match.push_back(expr::make("1"));
            } else if (!expr::match(expr::make(def->father->inner.at(2)->require_attr(kir::code)).get(),
                                    expr::make(node->str + "+=stride").get(), {"stride"}, {}, stride_match))
                return;
            auto new_expr = expr::make_op("+", expr::make(rvalue),
                expr::make_op("*", expr::make("loop_cnt"), stride_match.at(0)->clone()));
            node->swap(new_expr.get());
            return;
        }
        node->swap(expr::make(rvalue).get());
        // the substituted rvalue may itself be a bare symbol
        if (!node->children.size()) func(node);
    };
    ret->dfs(func);
    ir->attrs[kir::rely] = rely;
    return ret;
}

// After the loop nest has been strip-mined across threads, a statement of the
// form "a[i] = f(a[i], b)" is only safe if every thread writes a different
// address. Decide that per statement, and turn the unsafe ones into atomics.
//
// force_atomic short-circuits the analysis (reindex_reduce can alias
// arbitrarily, so its stores are always atomic).
static void replace_with_atomic(KernelIR* ir, bool is_cuda, int parallel_depth, bool force_atomic) {
    ir->dfs([&](unique_ptr<KernelIR>& stmt) {
        if (stmt->type != "") return;
        if (!stmt->has_attr(kir::code)) return;
        auto& code = stmt->attrs[kir::code];
        auto e = expr::make(code.substr(0, code.size() - 1));
        vector<unique_ptr<expr::Expr>> matches;
        auto pattern = expr::make("a=b");
        if (!expr::match(e.get(), pattern.get(), {"a", "b"}, {}, matches)) return;
        // only self-updating statements can race
        bool is_self_update = 0;
        matches[1]->dfs([&](expr::Expr* sub) {
            if (sub->to_string() == matches[0]->to_string()) is_self_update = 1;
        });
        if (!is_self_update) return;
        vector<unique_ptr<expr::Expr>> ptr_and_offset;
        if (!expr::match(matches[0].get(), expr::make("a[b]").get(), {"a", "b"}, {}, ptr_and_offset)) return;
        LOGvvvv << "ptr_and_offset" << ptr_and_offset;
        auto offset = expand_offset_expr(stmt.get(), ptr_and_offset.at(1).get())->simplify();
        LOGvvvv << "rely" << stmt->get_attr(kir::rely);
        LOGvvvv << "full offset expr" << offset->to_string(1);
        bool need_atomic = force_atomic;
        // The offset must look like (tid{d} + tnum{d}*a)*b + c for every
        // thread dimension d, with a and b non-zero: only then does changing
        // tid{d} always change the address.
        for (int d = 0; d < parallel_depth; d++) {
            vector<unique_ptr<expr::Expr>> m;
            if (!expr::match(offset.get(),
                    expr::make("(tid" + S(d) + "+tnum" + S(d) + "*a)*b+c").get(),
                    {"a", "b", "c"}, {"tid" + S(d)}, m)) {
                LOGvvvv << "offset" << offset << "not match, need atomic";
                need_atomic = true;
                break;
            }
            if (m[0]->to_string() == "0" || m[1]->to_string() == "0") {
                LOGvvvv << "offset" << offset << "has zero matched, need atomic";
                need_atomic = true;
                break;
            }
            LOGvvvv << "atomic optimize match:" << d << m;
            // this dimension is proven distinct, drop it and check the next
            offset = offset->assign_symbol({{"tid" + S(d), "0"}})->simplify();
            LOGvvvv << "new offset" << offset;
        }
        if (!need_atomic) return;
        auto try_atomic = [&](const string& pat, const vector<string>& syms,
                              const string& cpu_code, const string& cuda_code) -> bool {
            auto pattern = expr::make(pat);
            if (!expr::match(e.get(), pattern.get(), syms, {}, matches)) return false;
            unordered_map<string, string> smap;
            for (uint i = 0; i < syms.size(); i++) smap[syms[i]] = matches[i]->to_string();
            code = OpCompiler::precompile(smap, is_cuda ? cuda_code : cpu_code) + ";";
            LOGvvvv << "matched" << matches << code;
            return true;
        };
        if (try_atomic("a=a+b", {"a", "b"}, "cpu_atomic_add(&@a,@b)", "atomicAdd(&@a,@b)") ||
            try_atomic("a=a-b", {"a", "b"}, "cpu_atomic_sub(&@a,@b)", "atomicSub(&@a,@b)") ||
            try_atomic("a=a*b", {"a", "b"}, "cpu_atomic_mul(&@a,@b)", "cuda_atomic_mul(&@a,@b)") ||
            try_atomic("a=std::max(T(a),T(b))", {"a", "b", "T"},
                "cpu_atomic_max(&@a,@T@@(@b))", "cuda_atomic_max(&@a,@T@@(@b))") ||
            try_atomic("a=::max(T(a),T(b))", {"a", "b", "T"},
                "cpu_atomic_max(&@a,@T@@(@b))", "cuda_atomic_max(&@a,@T@@(@b))") ||
            try_atomic("a=std::min(T(a),T(b))", {"a", "b", "T"},
                "cpu_atomic_min(&@a,@T@@(@b))", "cuda_atomic_min(&@a,@T@@(@b))") ||
            try_atomic("a=::min(T(a),T(b))", {"a", "b", "T"},
                "cpu_atomic_min(&@a,@T@@(@b))", "cuda_atomic_min(&@a,@T@@(@b))") ||
            try_atomic("a=a&b", {"a", "b"}, "cpu_atomic_and(&@a,@b)", "atomicAnd(&@a,@b)") ||
            try_atomic("a=a|b", {"a", "b"}, "cpu_atomic_or(&@a,@b)", "atomicOr(&@a,@b)") ||
            try_atomic("a=a^b", {"a", "b"}, "cpu_atomic_xor(&@a,@b)", "atomicXor(&@a,@b)") ||
            try_atomic("a=a&&b", {"a", "b"}, "cpu_atomic_and(&@a,bool(@b))", "atomicAnd(&@a,bool(@b))") ||
            try_atomic("a=a||b", {"a", "b"}, "cpu_atomic_or(&@a,bool(@b))", "atomicOr(&@a,bool(@b))") ||
            try_atomic("a=((bool(a))!=(bool(b)))", {"a", "b"},
                "cpu_atomic_xor(&@a,bool(@b))", "atomicXor(&@a,bool(@b))"))
            return;
        LOGf << "Expr not match" << e;
    });
}

// largest power of two not greater than v
int round_down_pow2(int v) {
    return 1 << (NanoVector::get_nbits(v) - 2);
}

void ParallelPass::run() {
    auto parallel = op->get_loop_option("parallel");
    auto fix_thread_num = op->get_loop_option("fix_thread_num", 0);
    bool is_cuda = op->flag(OpFlags::_cuda);
    if (is_cuda) parallel = 1;
    if (!parallel) return;
    int default_block_num = 256;
    for (auto arch : cuda_archs) {
        if (arch >= 80) default_block_num = 2048;
    }
    int block_num = round_down_pow2(op->get_loop_option("cuda_block_num", default_block_num));
    int cuda_thread_num = round_down_pow2(op->get_loop_option("cuda_thread_num", 1024));
    int cpu_thread_num = round_down_pow2(op->get_loop_option("cpu_thread_num", omp_get_max_threads()));
    int max_parallel_depth;
    if (!is_cuda) {
        ir->push_front("#include \"misc/cpu_atomic.h\"", &ir->before);
        ir->push_front("#include <omp.h>", &ir->before);
        max_parallel_depth = op->get_loop_option("max_parallel_depth", 2);
        auto* lva = pm->get_pass<LoopVarAnalyzePass>();
        auto number_of_ranges = lva->number_of_ranges;
        // leave the innermost range serial unless the user asked otherwise
        if (!op->loop_options->count("max_parallel_depth")) {
            if (number_of_ranges <= max_parallel_depth) max_parallel_depth = number_of_ranges - 1;
        }
        if (max_parallel_depth <= 0) return;
    } else {
        ir->push_front("#include \"helper_cuda.h\"", &ir->before);
        ir->push_front("#include \"misc/cuda_limits.h\"", &ir->before);
        ir->push_front("#include \"misc/cuda_atomic.h\"", &ir->before);
        max_parallel_depth = op->get_loop_option("max_parallel_depth", 4);
    }
    ir->push_back("#pragma GCC diagnostic ignored \"-Wunused-function\"", &ir->before, true);
    ir->push_back(get_thread_range_log_src, &ir->before, true);
    for (uint i = 0; i < ir->children.size(); i++) {
        auto& call = ir->children[i];
        if (!call->has_attr(kir::loop_func)) continue;
        auto& func_name = call->attrs[kir::loop_func];
        uint j = 0;
        while (j < ir->before.size() && !ir->before[j]->check_attr(kir::lvalue, func_name)) j++;
        ASSERT(j < ir->before.size()) << "loop func" << func_name << "not found.";
        auto& func = ir->before[j];
        auto loop = func->children.back().get();
        ASSERTop(loop->type, ==, "loop");
        ASSERT(func->children.size() == 1 ||
               func->children[func->children.size() - 2]->type != "loop");

        // Walk down the loop nest, collecting the ranges and strides of the
        // outermost `max_parallel_depth` loops. Stop at the first loop whose
        // header we do not recognise.
        vector<KernelIR*> loops;
        vector<string> ranges, strides;
        for (int d = 0; d < max_parallel_depth; d++) {
            if (!loop->has_attr(kir::rvalue)) break;
            if (!loop->has_attr(kir::lvalue)) break;
            auto& lvalue = loop->attrs[kir::lvalue];
            auto& step_code = loop->inner[2]->attrs[kir::code];
            if (step_code == lvalue + "++;") {
                strides.push_back("1");
            } else {
                if (!loop->has_attr(kir::rvalue2)) break;
                auto& stride = loop->attrs[kir::rvalue2];
                if (step_code != lvalue + "+=" + stride + ";") break;
                strides.push_back(stride);
            }
            ranges.push_back(loop->attrs[kir::rvalue]);
            loops.push_back(loop);
            LOGvvvv << "Parallel loop dep=" >> d << "range=" >> ranges.back()
                    << "stride=" >> strides.back() << "code:" << loop->inner;
            if (loop->children.size() == 1 && loop->children[0]->type == "loop") {
                loop = loop->children[0].get();
            } else {
                break;
            }
        }
        (void)get_thread_range_log;

        // new_block replaces the call site, new_func the kernel function
        KernelIR new_block("{}");
        auto new_call = call->clone();
        auto new_func = func->clone();
        vector<KernelIR*> new_loops;
        loop = new_func->children.back().get();
        for (uint d = 0; d < loops.size(); d++) {
            new_loops.push_back(loop);
            if (loop->children.size() == 0) break;
            loop = loop->children[0].get();
        }
        auto& call_code = new_call->attrs[kir::code];
        int total_thread_num = is_cuda ? block_num * cuda_thread_num : cpu_thread_num;
        // a range may be a variable; use its definition so the expression is
        // valid at the call site
        for (auto& range : ranges) {
            auto e = expr::make(range);
            if (!e->is(expr::_number)) {
                auto def = func->find_define(range);
                ASSERT(def);
                if (def->has_attr(kir::rvalue)) range = def->attrs[kir::rvalue];
            }
        }
        string total_range = ranges.at(0);
        for (uint k = 1; k < ranges.size(); k++) total_range += "*" + ranges[k];

        // Thread range setup, from the innermost dimension outwards, so the
        // innermost loop gets the low bits of the thread id.
        new_block.push_back("int thread_num=" + S(total_thread_num) + ";");
        new_block.push_back("int thread_num_left=thread_num;");
        for (int d = new_loops.size() - 1; d >= 0; d--) {
            auto& range = ranges[d];
            new_block.push_back("int tn" + S(d) + "=get_thread_range_log(thread_num_left, " + range + ");");
            // append ",tnd" to the kernel call arguments
            call_code = call_code.substr(0, call_code.size() - 2) + ",tn" + S(d) + ");";
            new_func->push_back("int tn" + S(d) + ";", &new_func->inner);
        }
        // The kernel decodes tid{d} from bit tn{d+1} up to bit tn{d}, so the
        // tn's have to be cumulative (suffix sums), while get_thread_range_log
        // returns one dimension's bit count at a time. Accumulate them here,
        // for both backends: the CPU template needs exactly the same boundaries
        // as CUDA, and used to get them from a regex that rewrote these lines
        // in the finished source (OpCompiler::fix_parallel_thread_ranges).
        //
        // The only exception is the CUDA para_opt_level==0 path, which does not
        // derive tn0 from the chain at all but from the thread count directly,
        // so its accumulation stops one dimension earlier.
        int accumulate_down_to = (is_cuda && !para_opt_level) ? 1 : 0;
        for (int d = (int)new_loops.size() - 2; d >= accumulate_down_to; d--) {
            new_block.push_back("tn" + S(d) + "=tn" + S(d) + "+tn" + S(d + 1) + ";");
        }
        if (is_cuda) {
            if (para_opt_level) {
                new_block.push_back("tn0=std::max(tn0, 5);");
                new_block.push_back("thread_num=1<<tn0;");
                new_block.push_back("int p1 = std::max(thread_num/" + S(cuda_thread_num) + ", 1);");
                new_block.push_back("int p2 = std::min(thread_num, " + S(cuda_thread_num) + ");");
            } else {
                new_block.push_back("tn0=NanoVector::get_nbits(thread_num)-2;");
                new_block.push_back("int p1 = std::max(thread_num/1024, 1);");
                new_block.push_back("int p2 = std::min(thread_num, 1024);");
            }
        } else if (new_loops.size()) {
            // Enter the omp region with the number of threads actually handed
            // out, which is 1<<tn0 now that tn0 is cumulative -- the same value
            // the CUDA branch computes one line above. With no parallel loop at
            // all there is no tn0 and nothing to correct.
            new_block.push_back("thread_num=1<<tn0;");
        }

        KernelIR tid_def("{}");
        if (!is_cuda) {
            tid_def.push_front("int thread_id = omp_get_thread_num();");
            new_call->push_back("#pragma omp parallel num_threads(thread_num)", &new_call->before);
        } else {
            new_func->get_attr(kir::dtype) =
                "__launch_bounds__(" + S(cuda_thread_num) + ") __global__ void";
            tid_def.push_front("int thread_id = blockIdx.x * blockDim.x + threadIdx.x;");
            auto& code = call_code;
            auto pos = code.find("(");
            ASSERT(pos != string::npos);
            code = code.substr(0, pos) + "<<<p1,p2>>>" + code.substr(pos);
        }
        new_block.push_back(move(new_call));
        LOGvvvv << "new block:" << new_block.to_string();

        // Decode this thread's coordinate per dimension and re-stride the
        // loops so that each thread walks its own slice.
        tid_def.push_back("int tn" + S(new_loops.size()) + "=0;");
        for (uint d = 0; d < new_loops.size(); d++) {
            tid_def.push_back("int tnum" + S(d) + " = 1<<(tn" + S(d) + "-tn" + S(d + 1) + ");");
            tid_def.push_back("int tid" + S(d) + " = (thread_id>>tn" + S(d + 1) + ") & (tnum" + S(d) + "-1);");
            auto nl = new_loops[d];
            auto& lvalue = nl->attrs[kir::lvalue];
            auto& step_code = nl->inner[2]->attrs[kir::code];
            string new_step, new_init;
            if (step_code == lvalue + "++;") {
                new_step = lvalue + "+=tnum" + S(d) + ";";
                new_init = lvalue + "=tid" + S(d) + ";";
            } else {
                if (!nl->has_attr(kir::rvalue2)) continue;
                auto& stride = nl->attrs[kir::rvalue2];
                if (step_code != lvalue + "+=" + stride + ";") continue;
                new_step = lvalue + "+=" + stride + "*tnum" + S(d) + ";";
                new_init = lvalue + "=" + stride + "*tid" + S(d) + ";";
            }
            LOGvvvv << "Parallel loop" << nl->attrs[kir::loop_id] << "with new stride" << new_step;
            if (nl->inner[0]->type == "define") new_init = nl->inner[0]->attrs[kir::dtype] + " " + new_init;
            step_code = new_step;
            nl->inner[0]->try_parse_define(new_init);
        }
        LOGvvvv << "new_tid_def:" << tid_def.to_string();

        bool force_atomic = false;
        for (auto o : op->ops) {
            if (o->name() == string("reindex_reduce")) {
                force_atomic = true;
            }
        }
        replace_with_atomic(new_func.get(), is_cuda, new_loops.size(), force_atomic);
        new_func->insert(0, tid_def.children);
        new_func->swap(*func, true);
        new_block.swap(*call, true);

        auto code = func->to_string();
        bool has_atomic = code.find("atomic") != string::npos;
        if (!fix_thread_num) {
            if (para_opt_level) {
                auto n_thread = total_thread_num;
                // a reduction-dominated kernel contends on its atomics, so
                // give it fewer threads
                if (has_atomic) {
                    int n_reduce = 0;
                    for (auto o : op->ops)
                        if (o->type() == OpType::reduce) n_reduce++;
                    if ((int)op->ops.size() <= n_reduce * 2)
                        n_thread = std::max(n_thread / 4, 32);
                    else if ((int)op->ops.size() <= n_reduce * 3)
                        n_thread = std::max(n_thread / 2, 32);
                }
                call->find_define("thread_num")->attrs[kir::rvalue] = S(n_thread);
            } else {
                if (has_atomic) {
                    total_range += "/16";
                }
                call->find_define("thread_num")->attrs[kir::rvalue] =
                    "min(max(1<<(NanoVector::get_nbits(" + total_range + ")-2),32)," +
                    S(total_thread_num) + ")";
            }
        }
    }
    ir->remove_all_unused();
}

} // jittor
