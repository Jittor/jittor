// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
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

#define __get_thread_range_log \
inline static int get_thread_range_log(int& thread_num, int64 range) { \
    int nbits = NanoVector::get_nbits(std::min((int64)thread_num, range)) - 2; \
    thread_num >>= nbits; \
    return nbits; \
}

__get_thread_range_log

#define STR(a) #a
#define STR_MACRO(a) STR(a)

unique_ptr<expr::Expr> trace_and_expand(KernelIR* ir, expr::Expr* e) {
    auto a = e->clone();
    string rely=",";
    std::function<void(expr::Expr*)> func =
    [&](expr::Expr* c) {
        if (!c->is_sym()) return;
        auto def = ir->find_define(c->str);
        if (!def) return;
        ASSERT(def->type=="define");
        if (!def->has_attr("rvalue")) return;
        auto& rvalue = def->attrs["rvalue"];
        if (def->father && def->flist==&def->father->inner) {
            if (def->father->type=="func") return;
            if (def->father->type!="loop") return;
            LOGvvvv << "expand loop expr" << def->father->inner;
            // find x < range
            vector<unique_ptr<expr::Expr>> r1;
            if (!expr::match(
                expr::make(def->father->inner.at(1)->attrs.at("code")).get(),
                expr::make(c->str+"<range").get(),
                {"range"}, {}, r1))
                return;
            rely+=c->str+",";
            // find x++ or x+=stride
            vector<unique_ptr<expr::Expr>> r2;
            if (expr::match(
                expr::make(def->father->inner.at(2)->attrs.at("code")).get(),
                expr::make(c->str+"++").get())) {
                r2.push_back(expr::make("1"));
            } else
            if (!expr::match(
                expr::make(def->father->inner.at(2)->attrs.at("code")).get(),
                expr::make(c->str+"+=stride").get(),
                {"stride"}, {}, r2))
                return;
            // tid + loop_cnt * tnum
            auto new_expr = expr::make_op("+",
                expr::make(rvalue),
                expr::make_op("*", 
                    expr::make("loop_cnt"),
                    r2.at(0)->clone()
                )
            );
            c->swap(new_expr.get());
            return;
        }
        c->swap(expr::make(rvalue).get());
        if (!c->children.size()) func(c);
    };
    a->dfs(func);
    // indexes of relyied loop, split with ","
    ir->attrs["rely"] = rely;
    return a;
}

static void check_atomic(KernelIR* ir, bool is_cuda, int tdim) {
    ir->dfs([&](unique_ptr<KernelIR>& c) {
        if (c->type != "") return;
        if (!c->has_attr("code")) return;
        auto& code = c->attrs["code"];
        auto e = expr::make(code.substr(0, code.size()-1)); // remove ';'
        vector<unique_ptr<expr::Expr>> results;
        auto target = expr::make("a=b");
        if (!expr::match(e.get(), target.get(), {"a", "b"}, {}, results))
            return;
        bool has_a = 0;
        results[1]->dfs([&](expr::Expr* p) {
            if (p->to_string()==results[0]->to_string())
                has_a = 1;
        });
        if (!has_a) return;
        vector<unique_ptr<expr::Expr>> ptr_and_offset;
        if (!expr::match(results[0].get(), expr::make("a[b]").get(), {"a", "b"}, {}, ptr_and_offset))
            return;
        LOGvvvv << "ptr_and_offset" << ptr_and_offset;
        auto offset = trace_and_expand(c.get(), ptr_and_offset.at(1).get())
            ->simplify();
        LOGvvvv << "rely" << c->get_attr("rely");
        LOGvvvv << "full offset expr" << offset->to_string(1);
        // try to optimize unneccesary atomic operation
        bool need_atomic = false;
        for (int i=0; i<tdim; i++) {
            vector<unique_ptr<expr::Expr>> xres;
            if (!expr::match(
                offset.get(),
                expr::make("(tid"+S(i)+"+tnum"+S(i)+"*a)*b+c").get(),
                {"a","b","c"}, {"tid"+S(i)}, xres
            )) {
                LOGvvvv << "offset" << offset << "not match, need atomic";
                need_atomic = true;
                break;
            }
            LOGvvvv << "atomic optimize match:" << i << xres;
            // set tid=0 and simplify
            offset = offset->assign_symbol({{"tid"+S(i),"0"}})->simplify();
            LOGvvvv << "new offset" << offset;
        }
        if (!need_atomic) return;

        // add atomic code
        auto check = [&](const string& t, const vector<string>& args, const string& cpu, const string& cuda) -> bool {
            auto target = expr::make(t);
            if (!expr::match(e.get(), target.get(), args, {}, results))
                return false;
            unordered_map<string,string> defs;
            for (int i=0; i<args.size(); i++)
                defs[args[i]] = results[i]->to_string();
            code = OpCompiler::precompile(defs, is_cuda ? cuda : cpu) + ";";
            LOGvvvv << "matched" << results << code;
            return true;
        };
        if (
            check("a=a+b", {"a","b"}, "cpu_atomic_add(&@a,@b)", "atomicAdd(&@a,@b)") ||
            check("a=a-b", {"a","b"}, "cpu_atomic_sub(&@a,@b)", "atomicSub(&@a,@b)") ||
            check("a=a*b", {"a","b"}, "cpu_atomic_mul(&@a,@b)", "cuda_atomic_mul(&@a,@b)") ||
            check("a=std::max(T(a),T(b))", {"a","b","T"}, "cpu_atomic_max(&@a,@T@@(@b))", "cuda_atomic_max(&@a,@T@@(@b))") ||
            check("a=::max(T(a),T(b))", {"a","b","T"}, "cpu_atomic_max(&@a,@T@@(@b))", "cuda_atomic_max(&@a,@T@@(@b))") ||
            check("a=std::min(T(a),T(b))", {"a","b","T"}, "cpu_atomic_min(&@a,@T@@(@b))", "cuda_atomic_min(&@a,@T@@(@b))") ||
            check("a=::min(T(a),T(b))", {"a","b","T"}, "cpu_atomic_min(&@a,@T@@(@b))", "cuda_atomic_min(&@a,@T@@(@b))") ||
            check("a=a&b", {"a","b"}, "cpu_atomic_and(&@a,@b)", "atomicAnd(&@a,@b)") ||
            check("a=a|b", {"a","b"}, "cpu_atomic_or(&@a,@b)", "atomicOr(&@a,@b)") ||
            check("a=a^b", {"a","b"}, "cpu_atomic_xor(&@a,@b)", "atomicXor(&@a,@b)") ||
            check("a=a&&b", {"a","b"}, "cpu_atomic_and(&@a,bool(@b))", "atomicAnd(&@a,bool(@b))") ||
            check("a=a||b", {"a","b"}, "cpu_atomic_or(&@a,bool(@b))", "atomicOr(&@a,bool(@b))") ||
            check("a=((bool(a))!=(bool(b)))", {"a","b"}, "cpu_atomic_xor(&@a,bool(@b))", "atomicXor(&@a,bool(@b))")
        )
            return;
        LOGf << "Expr not match" << e;
    });
}

int to_pow(int x) {
    return 1 << (NanoVector::get_nbits(x) - 2);
}

void ParallelPass::run() {
    auto choice = op->get_loop_option("parallel");
    auto fix_thread_num = op->get_loop_option("fix_thread_num", 0);
    bool is_cuda = op->flags.get(NodeFlags::_cuda);
    if (is_cuda) choice=1;
    if (!choice) return;

    int cuda_block_num = to_pow(op->get_loop_option("cuda_block_num", 256));
    int cuda_thread_num = to_pow(op->get_loop_option("cuda_thread_num", 1024));
    int cpu_thread_num = to_pow(op->get_loop_option("cpu_thread_num", omp_get_max_threads()));
    int max_parallel_depth;
    if (!is_cuda) {
        // omp include
        ir->push_front("#include \"misc/cpu_atomic.h\"", &ir->before);
        ir->push_front("#include <omp.h>", &ir->before);
        max_parallel_depth = op->get_loop_option("max_parallel_depth", 2);
        auto* lva_pass = pm->get_pass<LoopVarAnalyzePass>("loop_var_analyze");
        auto number_of_ranges = lva_pass->number_of_ranges;
        if (!op->loop_options->count("max_parallel_depth")) {
            if (number_of_ranges<=max_parallel_depth)
                max_parallel_depth = number_of_ranges-1;
        }
        if (max_parallel_depth<=0) return;
    } else {
        ir->push_front("#include \"helper_cuda.h\"", &ir->before);
        ir->push_front("#include \"misc/cuda_limits.h\"", &ir->before);
        ir->push_front("#include \"misc/cuda_atomic.h\"", &ir->before);
        max_parallel_depth = op->get_loop_option("max_parallel_depth", 4);
    }
    ir->push_back("#pragma GCC diagnostic ignored \"-Wunused-function\"", &ir->before, true);
    ir->push_back(STR_MACRO(__get_thread_range_log), &ir->before, true);
    
    for (uint i=0; i<ir->children.size(); i++) {
        auto& func_call = ir->children[i];
        if (!func_call->has_attr("loop_func")) continue;
        auto& func_name = func_call->attrs["loop_func"];
        uint j=0;
        while (j<ir->before.size() && !ir->before[j]->check_attr("lvalue", func_name))
            j++;
        ASSERT(j<ir->before.size()) << "loop func" << func_name << "not found.";
        
        auto& func_def = ir->before[j];
        auto c = func_def->children.back().get();
        ASSERTop(c->type,==,"loop");
        // only one loop
        ASSERT(func_def->children.size()==1 || 
            func_def->children[func_def->children.size()-2]->type!="loop");
        vector<KernelIR*> cs;
        vector<string> rvalues, strides;
        for (int j=0; j<max_parallel_depth; j++) {
            if (!c->has_attr("rvalue")) break;
            if (!c->has_attr("lvalue")) break;
            auto& lvalue = c->attrs["lvalue"];
            auto& stride = c->inner[2]->attrs["code"];
            if (stride == lvalue+"++;") {
                strides.push_back("1");
            } else {
                if (!c->has_attr("rvalue2")) break;
                auto& rvalue2 = c->attrs["rvalue2"];
                if (stride != lvalue+"+="+rvalue2+";") break;
                strides.push_back(rvalue2);
            }
            rvalues.push_back(c->attrs["rvalue"]);
            cs.push_back(c);
            LOGvvvv << "Parallel loop dep=">>j<<"range=" >> rvalues.back() << 
                "stride=" >> strides.back()
                << "code:" << c->inner;
            if (c->children.size()==1 && c->children[0]->type=="loop") {
                c = c->children[0].get();
            } else {
                break;
            }
        }
        (void)get_thread_range_log;
        KernelIR new_block("{}");
        auto new_func_call = func_call->clone();
        auto new_func_def = func_def->clone();
        vector<KernelIR*> ncs;
        c = new_func_def->children.back().get();
        for (int j=0; j<cs.size(); j++) {
            ncs.push_back(c);
            if (c->children.size()==0) break;
            c = c->children[0].get();
        } 
        auto& func_call_code = new_func_call->attrs["code"];
        int thread_num = is_cuda ?
            cuda_block_num * cuda_thread_num
            : cpu_thread_num;
        // resolve undefined rvalues
        for (auto& rv : rvalues) {
            auto e = expr::make(rv);
            if (!e->is(expr::_number)) {
                auto rdef = func_def->find_define(rv);
                ASSERT(rdef);
                if (rdef->has_attr("rvalue"))
                    rv = rdef->attrs["rvalue"];
            }
        }

        // calc max thread num
        string nums = rvalues.at(0);
        for (int i=1; i<rvalues.size(); i++)
            nums+="*"+rvalues[i];
        new_block.push_back("int thread_num=" + S(thread_num) + ";");
        new_block.push_back("int thread_num_left=thread_num;");

        for (int j=ncs.size()-1; j>=0; j--) {
            auto& rv = rvalues[j];
            new_block.push_back("int tn"+S(j)+
            "=get_thread_range_log(thread_num_left, "+rv+");");
            func_call_code = func_call_code.substr(0, func_call_code.size()-2)
                + ",tn" + S(j) + ");";
            new_func_def->push_back("int tn"+S(j)+";", &new_func_def->inner);
        }
        for (int j=ncs.size()-2; j>0; j--) {
            new_block.push_back("tn"+S(j)+"=tn"+S(j)+"+tn"+S(j+1)+";");
        }
        new_block.push_back("tn0=NanoVector::get_nbits(thread_num)-2;");
        new_block.push_back("int p1 = std::max(thread_num/1024, 1);");
        new_block.push_back("int p2 = std::min(thread_num, 1024);");
        KernelIR new_tid_def("{}");
        if (!is_cuda) {
            // omp thread id
            new_tid_def.push_front("int thread_id = omp_get_thread_num();");
            // omp func call
            // we set num_threads in code
            new_func_call->push_back(
                "#pragma omp parallel num_threads(thread_num)", 
                &new_func_call->before
            );
        } else {
            new_func_def->get_attr("dtype") = "__launch_bounds__("+S(cuda_thread_num)+") __global__ void";
            new_tid_def.push_front("int thread_id = blockIdx.x * blockDim.x + threadIdx.x;");
            // cuda kernel launch
            auto& code = func_call_code;
            auto pos = code.find("(");
            ASSERT(pos != string::npos);
            code = code.substr(0, pos) +
                "<<<p1,p2>>>" +
                code.substr(pos);
        }

        new_block.push_back(move(new_func_call));
        LOGvvvv << "new block:" << new_block.to_string();
        new_tid_def.push_back("int tn"+S(ncs.size())+"=0;");
        for (int j=0; j<ncs.size(); j++) {
            new_tid_def.push_back("int tnum"+S(j)+
                " = 1<<(tn"+S(j)+"-tn"+S(j+1)+");");
            new_tid_def.push_back("int tid"+S(j)+
                " = (thread_id>>tn"+S(j+1)+") & (tnum"+S(j)+"-1);");
            auto c = ncs[j];
            auto& lvalue = c->attrs["lvalue"];
            auto& stride = c->inner[2]->attrs["code"];
            string new_stride, new_init;
            // change
            // for (T i=0; i<range; i+=stride)
            // to
            // for (T i=stride*thread_id; i<range; i+=stride*thread_num)
            // TODO: check loop deps
            if (stride == lvalue+"++;") {
                new_stride = lvalue+"+=tnum"+S(j)+";";
                new_init = lvalue+"=tid"+S(j)+";";
            } else {
                if (!c->has_attr("rvalue2")) continue;
                auto& rvalue2 = c->attrs["rvalue2"];
                if (stride != lvalue+"+="+rvalue2+";") continue;
                new_stride = lvalue+"+="+rvalue2+"*tnum"+S(j)+";";
                new_init = lvalue+"="+rvalue2+"*tid"+S(j)+";";
            }
            LOGvvvv << "Parallel loop" << c->attrs["loop_id"] << "with new stride" << new_stride;
            if (c->inner[0]->type == "define")
                new_init = c->inner[0]->attrs["dtype"] + " " + new_init;
            stride = new_stride;
            c->inner[0]->try_parse_define(new_init);
        }
        LOGvvvv << "new_tid_def:" << new_tid_def.to_string();
        check_atomic(new_func_def.get(), is_cuda, ncs.size());
        new_func_def->insert(0, new_tid_def.children);
        new_func_def->swap(*func_def, true);
        new_block.swap(*func_call, true);
        auto code = func_def->to_string(); 
        bool has_atomic = code.find("atomic") != string::npos;
        if (!fix_thread_num) {
            if (has_atomic) {
                nums += "/16";
            }
            func_call->find_define("thread_num")->attrs["rvalue"] = "min(max(1<<(NanoVector::get_nbits(" + nums + ")-2),32)," + S(thread_num) + ")";
        }
    }
    ir->remove_all_unused();
}

} // jittor