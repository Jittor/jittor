// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// All Rights Reserved.
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

/*
move a statements and its relied statements from inner loop to outer loop:

for ... // outer loop
    for ...
        for ... // inner loop
            statement_not_rely
            statement_x
            statement_y // def

-->

statement_x
for ... // outer loop
    for ...
        for ... // inner loop
            statement_not_rely

statement_y // def

 */
static void move_rely(KernelIR* inner_loop, KernelIR* outer_loop, KernelIR* def){
    // move all dependence of def from inner_loop to outer_loop
    vector<KernelIR*> q{def};
    map<KernelIR*, int> visited;
    visited[def]=1;
    outer_loop->push_front(def->move_out(), &outer_loop->before);
    for (int i=0; i<q.size(); i++) {
        auto e = expr::make(q[i]->attrs["rvalue"]);
        LOGvvvv << "move_rely" << e->to_string();
        e->dfs([&](expr::Expr* a) {
            if (!a->is_sym()) return;
            auto ir = inner_loop->find_define(a->str);
            if (ir==nullptr) return;
            if (!ir->father) return;
            // TODO: definition between inner loop and outer loop
            if (ir->father != inner_loop) return;
            if (!visited.count(ir)) {
                outer_loop->push_front(ir->move_out(), &outer_loop->before);
                q.push_back(ir);
                visited[ir]=1;
            }
        });
    }
}

// find init value of correspondence op and var
string find_init_value(Op* op, Var* var, bool is_cuda) {
    // example: reindex_reduce.minimun
    auto names = split(op->name_ex(), ".");
    ASSERT(names.size()==2) << names;
    // find init value, such as
    // *  add: tmp = 0
    // *  min: tmp = numeric_max<float32>
    // the init value is load from binary_op_defs.h header file
    auto init_code = OpCompiler::precompile(
        {
            {"OP",names.back()}, 
            {"T", var->dtype().to_cstring()}, 
            {is_cuda?"JIT_cuda":"JIT_cpu", "1"}
        }, 
        "#include \"ops/binary_op_defs.h\"\n@expand_macro(init_@OP, @T)");
    return init_code;
}

// sorder: Array that saves the allocation order of "tn"
// sfunc: Array of function names
static void tune_atomic(Pass* pass, KernelIR* ir, bool is_cuda, int tdim, vector<vector<int>> &sorder, vector<string> &sfunc) {
    LOGvvvv << "tune_atomic" << ir->children;
    vector<string> relys;
    vector<string> idx_name;
    vector<KernelIR*> atomics;
    vector<KernelIR*> loops;
    vector<int> nrely;
    vector<int> order;
    int tmp_cnt=0;
    for (uint i=0; i<ir->children.size(); i++) {
        auto& c = ir->children[i];
        if (c->type != "loop") continue;
        relys.clear();
        idx_name.clear();
        atomics.clear();
        loops.clear();
        order.clear();
        nrely.clear();

        c->dfs([&](unique_ptr<KernelIR>& p) {
            auto& code = p->attrs["code"];
            if (code.find("atomic")!=-1 && p->has_attr("rely")){
                atomics.push_back(p.get());
            }
        });
        if (atomics.size()==0) continue;

        // get loops & idx_name
        KernelIR* loop = c.get();
        loops.push_back(loop);
        idx_name.push_back(loop->attrs["lvalue"]);
        order.push_back(loops.size()-1);
        nrely.push_back(-1);
        bool ok = true;
        while (1) {
            loop = loops.back();
            KernelIR* loop2 = nullptr;
            for (auto& p : loop->children) {
                if (p->type != "loop")
                    continue;
                // TODO: only support single loop children
                if (loop2 != nullptr) ok = false;
                loop2 = p.get();
            }
            if (loop2 == nullptr) break;
            // TODO: only support single loop children
            if (loop->children.size() != 1) ok = false;
            if (!ok) break;
            ASSERT(loop->children.size()==1);
            loops.push_back(loop2);
            idx_name.push_back(loop2->attrs["lvalue"]);
            order.push_back(loops.size()-1);
            nrely.push_back(-1);
        }
        // TODO: only support single loop children
        if (!ok) continue;

        // reorder
        for (uint j=0;j<atomics.size();j++) {
            KernelIR* p=atomics[j];
            auto si=split(p->get_attr("rely"),",");
            for (int k=(int)si.size()-2;k>=0;k--) {
                // ignore empty string
                if (!si[k].size())
                    continue;
                int sidx=-1;
                int sord=-1;
                for (uint l=0;l<idx_name.size();l++)
                    if (idx_name[l]==si[k]) sidx=l;
                ASSERT(sidx != -1);
                for (uint l=0;l<order.size();l++)
                    if (order[l]==sidx) sord=l;
                ASSERT(sord != -1);
                for (int l=sord;l;l--){
                    order[l]=order[l-1];
                    nrely[l]=nrely[l-1];
                }
                order[0]=sidx;
                nrely[0]=j;
            }
        }
        LOGvvvv << "atomic tuner order" << order;

        vector<int> tnorder;
        uint si;
        for (si=0;si<order.size();si++)
            if (nrely[si]!=nrely[0]) break;
        for (int j=si-1;j>=0;j--) tnorder.push_back(order[j]);
        for (int j=order.size()-1;j>=si;j--) tnorder.push_back(order[j]);
        sorder.push_back(tnorder);
        sfunc.push_back(ir->attrs["lvalue"]);

        // sort loop with order
        int count=0;
        for (auto j : order) {
            uint k;
            for (k=count; k<loops.size(); k++)
                if (loops[k]->check_attr("loop_id", S(j)))
                    break;
            if (k<loops.size())
                loops[k]->swap(*loops[count++]);
        }

        // move atomic
        for (uint j=0;j<atomics.size();j++) {
            KernelIR* p=atomics[j];
            auto si=split(p->get_attr("rely"),",");
            int sidx=-1;
            for (int k=si.size()-2;k>=0;k--)
                for (int l=0;l<order.size();l++)
                    if (idx_name[order[l]]==si[k] && l>sidx) sidx=l;

            vector<unique_ptr<expr::Expr>> results;
            string stmp = "tmp"+std::to_string(tmp_cnt++);
            auto& code = p->attrs["code"];
            LOGvvvv << "atomic code" << code;
            auto e = expr::make(code.substr(0, code.size()-1));
            // add atomic code
            auto check = [&](const string& t, const vector<string>& args, const string& cpu, const string& cuda, const string& acpu, const string& acuda) -> bool {
                auto target = is_cuda ? expr::make(cuda) : expr::make(cpu);
                if (!expr::match(e.get(), target.get(), args, {}, results))
                    return false;
                unordered_map<string,string> defs;
                for (int i=0; i<args.size(); i++)
                    defs[args[i]] = results[i]->to_string();

                string a=defs["a"];
                if (!expr::match(expr::make(a).get(), expr::make("(c[d])").get(), {"c","d"}, {}, results))
                    return false;
                // dvar[didx]
                string dvar=results[0]->to_string();
                string didx=results[1]->to_string();

                auto def=p->father->find_define(didx);
                ASSERT(def != nullptr);
                if (sidx>=0 && def->father == loops[sidx])
                    return true;
                auto& loop_i = loops.at(sidx+1);
                code = OpCompiler::precompile(defs, t) + ";";
                loop_i->push_back(
                    OpCompiler::precompile(defs, is_cuda ? acuda : acpu) + ";", 
                    &loop_i->after);
                uint op_id, opvar_id;
                Op* op;
                Var* var;
                pass->pm->oc->get_op_var_by_name(dvar.substr(0,dvar.length()-1), op_id, opvar_id, op, var);
                auto init_code = find_init_value(op, var, is_cuda);
                loop_i->push_back(string(var->dtype().to_cstring())+" "+stmp+"="+init_code+";", &loop_i->before);
                string sa=is_cuda ? cuda : cpu;
                LOGvvv << "atomictuner: move "+sa.substr(0,sa.find("("))+" to loop "+std::to_string(sidx);
                move_rely(def->father, loop_i, def);
                return true;
            };
            string sstd=is_cuda ? "" : "std";
            if (
                check(stmp+"="+stmp+"+@b", {"a","b"}, "cpu_atomic_add(&a,b)", "atomicAdd(&a,b)", "cpu_atomic_add(&@a,"+stmp+")", "atomicAdd(&@a,"+stmp+")") ||
                check(stmp+"="+stmp+"-@b", {"a","b"}, "cpu_atomic_sub(&a,b)", "atomicSub(&a,b)", "cpu_atomic_sub(&@a,"+stmp+")", "atomicSub(&@a,"+stmp+")") ||
                check(stmp+"="+stmp+"*@b", {"a","b"}, "cpu_atomic_mul(&a,b)", "cuda_atomic_mul(&a,b)", "cpu_atomic_mul(&@a,"+stmp+")", "cuda_atomic_mul(&@a,"+stmp+")") ||
                check(stmp+"="+sstd+"::max(@T@@("+stmp+"),@T@@(@b))", {"a","b","T"}, "cpu_atomic_max(&a,T(b))", "cuda_atomic_max(&a,T(b))", "cpu_atomic_max(&@a,@T@@("+stmp+"))", "cuda_atomic_max(&@a,@T@@("+stmp+"))") ||
                check(stmp+"="+sstd+"::max(@T@@("+stmp+"),@T@@(@b))", {"a","b","T"}, "cpu_atomic_max(&a,T(b))", "cuda_atomic_max(&a,T(b))", "cpu_atomic_max(&@a,@T@@("+stmp+"))", "cuda_atomic_max(&@a,@T@@("+stmp+"))") ||
                check(stmp+"="+sstd+"::min(@T@@("+stmp+"),@T@@(@b))", {"a","b","T"}, "cpu_atomic_min(&a,T(b))", "cuda_atomic_min(&a,T(b))", "cpu_atomic_min(&@a,@T@@("+stmp+"))", "cuda_atomic_min(&@a,@T@@("+stmp+"))") ||
                check(stmp+"="+sstd+"::min(@T@@("+stmp+"),@T@@(@b))", {"a","b","T"}, "cpu_atomic_min(&a,T(b))", "cuda_atomic_min(&a,T(b))", "cpu_atomic_min(&@a,@T@@("+stmp+"))", "cuda_atomic_min(&@a,@T@@("+stmp+"))") ||
                check(stmp+"="+stmp+"&@b", {"a","b"}, "cpu_atomic_and(&a,b)", "atomicAnd(&a,b)", "cpu_atomic_and(&@a,"+stmp+")", "atomicAnd(&@a,"+stmp+")") ||
                check(stmp+"="+stmp+"|@b", {"a","b"}, "cpu_atomic_or(&a,b)", "atomicOr(&a,b)", "cpu_atomic_or(&@a,"+stmp+")", "atomicOr(&@a,"+stmp+")") ||
                check(stmp+"="+stmp+"^@b", {"a","b"}, "cpu_atomic_xor(&a,b)", "atomicXor(&a,b)", "cpu_atomic_xor(&@a,"+stmp+")", "atomicXor(&@a,"+stmp+")") ||
                check(stmp+"="+stmp+"&&@b", {"a","b"}, "cpu_atomic_and(&a,bool(b))", "atomicAnd(&a,bool(b))", "cpu_atomic_and(&@a,bool("+stmp+"))", "atomicAnd(&@a,bool("+stmp+"))") ||
                check(stmp+"="+stmp+"||@b", {"a","b"}, "cpu_atomic_or(&a,bool(b))", "atomicOr(&a,bool(b))", "cpu_atomic_or(&@a,bool("+stmp+"))", "atomicOr(&@a,bool("+stmp+"))") ||
                check(stmp+"=((bool("+stmp+"))!=(bool(@b)))", {"a","b"}, "cpu_atomic_xor(&@a,bool(@b))", "atomicXor(&@a,bool(@b))", "cpu_atomic_xor(&@a,bool(@b))", "atomicXor(&@a,bool(@b))")
            ) continue;
            LOGf << "Atomic not match" << e;
        }
    }
    return;
}

void AtomicTunerPass::run() {
    auto choice = op->get_loop_option("parallel");
    bool is_cuda = op->flags.get(NodeFlags::_cuda);
    if (is_cuda) choice=1;
    if (!choice) return;

    vector<vector<int>> sorder;
    vector<string> sfunc;
    for (uint i=0; i<ir->before.size(); i++) {
        auto& func_call = ir->before[i];
        // TODO: remove this if
        if (func_call->get_attr("dtype").find("__global__ void") == string::npos) continue;
        tune_atomic(this, func_call.get(), is_cuda, 4, sorder, sfunc);
    }

    // Re-adjust the allocation order of "tn" according to the situation of atomic coverage, preferentially allocate the range not covered by atomic, for example:
    // for (op0_index_t id0 = tid0; id0<range0; id0+=tnum0) {
    //     for (op1_index_t id1 = tid1; id1<range1; id1+=tnum1) {
    //         for (op2_index_t id2 = tid2; id2<range2; id2+=tnum2) {
    //             for (op3_index_t id3 = tid3; id3<range3; id3+=tnum3) {
    //                 ...
    //             }
    //         }
    //         atomicAdd(...);
    //     }
    // }
    // The allocation order of "tn" will be: tn1, tn0, tn3, tn2
    for (uint j=0;j<sfunc.size();j++)
        for (uint i=0; i<ir->children.size(); i++) {
            auto& func_call = ir->children[i];
            int bo=0;
            for (uint k=0; k<func_call->children.size(); k++){
                auto& save = func_call->children[k];
                if (save->has_attr("loop_func") && save->attrs["loop_func"]==sfunc[j]){
                    bo=1;
                    break;
                }
            }
            if (!bo) continue;
            uint k;
            for (k=0; k<func_call->children.size(); k++){
                auto& save = func_call->children[k];
                if (save->has_attr("lvalue") && save->attrs["lvalue"].find("tn")==0) break;
            }
            for (uint l=0;l<sorder[j].size();l++){
                for (uint p=0; p<func_call->children.size(); p++){
                    auto& save = func_call->children[p];
                    if (save->has_attr("lvalue") && save->attrs["lvalue"].find("tn"+S(sorder[j][l]))==0){
                        func_call->children[p]->swap(*func_call->children[k++]);
                        break;
                    }
                }
            }
        }
    ir->remove_all_unused();
}

} // jittor