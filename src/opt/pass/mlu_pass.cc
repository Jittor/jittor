// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/mlu_pass.h"
#include "misc/mlu_flags.h"
#include "opt/pass/loop_var_analyze_pass.h"
#include "opt/expr.h"

namespace jittor {

unique_ptr<expr::Expr> mlu_trace_and_expand(KernelIR* ir, expr::Expr* e) {
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

void MLUPass::add_memcpy(KernelIR* loop_father, KernelIR* loop, vector<string> vars, vector<string> types, vector<int> is_input, string new_id, vector<string> &nram_vars, vector<string> &nram_types) {
    vector<int> solved;
    for (int i=0;i<is_input.size();i++) solved.push_back(0);
    for (auto& c : loop->children) {
        if (c->type != "" && c->type != "define") continue;
        string code;
        if (c->type == "define") code = c->attrs["lvalue"]+" = "+c->attrs["rvalue"];
        else {
            if (!c->has_attr("code")) continue;
            code = c->attrs["code"];
            code = code.substr(0, code.size()-1); // remove ';'
        }

        vector<string> vars_offsets;
        vector<int> vars_idxs;

        for (int i=0;i<vars.size();i++){
            int lastpos=-1;
            while (code.find(vars[i],lastpos+1)!=-1){
                lastpos=code.find(vars[i],lastpos+1);
                vars_offsets.push_back(code.substr(lastpos+vars[i].size()+1, code.find("]",lastpos)-lastpos-vars[i].size()-1));
                vars_idxs.push_back(i);
            }
        }
        for (int k=0;k<vars_offsets.size();k++){
            int vars_idx = vars_idxs[k];
            if (solved[vars_idx]) continue;
            solved[vars_idx] = 1;
            auto offset = mlu_trace_and_expand(c.get(), expr::make(vars_offsets[k]).get())
                ->simplify();

            string idx_code = offset->to_string(1);
            string cpy_size="1";
            if (idx_code.find(new_id)!=-1) cpy_size="range"+new_id.substr(2);
            while (idx_code.find(new_id)!=-1){
                idx_code.replace(idx_code.find(new_id), new_id.size(), "0");
            }
            if (is_input[vars_idx]){
                idx_code = "__memcpy("+vars[vars_idx]+"_nram, "+vars[vars_idx]+" + "+idx_code+", "+cpy_size+" * sizeof("+types[vars_idx]+"), GDRAM2NRAM);";
                loop->push_front(idx_code, &loop->before);
                idx_code = "__nramset("+vars[vars_idx]+"_nram, "+S(nram_space)+", 0);";
                loop->push_front(idx_code, &loop->before);
            }else{
                idx_code = "__memcpy("+vars[vars_idx]+" + "+idx_code+", "+vars[vars_idx]+"_nram, "+cpy_size+" * sizeof("+types[vars_idx]+"), NRAM2GDRAM);";
                loop->push_front(idx_code, &loop->after);
            }
        }
    }
    for (int i=0;i<solved.size();i++)
        if (solved[i]) {
            nram_vars.push_back(vars[i]);
            nram_types.push_back(types[i]);
        }
}

// is_vec&len=n : 2
// is_vec&len=1 : 1
// not_vec : 0
int check_is_vec(string str, vector<string>& define_vars) {
    if (str.size()>1){
        if (str.find("_1",str.size()-2)+2==str.size()) return 2;
        if (str.find("_0",str.size()-2)+2==str.size()) return 1;
    }
    for (int i=0; i<define_vars.size(); i++)
    if (str==define_vars[i]) return 2;
    return 0;
}

string remove_01(string str){
    if (str.find("_1",str.size()-2)+2==str.size() || str.find("_0",str.size()-2)+2==str.size()) return str.substr(0, str.size()-2);
    return str;
}
string create_bang_var(vector<string>& define_vars){
    string res = "bang_var_n"+S(define_vars.size());
    define_vars.push_back(res);
    return res;
}

int MLUPass::getConvertType(string a, string varb)
// return
// 0:is not cast
// 1:int8 to float32
// 2:float32 to int8
{
    if (a.find("op")!=0 ||  a.find("_")==-1) return 0;
    int opnum=std::stoi(a.substr(2, a.find("_")-2));
    Op* sop = op->ops[opnum];
    if (sop->name_ex()!="unary.cast") return 0;
    string ta=sop->outputs().front()->dtype().to_cstring();
    string tb=sop->inputs().front()->dtype().to_cstring();
    ASSERT(ta=="int8" || ta=="float32");
    ASSERT(tb=="int8" || tb=="float32");
    if (ta==tb) return 0;
    else if (ta=="float32" && tb=="int8") return 1;
    else if (ta=="int8" && tb=="float32") return 2;
    else ASSERT(0);
    return 0;
}

int MLUPass::bang_dfs(unique_ptr<KernelIR>& func, string dst, unique_ptr<expr::Expr>& rval, vector<string>& define_vars, vector<string> &bang_code, string new_range){
    vector<unique_ptr<expr::Expr>> res;
    string vara,varb,varc;
    string fake_range="fake_range";
    bool is_addequ=expr::match(rval.get(), expr::make("a+b").get(), {"a","b"}, {}, res);
    is_addequ = (dst.find("tmp")==0 || (is_addequ && res.at(0)->clone()->to_string()==dst));
    if (!(check_is_vec(dst, define_vars)==2) && !is_addequ) return 0;

    string ori_dst=dst;
    dst = remove_01(dst);
    LOGvvvv << "bang_dfs  dst:" << dst << "rval:" << rval;
    if (is_addequ)
    // dst = dst+a
    {   
        if (!expr::match(rval.get(), expr::make("a+b").get(), {"a","b"}, {}, res)) return 0;
        auto a=res.at(0)->clone();
        auto b=res.at(1)->clone();
        LOGvvvv << "a=a+b:" << a << "#+#" << b;
        if (a->children.size()==0) vara = a->to_string();
        else{
            vara=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, vara, a, define_vars, bang_code, new_range));
        }
        if (b->children.size()==0) varb = b->to_string();
        else{
            varb=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, varb, b, define_vars, bang_code, new_range));
        }
        int is_single_vec=false;
        if (vara.find("_0",vara.size()-2)+2==vara.size()) is_single_vec=1;
        vara = remove_01(vara);
        if (vara!=dst) return 0;
        
        if (is_single_vec) dst+="[0]";
        LOGvvvv << "dst = dst+a";
        ASSERT(check_is_vec(varb, define_vars)==2);
        varb = remove_01(varb);
        varc=create_bang_var(define_vars);
        bang_code.push_back("__bang_reduce_sum("+varc+", "+varb+", "+fake_range+");");
        string vard=create_bang_var(define_vars);
        bang_code.push_back("__memcpy("+vard+", "+varc+", 1*sizeof(float), NRAM2NRAM, 1*sizeof(float), 32*sizeof(float), "+fake_range+"/32-1);");
        bang_code.push_back("__bang_reduce_sum("+varc+", "+vard+", "+fake_range+"/ 32);");
        string vare=create_bang_var(define_vars);
        bang_code.push_back("__memcpy("+vare+", "+varc+", 1*sizeof(float), NRAM2NRAM, 1*sizeof(float), 32*sizeof(float), "+fake_range+"/32/32-1);");
        bang_code.push_back("__bang_reduce_sum("+varc+", "+vare+", 32);");
        bang_code.push_back(dst+"="+dst+"+"+varc+"[0];");
    }else if (rval->children.size()==0)
    // dst = a
    {
        auto a=rval->clone();
        LOGvvvv << "memcpy:" << a;
        vara = a->to_string();
        int sv=check_is_vec(vara, define_vars);
        vara = remove_01(vara);
        if (sv==2){
            bang_code.push_back("__memcpy("+dst+", "+vara+", "+fake_range+" * sizeof(float), NRAM2NRAM);");
        } else if (sv==1) {
            // bang_code.push_back("__nramset("+dst+", "+fake_range+", "+vara+"[0]);");
            bang_code.push_back("for (int iset=0;iset<"+new_range+";iset++) {"+dst+"[iset]="+vara+"[0];}");
            bang_code.push_back(dst+"[0]="+vara+"[0];");
        } else {
            // bang_code.push_back("__nramset("+dst+", "+fake_range+", "+vara+");");
            bang_code.push_back("for (int iset=0;iset<"+new_range+";iset++) {"+dst+"[iset]="+vara+";}");
            bang_code.push_back(dst+"[0]="+vara+";");
        }
    } else if (expr::match(rval.get(), expr::make("std::sqrt(a)").get(), {"a"}, {}, res))
    // dst = sqrt(a)
    {
        auto a=res.at(0)->clone();
        LOGvvvv << "sqrt(a):" << a;
        if (a->children.size()==0) vara = a->to_string();
        else{
            vara=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, vara, a, define_vars, bang_code, new_range));
        }
        ASSERT(check_is_vec(vara, define_vars)==2);
        vara = remove_01(vara);
        bang_code.push_back("__bang_active_sqrt("+dst+", "+vara+", "+fake_range+");");

    } else if (expr::match(rval.get(), expr::make("a(b)").get(), {"a","b"}, {}, res))
    // dst = a(b)
    {
        auto a=res.at(0)->clone();
        auto b=res.at(1)->clone();
        varb = b->to_string();
        varb = remove_01(varb);

        int res=getConvertType(a->to_string(), varb);
        if (res==0){
            ASSERT(bang_dfs(func, ori_dst, b, define_vars, bang_code, new_range));
        }else if (res==1){
            bang_code.push_back("__bang_int82float("+dst+", "+varb+", "+fake_range+", 0);");
        }else if (res==2){
            bang_code.push_back("__bang_float2int8_rd("+dst+", "+varb+", "+fake_range+", 0);");
        }
    } else if (expr::match(rval.get(), expr::make("a+b").get(), {"a","b"}, {}, res))
    // dst = a+b
    {
        auto a=res.at(0)->clone();
        auto b=res.at(1)->clone();
        LOGvvvv << "a+b:" << a << "#+#" << b;
        if (a->children.size()==0) vara = a->to_string();
        else{
            vara=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, vara, a, define_vars, bang_code, new_range));
        }
        if (b->children.size()==0) varb = b->to_string();
        else{
            varb=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, varb, b, define_vars, bang_code, new_range));
        }
        ASSERT(check_is_vec(vara, define_vars)==2);
        ASSERT(check_is_vec(varb, define_vars)==2);
        vara = remove_01(vara);
        varb = remove_01(varb);
        bang_code.push_back("__bang_add("+dst+", "+vara+", "+varb+", "+fake_range+");");
    } else if (expr::match(rval.get(), expr::make("a*b").get(), {"a","b"}, {}, res))
    // dst = a*b
    {
        auto a=res.at(0)->clone();
        auto b=res.at(1)->clone();
        LOGvvvv << "a*b:" << a << "#*#" << b;
        if (a->children.size()==0) vara = a->to_string();
        else{
            vara=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, vara, a, define_vars, bang_code, new_range));
        }
        if (b->children.size()==0) varb = b->to_string();
        else{
            varb=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, varb, b, define_vars, bang_code, new_range));
        }
        if (vara=="(-1)"){
            ASSERT(check_is_vec(varb, define_vars)==2);
            varb = remove_01(varb);
            bang_code.push_back("__bang_mul_const("+dst+", "+varb+", -1.0, "+fake_range+");");
        } else {
            ASSERT(check_is_vec(vara, define_vars)==2);
            ASSERT(check_is_vec(varb, define_vars)==2);
            vara = remove_01(vara);
            varb = remove_01(varb);
            bang_code.push_back("__bang_mul("+dst+", "+vara+", "+varb+", "+fake_range+");");
        }
    } else if (expr::match(rval.get(), expr::make("a/b").get(), {"a","b"}, {}, res))
    // dst = a/b
    {
        auto a=res.at(0)->clone();
        auto b=res.at(1)->clone();
        LOGvvvv << "a/b:" << a << "#/#" << b;
        if (a->children.size()==0) vara = a->to_string();
        else{
            vara=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, vara, a, define_vars, bang_code, new_range));
        }
        if (b->children.size()==0) varb = b->to_string();
        else{
            varb=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, varb, b, define_vars, bang_code, new_range));
        }
        ASSERT(check_is_vec(vara, define_vars)==2);
        ASSERT(check_is_vec(varb, define_vars)==2);
        varc=create_bang_var(define_vars);
        vara = remove_01(vara);
        varb = remove_01(varb);
        bang_code.push_back("__bang_div("+dst+", "+vara+", "+varb+", "+varc+", "+fake_range+");");
    } else if (expr::match(rval.get(), expr::make("a>b").get(), {"a","b"}, {}, res))
    // dst = a>b
    {
        auto a=res.at(0)->clone();
        auto b=res.at(1)->clone();
        LOGvvvv << "a>b:" << a << "#>#" << b;
        if (a->children.size()==0) vara = a->to_string();
        else{
            vara=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, vara, a, define_vars, bang_code, new_range));
        }
        if (b->children.size()==0) varb = b->to_string();
        else{
            varb=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, varb, b, define_vars, bang_code, new_range));
        }
        ASSERT(check_is_vec(vara, define_vars)==2);
        ASSERT(check_is_vec(varb, define_vars)==2);
        vara = remove_01(vara);
        varb = remove_01(varb);
        bang_code.push_back("__bang_gt("+dst+", "+vara+", "+varb+", "+fake_range+");");
    }  else if (expr::match(rval.get(), expr::make("a<b").get(), {"a","b"}, {}, res))
    // dst = a<b
    {
        auto a=res.at(0)->clone();
        auto b=res.at(1)->clone();
        LOGvvvv << "a<b:" << a << "#<#" << b;
        if (a->children.size()==0) vara = a->to_string();
        else{
            vara=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, vara, a, define_vars, bang_code, new_range));
        }
        if (b->children.size()==0) varb = b->to_string();
        else{
            varb=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, varb, b, define_vars, bang_code, new_range));
        }
        ASSERT(check_is_vec(vara, define_vars)==2);
        ASSERT(check_is_vec(varb, define_vars)==2);
        vara = remove_01(vara);
        varb = remove_01(varb);
        bang_code.push_back("__bang_lt("+dst+", "+vara+", "+varb+", "+fake_range+");");
    }  else if (expr::match(rval.get(), expr::make("a>=b").get(), {"a","b"}, {}, res))
    // dst = a>=b
    {
        auto a=res.at(0)->clone();
        auto b=res.at(1)->clone();
        LOGvvvv << "a>=b:" << a << "#>=#" << b;
        if (a->children.size()==0) vara = a->to_string();
        else{
            vara=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, vara, a, define_vars, bang_code, new_range));
        }
        if (b->children.size()==0) varb = b->to_string();
        else{
            varb=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, varb, b, define_vars, bang_code, new_range));
        }
        ASSERT(check_is_vec(vara, define_vars)==2);
        ASSERT(check_is_vec(varb, define_vars)==2);
        vara = remove_01(vara);
        varb = remove_01(varb);
        bang_code.push_back("__bang_ge("+dst+", "+vara+", "+varb+", "+fake_range+");");
    }  else if (expr::match(rval.get(), expr::make("a<=b").get(), {"a","b"}, {}, res))
    // dst = a<=b
    {
        auto a=res.at(0)->clone();
        auto b=res.at(1)->clone();
        LOGvvvv << "a<=b:" << a << "#<=#" << b;
        if (a->children.size()==0) vara = a->to_string();
        else{
            vara=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, vara, a, define_vars, bang_code, new_range));
        }
        if (b->children.size()==0) varb = b->to_string();
        else{
            varb=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, varb, b, define_vars, bang_code, new_range));
        }
        ASSERT(check_is_vec(vara, define_vars)==2);
        ASSERT(check_is_vec(varb, define_vars)==2);
        vara = remove_01(vara);
        varb = remove_01(varb);
        bang_code.push_back("__bang_le("+dst+", "+vara+", "+varb+", "+fake_range+");");
    } else if (expr::match(rval.get(), expr::make("a&&b").get(), {"a","b"}, {}, res))
    // dst = a&&b
    {
        auto a=res.at(0)->clone();
        auto b=res.at(1)->clone();
        LOGvvvv << "a&&b:" << a << "#&&#" << b;
        if (a->children.size()==0) vara = a->to_string();
        else{
            vara=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, vara, a, define_vars, bang_code, new_range));
        }
        if (b->children.size()==0) varb = b->to_string();
        else{
            varb=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, varb, b, define_vars, bang_code, new_range));
        }
        ASSERT(check_is_vec(vara, define_vars)==2);
        ASSERT(check_is_vec(varb, define_vars)==2);
        vara = remove_01(vara);
        varb = remove_01(varb);
        bang_code.push_back("__bang_and("+dst+", "+vara+", "+varb+", "+fake_range+");");
    } else if (expr::match(rval.get(), expr::make("std::max(a,b)").get(), {"a","b"}, {}, res))
    // dst = std::max(a,b)
    {
        auto a=res.at(0)->clone();
        auto b=res.at(1)->clone();
        LOGvvvv << "max(a,b):" << a << "#max#" << b;
        if (a->children.size()==0) vara = a->to_string();
        else{
            vara=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, vara, a, define_vars, bang_code, new_range));
        }
        if (b->children.size()==0) varb = b->to_string();
        else{
            varb=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, varb, b, define_vars, bang_code, new_range));
        }
        ASSERT(check_is_vec(vara, define_vars)==2);
        ASSERT(check_is_vec(varb, define_vars)==2);
        vara = remove_01(vara);
        varb = remove_01(varb);
        varc = create_bang_var(define_vars);
        string varaa = create_bang_var(define_vars);
        string varbb = create_bang_var(define_vars);
        string varcc = create_bang_var(define_vars);
        bang_code.push_back("__bang_gt("+varc+", "+vara+", "+varb+", "+fake_range+");");
        bang_code.push_back("__nramset("+varaa+", "+fake_range+", (float)1.0);");
        bang_code.push_back("__bang_sub("+varaa+", "+varaa+", "+varc+", "+fake_range+");");
        bang_code.push_back("__bang_mul("+varbb+", "+varc+", "+vara+", "+fake_range+");");
        bang_code.push_back("__bang_mul("+varcc+", "+varaa+", "+varb+", "+fake_range+");");
        bang_code.push_back("__bang_add("+dst+", "+varbb+", "+varcc+", "+fake_range+");");
    } else if (expr::match(rval.get(), expr::make("a?b:c").get(), {"a","b","c"}, {}, res))
    // dst = a?b:c
    {
        auto a=res.at(0)->clone();
        auto b=res.at(1)->clone();
        auto c=res.at(2)->clone();
        LOGvvvv << "a?b:c:" << a << "#?#" << b << "#:#" << c;
        if (a->children.size()==0) vara = a->to_string();
        else{
            vara=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, vara, a, define_vars, bang_code, new_range));
        }
        if (b->children.size()==0) varb = b->to_string();
        else{
            varb=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, varb, b, define_vars, bang_code, new_range));
        }
        if (c->children.size()==0) varc = c->to_string();
        else{   
            varc=create_bang_var(define_vars);
            ASSERT(bang_dfs(func, varc, c, define_vars, bang_code, new_range));
        }
        ASSERT(check_is_vec(vara, define_vars)==2);
        ASSERT(check_is_vec(varb, define_vars)==2);
        ASSERT(check_is_vec(varc, define_vars)==2);
        vara = remove_01(vara);
        varb = remove_01(varb);
        varc = remove_01(varc);
        string varaa=create_bang_var(define_vars);
        string varbb=create_bang_var(define_vars);
        string varcc=create_bang_var(define_vars);
        bang_code.push_back("__nramset("+varaa+", "+fake_range+", (float)1.0);");
        bang_code.push_back("__bang_sub("+varaa+", "+varaa+", "+vara+", "+fake_range+");");
        bang_code.push_back("__bang_mul("+varbb+", "+vara+", "+varb+", "+fake_range+");");
        bang_code.push_back("__bang_mul("+varcc+", "+varaa+", "+varc+", "+fake_range+");");
        bang_code.push_back("__bang_add("+dst+", "+varbb+", "+varcc+", "+fake_range+");");
    } else ASSERT(0);
    return 1;
}

int MLUPass::check_int(){
    for (auto& c : ir->father->children)
    if (c->type=="macro") {
        string str=c->get_attr("code");
        if (str.find("#define")==-1 || str.find("_T")==-1) continue;
        if (str.find("int")!=-1) return 1;
    }
    return 0;
}

void MLUPass::convert_to_bang(unique_ptr<KernelIR>& func, KernelIR* loop, vector<string> vars, string new_id, string new_range) {
    vector<string> define_vars;
    vector<string> bang_code;
    vector<string> calc_code;
    vector<string> calc_idxs;
    vector<int> is_vec;
    for (auto& c : loop->children) {
        if (c->type != "" && c->type != "define") continue;
        string code;
        if (c->type == "define") code = c->attrs["lvalue"]+" = "+c->attrs["rvalue"];
        else {
            if (!c->has_attr("code")) continue;
            code = c->attrs["code"];
            code = code.substr(0, code.size()-1);
        }

        vector<string> vars_offsets;
        vector<int> vars_idxs;

        for (int i=0;i<vars.size();i++){
            int lastpos=-1;
            while (code.find(vars[i],lastpos+1)!=-1){
                lastpos=code.find(vars[i],lastpos+1);
                vars_offsets.push_back(code.substr(code.find("[",lastpos)+1, code.find("]",lastpos)-code.find("[",lastpos)-1));
                vars_idxs.push_back(i);
            }
        }
        for (int i=0;i<vars_offsets.size();i++){
            auto e = expr::make(loop->find_define(vars_offsets[i])->attrs["rvalue"]);
            auto ee = mlu_trace_and_expand(c.get(), e.get());
            string simp=ee->simplify()->to_string();
            if (simp!="0" && simp!=new_id) return;
            calc_idxs.push_back(vars_offsets[i]);
            is_vec.push_back(simp==new_id);
        }
        if (c->type == "define"){
            string str=c->attrs["lvalue"];
            if (str.find("i")!=-1) continue;
            define_vars.push_back(c->attrs["lvalue"]);
        }
        calc_code.push_back(code);
    }

    for (int i=0;i<calc_code.size();i++){
        for (int j=0;j<calc_idxs.size();j++){
            string sidx="["+calc_idxs[j]+"]";
            while (calc_code[i].find(sidx)!=-1){
                calc_code[i].replace(calc_code[i].find(sidx),sidx.size(),"_"+S(is_vec[j]));
            }
        }
        if (calc_code[i].find("std::numeric_limits<")!=-1) return;

        vector<unique_ptr<expr::Expr>> res;
        ASSERT(expr::match(expr::make(calc_code[i]).get(), expr::make("a=b").get(), {"a","b"}, {}, res));
        string dst = res.at(0)->to_string();
        auto rval = res.at(1)->clone();
        if (dst.find("check_overflow")!=-1) return;
        
        if (!bang_dfs(func, dst, rval, define_vars, bang_code, new_range)) return;
    }
    for (int i=0;i<define_vars.size();i++) {
        func->push_back("__nram__ float "+define_vars[i]+"["+S(nram_space)+"];");
    }
    for (int i=0;i<bang_code.size();i++){
        if (bang_code[i].find("for")==0) {
            loop->push_back("a=1;", &loop->before);
            loop->before.back()->get_attr("code") = bang_code[i];
        } else {
            loop->push_back(bang_code[i], &loop->before);
        }
    }
    loop->push_front("break;", &loop->children);
    auto& fat=loop->father;
    fat->push_back("int fake_range=("+new_range+"=="+S(nram_space)+") ? "+S(nram_space)+" : int("+new_range+"/1024+1)*1024;");
    for (auto& c : loop->before){
        fat->push_back(c->clone());
    }
    for (auto& c : loop->after){
        fat->push_back(c->clone());
    }
    loop->erase();
    func->move_loop_back();
    func->remove_all_unused();
}

void MLUPass::run() {
    if (!use_mlu) return;
    
    vector<string> var_names;
    vector<int> is_input;
    vector<Var*> vars;
    int func_num=0;
    for (auto& c : ir->children) {
        string& name = c->get_attr("lvalue");
        if (name.find("op")==0 && c->get_attr("dtype")=="auto* __restrict__"){
            uint sop_id, sopvar_id;
            Op* sop;
            Var* svar;
            string sname = name.substr(0, name.length()-1);
            pm->oc->get_op_var_by_name(sname, sop_id, sopvar_id, sop, svar);
            var_names.push_back(name);
            vars.push_back(svar);
            if (sopvar_id<sop->inputs().size())
                is_input.push_back(1);
            else
                is_input.push_back(0);
        }
    }

    if (var_names.size()==0) return;

    ir->push_front("#include <bang.h>", &ir->before);
    ir->push_front("#include <bang_pipeline.h>", &ir->before);
    ir->push_front("#include \"mlu_warper.h\"", &ir->before);
    
    vector<string> strs;

    strs.push_back("cnrtDim3_t dim = cnrtDim3_t({64, 1, 1});");
    strs.push_back("cnrtFunctionType_t ktype = CNRT_FUNC_TYPE_BLOCK;");
    

    for (int i=strs.size()-1;i>=0;i--){
        ir->push_front(strs[i], &ir->children);
    }

    strs.clear();
    
    int len=ir->children.size();
    for (int i=strs.size()-1;i>=0;i--){
        ir->insert(len-func_num, strs[i], &ir->children);
    }
    ir->push_back("JT_MLU_CHECK(cnrtGetLastErr());", &ir->children);
    
    int num = 0;
    int choice = nram_space;
    // int split_size = std::max(1, choice);
    KernelIR *loop;
    KernelIR *loop_father;
    string i, j;
    for (auto& c : ir->before) {
        if (c->get_attr("dtype") == "__mlu_global__ void") {
            KernelIR* sc=&(*c);
            vector<string> ids;
            while (1){
                int bo=0;
                for (auto& cc : sc->children)
                if (cc->type=="loop") {
                    ids.push_back(cc->inner[0]->get_attr("lvalue"));
                    bo=1;
                    loop_father = sc;
                    sc = &(*cc);
                    loop = sc;
                    break;
                }
                if (!bo) break;
                else num++;
            }
            j = loop->get_attr("loop_id");

            vector<string> vars;
            vector<string> types;
            vector<int> is_input_infunc;
            for (auto& cc : c->inner)
            if (cc->get_attr("lvalue")[cc->get_attr("lvalue").size()-1]=='p'){
                vars.push_back(cc->get_attr("lvalue"));
                types.push_back(cc->get_attr("dtype"));
                int si=types.size()-1;
                types[si]=types[si].substr(0,types[si].size()-14);
                for (int i=0;i<var_names.size();i++)
                if (var_names[i]==vars[vars.size()-1])
                    is_input_infunc.push_back(is_input[i]);
                ASSERT(is_input_infunc.size()==vars.size());
            }

            // __memcpy(op9_zp + id0*xxx, op9_zp_nram, 4096 * sizeof(float), NRAM2GDRAM);
            vector<string> nram_vars;
            vector<string> nram_types;
            add_memcpy(loop_father, loop, vars, types, is_input_infunc, "id"+j, nram_vars, nram_types);
            // LOGir << "nram_vars" << nram_vars;
            // LOGir << "types" << types;
            ASSERT(nram_vars.size()==nram_types.size());

            //__nram__ op0_Tx op0_xp_nram[4096];
            for (int i=0;i<nram_vars.size();i++){
                c->push_back("__nram__ "+nram_types[i]+" "+nram_vars[i]+"_nram["+S(choice)+"];");
            }

            // replace idxx to 0
            for (auto& cc : loop->children) {
                for (int i=0;i<ids.size()-1;i++)
                    cc->replace({{ids[i], "0"}}, true);
                for (int i=0;i<vars.size();i++)
                    cc->replace({{vars[i], vars[i]+"_nram"}}, true);
            }
            
            // set tmp nram var to 0
            for (int i=0;i<vars.size();i++)
            if (!is_input_infunc[i])
                for (int j=0;j<nram_vars.size();j++)
                if (nram_vars[j]==vars[i]){
                    loop->push_back("__nramset("+vars[i]+"_nram, "+S(nram_space)+", 0);", &loop->before);
                }
            convert_to_bang(c, loop, vars, "id"+j, "range"+j);
            
            c->move_loop_back();
        }
    }
}

} // jittor