// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
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
            // tid + loop_cnt * tnum
            // auto new_expr = expr::make_op("+",
            //     expr::make(rvalue),
            //     expr::make_op("*", 
            //         expr::make("loop_cnt"),
            //         r2.at(0)->clone()
            //     )
            // );
            // c->swap(new_expr.get());
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

void MLUPass::add_memcpy(KernelIR* loop_father, KernelIR* loop, vector<string> vars, vector<int> is_input, string new_id) {
    LOGir << "vars" << vars;
    vector<int> solved;
    for (int i=0;i<is_input.size();i++) solved.push_back(0);
    // return;
    for (auto& c : loop->children) {
        if (c->type != "" && c->type != "define") continue;
        LOGir << "c" << c;
        string code;
        LOGir << "c->attrs[rvalues]" << c->attrs["rvalue"];
        if (c->type == "define") code = c->attrs["lvalue"]+" = "+c->attrs["rvalue"];
        else {
            if (!c->has_attr("code")) continue;
            code = c->attrs["code"];
            code = code.substr(0, code.size()-1); // remove ';'
        }
        LOGir << "code" << code;

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
        LOGir << "vars_offsets" << vars_offsets;
        for (int k=0;k<vars_offsets.size();k++){/*
            auto e = expr::make(code);
            vector<unique_ptr<expr::Expr>> results;
            auto target = expr::make("a=b");
            LOGir << "e" << e;
            if (!expr::match(e.get(), target.get(), {"a", "b"}, {}, results))
                continue;
            LOGir << "results" << results;

            vector<unique_ptr<expr::Expr>> save_var;
            vector<unique_ptr<expr::Expr>> ptr_and_offset;
            // if (expr::match(results[0].get(), expr::make("a[b]").get(), {"a", "b"}, {}, ptr_and_offset))
            //     for (int i=0;i<ptr_and_offset.size();i++) save_var.push_back(ptr_and_offset[i]->clone());

            LOGir<<"666";
            if (expr::match(results[1].get(), expr::make("a[b]").get(), {"a", "b"}, {}, ptr_and_offset))
                for (int i=0;i<ptr_and_offset.size();i++) save_var.push_back(ptr_and_offset[i]->clone());
            
            if (save_var.size()==0) continue;

            LOGir << "save_var" << save_var;
            int vars_idx=-1;
            for (int i=0;i<vars.size();i++)
            if (vars[i]==ptr_and_offset.at(0)->to_string()) vars_idx=i;
            ASSERT(vars_idx!=-1);*/
            int vars_idx = vars_idxs[k];
            if (solved[vars_idx]) continue;
            solved[vars_idx] = 1;
            auto offset = mlu_trace_and_expand(c.get(), expr::make(vars_offsets[k]).get())
                ->simplify();
            LOGir << "offset" << offset->to_string(1);

            string idx_code = offset->to_string(1);
            string cpy_size="1";
            if (idx_code.find(new_id)!=-1) cpy_size="range"+new_id.substr(2);
            while (idx_code.find(new_id)!=-1){
                idx_code.replace(idx_code.find(new_id), new_id.size(), "0");
            }
            LOGir << "idx_code" << idx_code;
            if (is_input[vars_idx]){
                idx_code = "__memcpy("+vars[vars_idx]+"_nram, "+vars[vars_idx]+" + "+idx_code+", "+cpy_size+" * sizeof(float), GDRAM2NRAM);";
                loop->push_front(idx_code, &loop->before);
            }else{
                idx_code = "__memcpy("+vars[vars_idx]+" + "+idx_code+", "+vars[vars_idx]+"_nram, "+cpy_size+" * sizeof(float), NRAM2GDRAM);";
                loop->push_front(idx_code, &loop->after);
            }

            // vector<unique_ptr<expr::Expr>> idxs;
            // expr::match(expr::make(offset->to_string(1)).get(), expr::make("a+b").get(), {"a", "b"}, {}, idxs);
            // LOGir << "idxs" << idxs;
        }
    }
    LOGir << "solved" << solved;
    // return solved;
    for (int i=0;i<solved.size();i++) ASSERT(solved[i]==1);
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

        // for (auto& cc : c->children) {
        //     if (cc->has_attr("code")) {
        //         string& ccstr = cc->get_attr("code");
        //         if (ccstr.find("<<<")!=-1){
        //             ++func_num;
        //             while (ccstr.find("p,")!=-1) {
        //                 ccstr.replace(ccstr.find("p,"),1,"p_mlu");
        //             }
        //         }
        //     }
        // }
    }

    if (var_names.size()==0) return;

    ir->push_front("#include <bang.h>", &ir->before);
    ir->push_front("#include <bang_pipeline.h>", &ir->before);
    ir->push_front("#include \"mlu_warper.h\"", &ir->before);
    
    vector<string> strs;
    // strs.push_back("cnrtDev_t dev;");
    // strs.push_back("cnrtQueue_t mlu_queue;");
    // strs.push_back("cnrtInit(0);");
    // strs.push_back("cnrtGetDeviceHandle(&dev, 0);");
    // strs.push_back("cnrtSetCurrentDevice(dev);");
    // strs.push_back("cnrtCreateQueue(&mlu_queue);");

    strs.push_back("cnrtDim3_t dim = cnrtDim3_t({16, 1, 1});");
    strs.push_back("cnrtFunctionType_t ktype = CNRT_FUNC_TYPE_BLOCK;");
    

    for (int i=strs.size()-1;i>=0;i--){
        ir->push_front(strs[i], &ir->children);
    }

    strs.clear();
    // for (int i=0;i<var_names.size();i++){
    //     string str_type = var_names[i];
    //     string name_mlu = var_names[i]+"p_mlu";
    //     string name_p = var_names[i]+"p";
    //     str_type.insert(str_type.find("_")+1,"T");

    //     strs.push_back(str_type+"* __restrict__ "+name_mlu+";");
        
    //     string str_size = "size_t size_"+var_names[i]+" = sizeof("+str_type+")";
    //     for (int j=0;j<vars[i]->shape.size();j++)
    //         str_size += "*"+var_names[i]+"->shape["+(char)(j+48)+"]";
    //     str_size += ";";
    //     strs.push_back(str_size);
        
    //     strs.push_back("JT_MLU_CHECK(cnrtMalloc((void **)&"+name_mlu+", size_"+var_names[i]+"));");
    //     if (is_input[i])
    //         strs.push_back("JT_MLU_CHECK(cnrtMemcpy("+name_mlu+", "+name_p+", size_"+var_names[i]+", CNRT_MEM_TRANS_DIR_HOST2DEV));");
    // }
    // for (int i=0;i<var_names.size();i++){
    //     string str_type = var_names[i];
    //     string name_mlu = var_names[i]+"p_mlu";
    //     string name_p = var_names[i]+"p";
    //     str_type.insert(str_type.find("_")+1,"T");

    //     strs.push_back(str_type+"* __restrict__ "+name_mlu+";");
    // }

    // for (int i=0;i<var_names.size();i++){
    //     string str_type = var_names[i];
    //     str_type.insert(str_type.find("_")+1,"T");
    //     string str_size = "size_t size_"+var_names[i]+" = sizeof("+str_type+")";
    //     for (int j=0;j<vars[i]->shape.size();j++)
    //         str_size += "*"+var_names[i]+"->shape["+(char)(j+48)+"]";
    //     str_size += ";";
    //     strs.push_back(str_size);
    // }

    // for (int i=0;i<var_names.size();i++){
    //     string name_mlu = var_names[i]+"p_mlu";
    //     string name_p = var_names[i]+"p";
    //     strs.push_back("JT_MLU_CHECK(cnrtMalloc((void **)&"+name_mlu+", size_"+var_names[i]+"));");
    //     if (is_input[i])
    //         strs.push_back("JT_MLU_CHECK(cnrtMemcpy("+name_mlu+", "+name_p+", size_"+var_names[i]+", CNRT_MEM_TRANS_DIR_HOST2DEV));");
    // }
    // strs.push_back("op0_Tx* __restrict__ op0_xp_mlu;");
    // strs.push_back("op0_Tx* __restrict__ op0_yp_mlu;");
    // strs.push_back("op0_Tx* __restrict__ op0_zp_mlu;");
    
    // strs.push_back("size_t size = op0_x->shape[0]*op0_x->shape[1]*sizeof(op0_Tx);");
    // strs.push_back("JT_MLU_CHECK(cnrtMalloc((void **)&op0_xp_mlu, size));");
    // strs.push_back("JT_MLU_CHECK(cnrtMalloc((void **)&op0_yp_mlu, size));");
    // strs.push_back("JT_MLU_CHECK(cnrtMalloc((void **)&op0_zp_mlu, size));");

    // strs.push_back("JT_MLU_CHECK(cnrtMemcpy(op0_xp_mlu, op0_xp, size, CNRT_MEM_TRANS_DIR_HOST2DEV));");
    // strs.push_back("JT_MLU_CHECK(cnrtMemcpy(op0_yp_mlu, op0_yp, size, CNRT_MEM_TRANS_DIR_HOST2DEV));");

    int len=ir->children.size();
    for (int i=strs.size()-1;i>=0;i--){
        ir->insert(len-func_num, strs[i], &ir->children);
    }
    ir->push_back("JT_MLU_CHECK(cnrtGetLastErr());", &ir->children);
    ir->push_back("JT_MLU_CHECK(cnrtSyncQueue(mlu_queue));", &ir->children);
    
    int num = 0;
    int choice = 4096;
    int split_size = std::max(1, choice);
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
            // LOGir << "ids" << ids << loop_father;
            i = loop->get_attr("loop_id");
            j = i+"0";
            c->split_loop(i, j);
            c->push_back(loop->attrs["dtype"]+" stride"+i+" = "+S(split_size)+";");
            
            for (auto& cc : sc->children)
            if (cc->type=="loop") {
                loop_father = loop;
                loop = &(*cc);
            }

            vector<string> vars;
            vector<string> types;
            vector<int> is_input_infunc;
            for (auto& cc : c->inner)
            if (cc->get_attr("lvalue")[cc->get_attr("lvalue").size()-1]=='p'){
                vars.push_back(cc->get_attr("lvalue"));
                types.push_back(cc->get_attr("dtype"));
                for (int i=0;i<var_names.size();i++)
                if (var_names[i]==vars[vars.size()-1])
                    is_input_infunc.push_back(is_input[i]);
                ASSERT(is_input_infunc.size()==vars.size());
            }

            // __memcpy(op9_zp + id0*xxx, op9_zp_nram, 4096 * sizeof(float), NRAM2GDRAM);
            LOGir << "before add_memcpy" << c;
            add_memcpy(loop_father, loop, vars, is_input_infunc, "id"+j);
            // LOGir << vars << types;

            //__nram__ op0_Tx op0_xp_nram[4096];
            for (int i=0;i<vars.size();i++){
                c->push_back("__nram__ "+types[i].substr(0,types[i].size()-14)+" "+vars[i]+"_nram["+S(choice)+"];");
            }

            // replace idxx to 0
            for (auto& cc : loop->children) {
                for (int i=0;i<ids.size();i++)
                    cc->replace({{ids[i], "0"}}, true);
                for (int i=0;i<vars.size();i++)
                    cc->replace({{vars[i], vars[i]+"_nram"}}, true);
            }
            
            c->move_loop_back();
            LOGir << c;
            break;
        }
    }
    // for (int i=0;i<var_names.size();i++){
    //     string str_type = var_names[i];
    //     string name_mlu = var_names[i]+"p_mlu";
    //     string name_p = var_names[i]+"p";
    //     if (!is_input[i])
    //         ir->push_back("JT_MLU_CHECK(cnrtMemcpy("+name_p+", "+name_mlu+", size_"+var_names[i]+", CNRT_MEM_TRANS_DIR_DEV2HOST));", &ir->children);
    // }
    // for (int i=0;i<var_names.size();i++){
    //     string name_mlu = var_names[i]+"p_mlu";
    //     ir->push_back("JT_MLU_CHECK(cnrtFree("+name_mlu+"));", &ir->children);
    // }
    // ir->push_back("JT_MLU_CHECK(cnrtDestroyQueue(mlu_queue));", &ir->children);
    // ir->push_back("cnrtDestroy();", &ir->children);
}

} // jittor