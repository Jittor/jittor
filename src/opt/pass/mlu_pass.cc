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

namespace jittor {

void MLUPass::run() {
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
            var_names.push_back(sname);
            vars.push_back(svar);
            if (sopvar_id<sop->inputs().size())
                is_input.push_back(1);
            else
                is_input.push_back(0);
        }

        for (auto& cc : c->children) {
            if (cc->has_attr("code")) {
                string& ccstr = cc->get_attr("code");
                if (ccstr.find("<<<")!=-1){
                    ++func_num;
                    while (ccstr.find("p,")!=-1) {
                        ccstr.replace(ccstr.find("p,"),1,"p_mlu");
                    }
                }
            }
        }
    }

    if (var_names.size()==0) return;

    ir->push_front("#include <bang.h>", &ir->before);
    ir->push_front("#include <bang_pipeline.h>", &ir->before);
    ir->push_front("#include \"mlu_warper.h\"", &ir->before);
    
    vector<string> strs;
    // strs.push_back("cnrtDev_t dev;");
    // strs.push_back("cnrtQueue_t queue;");
    // strs.push_back("cnrtInit(0);");
    // strs.push_back("cnrtGetDeviceHandle(&dev, 0);");
    // strs.push_back("cnrtSetCurrentDevice(dev);");
    // strs.push_back("cnrtCreateQueue(&queue);");

    strs.push_back("cnrtDim3_t dim = cnrtDim3_t({1, 1, 1});");
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
    for (int i=0;i<var_names.size();i++){
        string str_type = var_names[i];
        string name_mlu = var_names[i]+"p_mlu";
        string name_p = var_names[i]+"p";
        str_type.insert(str_type.find("_")+1,"T");

        strs.push_back(str_type+"* __restrict__ "+name_mlu+";");
    }

    for (int i=0;i<var_names.size();i++){
        string str_type = var_names[i];
        str_type.insert(str_type.find("_")+1,"T");
        string str_size = "size_t size_"+var_names[i]+" = sizeof("+str_type+")";
        for (int j=0;j<vars[i]->shape.size();j++)
            str_size += "*"+var_names[i]+"->shape["+(char)(j+48)+"]";
        str_size += ";";
        strs.push_back(str_size);
    }

    for (int i=0;i<var_names.size();i++){
        string name_mlu = var_names[i]+"p_mlu";
        string name_p = var_names[i]+"p";
        strs.push_back("JT_MLU_CHECK(cnrtMalloc((void **)&"+name_mlu+", size_"+var_names[i]+"));");
        if (is_input[i])
            strs.push_back("JT_MLU_CHECK(cnrtMemcpy("+name_mlu+", "+name_p+", size_"+var_names[i]+", CNRT_MEM_TRANS_DIR_HOST2DEV));");
    }
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
    ir->push_back("JT_MLU_CHECK(cnrtSyncQueue(queue));", &ir->children);

    for (int i=0;i<var_names.size();i++){
        string str_type = var_names[i];
        string name_mlu = var_names[i]+"p_mlu";
        string name_p = var_names[i]+"p";
        if (!is_input[i])
            ir->push_back("JT_MLU_CHECK(cnrtMemcpy("+name_p+", "+name_mlu+", size_"+var_names[i]+", CNRT_MEM_TRANS_DIR_DEV2HOST));", &ir->children);
    }
    for (int i=0;i<var_names.size();i++){
        string name_mlu = var_names[i]+"p_mlu";
        ir->push_back("JT_MLU_CHECK(cnrtFree("+name_mlu+"));", &ir->children);
    }
    // ir->push_back("JT_MLU_CHECK(cnrtDestroyQueue(queue));", &ir->children);
    // ir->push_back("cnrtDestroy();", &ir->children);
}

} // jittor