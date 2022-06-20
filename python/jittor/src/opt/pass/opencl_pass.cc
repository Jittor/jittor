// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/opencl_pass.h"
#include "opt/pass/loop_var_analyze_pass.h"
#include "opt/expr.h"
#include "opencl_warper.h"

namespace jittor {
    
void OpenclPass::add_parentheses(string& str) {
    int last=-1;
    for (int i=0;i<str.size();i++){
        if (('0'<=str[i] && str[i]<='9') || ('a'<=str[i] && str[i]<='z') || ('A'<=str[i] && str[i]<='Z') || str[i]=='_') continue;
        if (last<i-1) {
            string ss=str.substr(last+1,i-last-1);
            if (ss.find("_T")!=-1 && !(str[last]=='(' && str[i]==')')){
                str = str.insert(last+1,"(");
                str = str.insert(i+1,")");
                i += 2;
            }
        }
        last = i;
    }
    if (last<str.size()-1) {
        string ss=str.substr(last+1);
        if (ss.find("_T")!=-1){
            str = str.insert(last+1,"(");
            str = str.insert(str.size(),")");
        }
    }
}

void OpenclPass::solve_kernel(KernelIR* c) {
    for (auto& cc : c->inner) {
        string &str = cc->get_attr("dtype");
        if (str.find("_T")!=-1 && str.find("*")!=-1)
            str = "__global " + str;
        if (str.find("float32")!=-1)
            str=str.replace(str.find("float32"), 7, "float");
        if (str.find("int32")!=-1)
            str=str.replace(str.find("int32"), 5, "int");
    }
    for (auto& cc : c->children) {
        if (cc->get_attr("lvalue")=="thread_id")
            cc->get_attr("rvalue") = "get_global_id(0)";
    }
    
    // fix auto
    KernelIR* sc=&(*c);
    KernelIR *loop;
    vector<string> ids;
    while (1){
        int bo=0;
        for (auto& cc : sc->children)
        if (cc->type=="loop") {
            ids.push_back(cc->inner[0]->get_attr("lvalue"));
            bo=1;
            sc = &(*cc);
            loop = sc;
            break;
        }
        if (!bo) break;
    }
    for (auto& cc : loop->children) {
        if (cc->get_attr("dtype")!="auto") continue;
        string lvalue = cc->get_attr("lvalue");
        string &dtype = cc->get_attr("dtype");

        if (lvalue.find("id")!=-1){
            dtype="int";
            continue;
        }
        uint op_id, opvar_id;
        Op* op;
        Var* var;
        pm->oc->get_op_var_by_name(lvalue.substr(0, lvalue.size()-1), op_id, opvar_id, op, var);
        string tar=var->dtype().to_cstring();
        if (tar=="float32" || tar=="float") dtype="float";
        else if (tar=="int32" || tar=="int") dtype="int";
        else if (tar=="bool") dtype="bool";
        else ASSERT(false);
    }

    for (auto& cc : c->children) {
        if (cc->get_attr("dtype")!="auto") continue;
        string lvalue = cc->get_attr("lvalue");
        string &dtype = cc->get_attr("dtype");
        if (lvalue.find("stride")!=-1) dtype="int";
    }
    
    // fix type convert
    for (auto& cc : loop->children) {
        if (cc->get_attr("code")!="") add_parentheses(cc->get_attr("code"));
        else if (cc->get_attr("rvalue")!="") add_parentheses(cc->get_attr("rvalue"));
        else ASSERT(false);
    }
}

void OpenclPass::run() {
    if (!use_opencl) return;
    // add some header
    ir->push_front("#include \"opencl_warper.h\"", &ir->before);
    ir->push_front("#include <CL/cl.h>", &ir->before);

    // modify kernel to fit opencl
    string type_defs="";
    for (auto& c : all->children) {
        // LOGir << "before" << c << "dtype" << c->get_attr("dtype") << "lvalue" << c->get_attr("lvalue") << "rvalue" << c->get_attr("rvalue");
        string str = c->to_string();
        if (str.find("#define")!=0) continue;
        if (c->get_attr("lvalue").find("op")!=0) continue;
        if (str.find("float32")!=-1)
            str = str.replace(str.find("float32"), 7, "float");
        if (str.find("int32")!=-1)
            str = str.replace(str.find("int32"), 5, "int");
        type_defs += str;
    }
    type_defs += "\n";

    // convert kernel to string
    int len=ir->before.size();
    for (int i=0;i<len;i++){
        auto& c=ir->before[i];
        if (c->to_string().find("__kernel")==-1) continue;
        solve_kernel(&(*c));
        string code = c->to_string();
        code= "const char* code_"+c->get_attr("lvalue")+" = R\"(\n"+
        type_defs + 
        code +
        ")\";";
        ir->insert(i, code, false, &ir->before);
        ir->before[i+1]->erase();
    }

    // modify type of array
    for (auto& c : ir->children) {
        if (c->to_string().find("->ptr<")==-1) continue;
        if (c->to_string().find("outputd")!=-1) continue;
        string &str = c->get_attr("rvalue");
        int l=str.find("->ptr<")+6;
        int r=str.find(">", l);
        str=str.replace(l,r-l,"cl_mem");
    }

    for (auto& c : ir->children) {
        if (c->children.size()==0) continue;

        // get parameters
        int len = c->children.size();
        auto& fc = c->children[len-1];
        string str = fc->to_string();
        string func_name=str.substr(0,str.find("<<<"));
        int pos=str.find("(")+1;
        vector<string> paras;
        while (str.find(",",pos)!=-1) {
            paras.push_back(str.substr(pos,str.find(",",pos)-pos));
            pos=str.find(",",pos)+1;
        }
        paras.push_back(str.substr(pos,str.size()-pos-3));

        // set parameters & run kernel
        c->children[len-1]->erase();
        c->children[len-2]->erase();
        c->children[len-3]->erase();
        for (int i=0;i<paras.size();i++){
            string tp;
            if (paras[i].find("op")==0) {
                if (paras[i].find("outputd")!=-1) {
                    string df=ir->find_define(paras[i])->to_string();
                    int l=df.find("<"), r=df.find(">", df.find("<"));
                    ASSERT(df.find(">")!=-1 && df.find("<")!=-1);
                    df=df.substr(l+1, r-l-1);
                    if (df=="float32" || df=="float") tp="cl_float";
                    else if (df=="int32" || df=="int") tp="cl_int";
                    else ASSERT(false);
                } else if (paras[i].find("rcount")!=-1) {
                    string df=ir->find_define(paras[i])->get_attr("dtype");
                    df=all->find_define(df)->get_attr("rvalue");
                    if (df=="float32" || df=="float") tp="cl_float";
                    else if (df=="int32" || df=="int") tp="cl_int";
                    else ASSERT(false);
                } else if (paras[i].find("shape")!=-1) tp="cl_int";
                else tp="cl_mem";
            } else {
                tp="cl_int";
            }

            c->push_back("clSetKernelArg(adder, "+S(i)+", sizeof("+tp+"), "+(tp=="cl_mem" ? "" : "&")+paras[i]+");", &c->children, true);
        }
        c->push_back("size_t work_size = thread_num;", &c->children, true);
        c->push_back("err = clEnqueueNDRangeKernel(opencl_queue, adder, 1, 0, &work_size, 0, 0, 0, 0);", &c->children, true);
        c->push_back("ASSERT(err==CL_SUCCESS);");

        // build & destory kernel
        vector<string> build_codes;
        build_codes.push_back("static cl_program program = clCreateProgramWithSource(opencl_context, 1, &code_"+func_name+", 0, 0);");
        build_codes.push_back("static cl_int err = clBuildProgram(program, 0, 0, 0, 0, 0);");
        build_codes.push_back("ASSERT(err==CL_SUCCESS);");
        build_codes.push_back("static cl_kernel adder = clCreateKernel(program, \""+func_name+"\", &err);");
        build_codes.push_back("ASSERT(err==CL_SUCCESS);");
        for (int i=build_codes.size()-1;i>=0;i--)
            c->push_front(build_codes[i], &c->children, true);
    }
}

} // jittor