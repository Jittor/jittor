// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/loop_to_func_pass.h"

namespace jittor {

DECLARE_FLAG(string, cc_type);

void LoopToFuncPass::run() {
    auto choice = op->get_loop_option("parallel");
    bool is_cuda = op->flag(OpFlags::_cuda);
    if (is_cuda) choice=1;
    if (cc_type=="clang") choice=1;
    if (!choice) return;
    int func_num=0;
    string hash_name = op->get_hash_name();
    
    ir->push_back("using namespace jittor;", &ir->before);
    if ((cc_type=="icc" || cc_type=="g++") && choice)
        // icc will failed if not inline when parallel
        ir->push_back("#define INLINE_FUNC inline static void ", &ir->before);
    else
        ir->push_back("#define INLINE_FUNC __attribute__((always_inline)) static void ", &ir->before);
    for (uint i=0; i<ir->children.size(); i++) {
        auto& c = ir->children[i];
        if (c->type != KernelIRType::loop) continue;
        if (c->has_attr(kir::vectorized) || c->has_attr(kir::unrolled) || c->has_attr(kir::resplited))
            continue;
        if (c->before.size())
            continue;
        if (c->inner.size() < 3)
            continue;
        if (!c->has_attr(kir::lvalue))
            continue;
        if (c->has_attr(kir::raw))
            continue;
        
        // func definition
        ir->push_back("INLINE_FUNC func_"+hash_name+"_"+S(func_num++)+"() {}", &ir->before);
        auto& func = ir->before.back();
        
        // generate function arguments
        vector<KernelIR*> args;
        for (auto& d : ir->children) {
            if (d->has_attr(kir::raw)) continue;
            if (d->type == KernelIRType::loop) break;
            if (d->has_attr(kir::code) && startswith(d->attrs[kir::code], "func")) break;
            if (d->type == KernelIRType::define) {
                if (d->has_attr(kir::rvalue)) {
                    auto& rvalue = d->attrs[kir::rvalue];
                    auto& dtype = d->attrs[kir::dtype];
                    if (endswith(d->attrs[kir::lvalue], "_value") ||
                        endswith(d->attrs[kir::lvalue], "_outputv")) {
                        args.push_back(d.get());
                        continue;
                    }
                    if (rvalue.find("ops") != string::npos)
                        continue;
                    if (dtype=="Var*")
                        continue;
                    if (dtype=="Op*")
                        continue;
                    if (rvalue.find("->") != string::npos ||
                        dtype.find("*") != string::npos) {
                        args.push_back(d.get());
                        continue;
                    }
                }
            }
            func->push_back(d->clone());
        }
        func->push_back(c->clone());
        string func_call = func->attrs[kir::lvalue]+"(";
        for (auto arg : args) {
            if (arg != args.front())
                func_call += ',';
            auto dtype = arg->attrs[kir::dtype];
            auto& lvalue = arg->attrs[kir::lvalue];
            auto& rvalue = arg->attrs[kir::rvalue];
            if (startswith(dtype, "auto")) {
                    // resolve auto
                if (rvalue.find("<") == -1 || rvalue.find(">") == -1) {
                    //resolve auto xxx = ((T*)xxx)[0];
                    std::vector<string> temp = split(split(rvalue, "*)", 2).at(0), "(", 0);
                    dtype = temp[temp.size() - 1] + dtype.substr(4);
                } else {
                    dtype = split(split(rvalue, "<", 2).at(1), ">", 2).at(0) + dtype.substr(4);
                }
            }
            func_call += arg->attrs[kir::lvalue];
            func->push_back(dtype+" "+lvalue+";", &func->inner);
        }
        func_call += ");";
        c->erase();
        ir->insert(i, func_call);
        
        auto& fc = ir->children[i];
        fc->attrs[kir::loop_func] = func->attrs[kir::lvalue];
    }
    #ifdef __APPLE__
    ir->remove_all_unused();
    #endif
}

} // jittor