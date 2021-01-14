// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "fused_op.h"
#include "op_compiler.h"
#include "opt/kernel_ir.h"
#include "opt/pass/pass.h"

namespace jittor {

DECLARE_FLAG(string, exclude_pass);

struct PassManager {
    OpCompiler* oc;
    KernelIR all;
    KernelIR* main_ir;
    unordered_map<string, Pass*> pass_map;
    vector<unique_ptr<Pass>> finished_passes;

    PassManager(OpCompiler* oc);
    // run and store a pass
    template <class T> void run_pass();
    // get a pass by pass name, return nullptr if not found
    template <class T> T* get_pass(const string& name);

    bool check(Pass* pass);

    void run_passes();
    
};

template <class T>
void PassManager::run_pass() {
    auto pass = std::make_unique<T>();
    if (!check(pass.get())) {
        LOGvvv << "exclude pass" << pass->name;
        return;
    }
    LOGvvv << "run pass" << pass->name;
    pass->init(this);
    pass->run();
    LOGvvvv << "Kernel IR after pass" << pass->name << ":\n"
        << main_ir->to_string(0, true);
    pass_map.emplace(pass->name, pass.get());
    finished_passes.push_back(move(pass));
}

template <class T>
T* PassManager::get_pass(const string& name) {
    auto iter = pass_map.find(name);
    if (iter == pass_map.end()) return nullptr;
    return (T*)iter->second;
}

} // jittor
