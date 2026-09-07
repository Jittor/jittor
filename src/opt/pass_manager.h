// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <typeindex>
#include "common.h"
#include "fused_op.h"
#include "op_compiler.h"
#include "opt/kernel_ir.h"
#include "opt/pass/pass.h"

namespace jittor {

DECLARE_FLAG(string, exclude_pass);
DECLARE_FLAG(string, log_op_hash);

struct PassManager {
    OpCompiler* oc;
    KernelIR all;
    KernelIR* main_ir;
    // Keyed by the pass's C++ type, not by its name: two passes used to ship
    // the same name ("expand_empty_block" for both ExpandEmptyBlockPass and
    // UnrollPass), and because this is an unordered_map filled with emplace,
    // the second registration was dropped and get_pass returned the first pass
    // under a C-style cast to the wrong type. A type key cannot collide, and
    // the cast that follows it is exact by construction.
    //
    // A pass that runs more than once (SolveConflictDefinePass runs three
    // times) now leaves the most recent instance here; emplace used to leave
    // the first. Nothing reads those passes today, and "the state after the
    // last run" is the answer a caller would want.
    unordered_map<std::type_index, Pass*> pass_map;
    vector<unique_ptr<Pass>> finished_passes;
    // KernelIR attributes available at this point in the pipeline: the ones the
    // parser sets, plus what every pass declared so far says it writes. A pass
    // that reads something not in here is a pipeline ordering bug, and this is
    // where it is caught -- see Pass::reads.
    unordered_set<string> produced;

    PassManager(OpCompiler* oc);
    // run and store a pass. `enabled` false declares the pass's attribute
    // contract without running it, so a pass that only runs under some
    // compilers still counts as the producer of what it writes.
    template <class T> void run_pass(bool enabled=true);
    // every attribute `pass` declares it reads must already be produced
    void check_attr_contract(Pass* pass);
    // get a pass that already ran, nullptr if it did not (it may have been
    // excluded); the type is the key, so no downcast is involved
    template <class T> T* get_pass();

    bool check(Pass* pass);

    void run_passes();
    
};

template <class T>
void PassManager::run_pass(bool enabled) {
    auto pass = std::make_unique<T>();
    check_attr_contract(pass.get());
    for (auto w : pass->writes) produced.insert(w);
    if (!enabled) return;
    if (!check(pass.get())) {
        LOGvvv << "exclude pass" << pass->name;
        return;
    }
    LOGvvv << "run pass" << pass->name;
    pass->init(this);
    pass->run();
    LOGvvvv << "Kernel IR after pass" << pass->name << ":\n"
        << main_ir->to_string(0, true);
        
    if (log_op_hash.size() && log_op_hash == oc->op->get_hash_name())
        LOGi << "hash mach:" << log_op_hash << "pass:" << pass->name 
        << main_ir->to_string(0, true);
    pass_map[std::type_index(typeid(T))] = pass.get();
    finished_passes.push_back(move(pass));
}

template <class T>
T* PassManager::get_pass() {
    auto iter = pass_map.find(std::type_index(typeid(T)));
    if (iter == pass_map.end()) return nullptr;
    return static_cast<T*>(iter->second);
}

} // jittor
