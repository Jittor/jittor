// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cctype>
#include "var.h"
#include "opt/pass_manager.h"
#include "opt/pass/use_movnt_pass.h"

namespace jittor {

DECLARE_FLAG(string, cc_type);

static bool split_output_store(const string& code, string& target, string& value) {
    auto eq = code.find('=');
    if (eq == string::npos || (eq + 1 < code.size() && code[eq+1] == '=') ||
        (eq && string("=!<>+-%&|^*/").find(code[eq-1]) != string::npos))
        return false;
    target = code.substr(0, eq);
    value = code.substr(eq + 1);
    while (target.size() && isspace(target.back())) target.pop_back();
    auto first = target.find_first_not_of(" \t");
    if (first == string::npos) return false;
    target = target.substr(first);
    first = value.find_first_not_of(" \t");
    if (first == string::npos) return false;
    value = value.substr(first);
    while (value.size() && (isspace(value.back()) || value.back() == ';'))
        value.pop_back();
    auto pointer = target.find("_zp[");
    return pointer != string::npos && target.back() == ']';
}

void UseMovntPass::run() {
    if (!op->get_loop_option("use_movnt") || op->flag(OpFlags::_cuda) ||
        cc_type != "clang") return;

    bool changed = false;
    vector<KernelIR*> queue({all});
    for (uint i=0; i<queue.size(); i++) {
        auto* node = queue[i];
        node->for_each([&](const unique_ptr<KernelIR>& child) {
            queue.push_back(child.get());
        });
        if (node->type != KernelIRType::none || !node->has_attr(kir::code))
            continue;
        string target, value;
        if (!split_output_store(node->get_attr(kir::code), target, value))
            continue;
        auto pointer_end = target.find('[');
        string var_name = target.substr(0, pointer_end);
        if (!var_name.size() || var_name.back() != 'p') continue;
        var_name.pop_back();
        uint op_id, opvar_id;
        Op* target_op;
        Var* target_var;
        if (!pm->oc->try_get_op_var_by_name(
                var_name, op_id, opvar_id, target_op, target_var))
            continue;
        auto dtype = target_var->dtype();
        if (dtype == ns_float16 || dtype == ns_bfloat16 || dtype.is_complex())
            continue;
        node->get_attr(kir::code) =
            "__builtin_nontemporal_store((" + value + "), &(" + target + "));";
        changed = true;
    }
    LOGvvvv << "UseMovntPass rewrote output stores:" << changed;
}

} // jittor
