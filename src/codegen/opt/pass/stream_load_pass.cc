// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/var.h"
#include "codegen/op_compiler.h"
#include "codegen/opt/pass_manager.h"
#include "codegen/opt/pass/stream_load_pass.h"
#include "runtime/backend.h"
#include "utils/str_utils.h"

namespace jittor {

static bool ident_char(char c) { return isalnum((unsigned char)c) || c == '_'; }

// The position of the bracket or parenthesis closing the one at `open`.
static size_t closing(const string& code, size_t open) {
    char o = code[open], c = o == '[' ? ']' : ')';
    int depth = 0;
    for (size_t k = open; k < code.size(); k++) {
        if (code[k] == o) depth++;
        else if (code[k] == c && --depth == 0) return k;
    }
    return string::npos;
}

// Rewrites the loads of `pointers` in one statement: `p[i]` and a vectorised
// loop's `*reinterpret_cast<const V*>(p + ...)`. A store -- the statement's
// own target -- and an address taken are left alone.
static bool rewrite_loads(string& code, const unordered_map<string, int>& pointers, bool is_statement) {
    bool changed = false;
    const string vec = "*reinterpret_cast<const ";
    for (size_t i = 0; i < code.size(); i++) {
        if (code.compare(i, vec.size(), vec) == 0) {
            size_t star = code.find("*>(", i + vec.size());
            if (star == string::npos) continue;
            size_t open = star + 2, name_end = open + 1;
            while (name_end < code.size() && ident_char(code[name_end])) name_end++;
            auto it = pointers.find(code.substr(open + 1, name_end - open - 1));
            size_t close = closing(code, open);
            if (it == pointers.end() || close == string::npos) continue;
            string load = "jt_stream_ld(" + code.substr(i + 1, close - i) + ", jt_stream, " + S(it->second) + ")";
            code = code.substr(0, i) + load + code.substr(close + 1);
            i += load.size() - 1;
            changed = true;
            continue;
        }
        if (!ident_char(code[i]) || (i && ident_char(code[i-1]))) continue;
        size_t j = i;
        while (j < code.size() && ident_char(code[j])) j++;
        auto it = pointers.find(code.substr(i, j - i));
        if (it == pointers.end() || j >= code.size() || code[j] != '[') { i = j - 1; continue; }
        size_t close = closing(code, j);
        if (close == string::npos) return changed;
        size_t before = i;
        while (before && (code[before-1] == ' ' || code[before-1] == '(')) before--;
        bool address = before && code[before-1] == '&';
        size_t after = close + 1;
        while (after < code.size() && code[after] == ' ') after++;
        bool store = is_statement && i == 0 && after < code.size() && code[after] == '='
            && (after + 1 >= code.size() || code[after+1] != '=');
        if (address || store) { i = close; continue; }
        string load = "jt_stream_ld(" + it->first + " + (" + code.substr(j + 1, close - j - 1) + "), jt_stream, "
            + S(it->second) + ")";
        code = code.substr(0, i) + load + code.substr(close + 1);
        i += load.size() - 1;
        changed = true;
    }
    return changed;
}

void StreamLoadPass::run() {
    if (!op->executes_on_accelerator() || op->execution_backend() != BackendId::Cuda) return;
    bool any = false;
    for (auto& f : ir->before) {
        if (f->type != KernelIRType::func || f->get_attr(kir::dtype).find("__global__") == string::npos)
            continue;
        const string& name = f->get_attr(kir::lvalue);
        // The inputs of the fusion this kernel takes by pointer, and their
        // bit: their index among the fusion's vars.
        unordered_map<string, int> pointers;
        for (auto& a : f->inner) {
            if (a->type != KernelIRType::define) continue;
            const string& p = a->get_attr(kir::lvalue);
            if (p.size() < 2 || !startswith(p, "op") || p.back() != 'p'
                    || a->get_attr(kir::dtype).find('*') == string::npos)
                continue;
            uint op_id, opvar_id; Op* o; Var* var;
            if (!pm->oc->try_get_op_var_by_name(p.substr(0, p.size() - 1), op_id, opvar_id, o, var)) continue;
            for (uint k = 0; k < op->vars.size() && k < 64; k++)
                if (op->vars[k].var == var && op->vars[k].type == 0) pointers[p] = k;
        }
        if (pointers.empty()) continue;
        // The one launch of this kernel, which passes the mask.
        KernelIR* launch = nullptr;
        int launches = 0;
        string call = name + "<<<";
        ir->dfs([&](unique_ptr<KernelIR>& c) {
            if (c->has_attr(kir::code) && c->attrs[kir::code].find(call) != string::npos) {
                launch = c.get();
                launches++;
            }
        });
        if (launches != 1) continue;
        auto& launch_code = launch->attrs[kir::code];
        if (!endswith(launch_code, ");")) continue;
        bool changed = false;
        f->dfs([&](unique_ptr<KernelIR>& c) {
            if (c->has_attr(kir::code)) changed |= rewrite_loads(c->attrs[kir::code], pointers, true);
            if (c->has_attr(kir::rvalue)) changed |= rewrite_loads(c->attrs[kir::rvalue], pointers, false);
        });
        if (!changed) continue;
        f->push_back("unsigned long long jt_stream;", &f->inner);
        bool first = launch_code[launch_code.size() - 3] == '(';
        launch_code = launch_code.substr(0, launch_code.size() - 2) + (first ? "" : ",") + "streamed_inputs);";
        any = true;
    }
    if (any) ir->push_front("#include \"type/stream_load.h\"", &ir->before);
}

} // jittor
