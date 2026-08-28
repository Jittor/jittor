// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "op_compiler.h"
#include "opt/pass_manager.h"
#include "opt/pass/reduce_accumulator_pass.h"

namespace jittor {

// A reduction writes its result straight back into the output element:
//
//     for (id3 ...) {
//         auto yid = <does not depend on id3>;
//         yp[yid] = op(yp[yid], <value>);
//     }
//
// That read-modify-write through memory is something the C++ compiler will not
// look through, and it stops the whole loop body from vectorising -- g++ reports
// "complicated access pattern" and leaves any transcendental in the body as a
// scalar call. Accumulating in a local and storing once after the loop restores
// it: on the fused softmax backward of a diffusion UNet the same source goes
// from "vectorized 0 loops" to one loop vectorised with 32-byte vectors.
//
// This runs on CPU only. The CUDA reduction template already accumulates in a
// register, and the pass sits after every loop-restructuring pass so that it
// only ever sees the final nest.

static bool mentions(const string& text, const string& name) {
    // Whole-identifier search: "id3" must not match "id30" or "xid3".
    for (size_t i = text.find(name); i != string::npos;
         i = text.find(name, i + 1)) {
        bool left = i > 0 && (isalnum(text[i-1]) || text[i-1] == '_');
        size_t end = i + name.size();
        bool right = end < text.size() && (isalnum(text[end]) || text[end] == '_');
        if (!left && !right) return true;
    }
    return false;
}

// Split "yp[yid] = rest" into its target and the rest, or return false.
static bool split_assignment(const string& src, string& target, string& rest) {
    auto eq = src.find('=');
    if (eq == string::npos || eq + 1 >= src.size()) return false;
    // Reject ==, +=, and friends: only a plain assignment is handled.
    if (src[eq+1] == '=') return false;
    if (eq && (src[eq-1] == '=' || src[eq-1] == '!' || src[eq-1] == '<'
               || src[eq-1] == '>' || src[eq-1] == '+' || src[eq-1] == '-'
               || src[eq-1] == '*' || src[eq-1] == '/'))
        return false;
    target = src.substr(0, eq);
    rest = src.substr(eq + 1);
    while (target.size() && (target.back() == ' ' || target.back() == '\t'))
        target.pop_back();
    size_t b = target.find_first_not_of(" \t");
    if (b == string::npos) return false;
    target = target.substr(b);
    return true;
}

void ReduceAccumulatorPass::run() {
    if (op->flags.get(NodeFlags::_cuda)) return;
    // CheckCachePass has already run and paired every array access with a
    // memory_checker call. Introducing accesses now would leave them
    // uninstrumented and quietly skew the cache figures that mode exists to
    // produce, so leave the kernel alone while it is being measured.
    if (op->get_loop_option("check_cache")) return;

    vector<KernelIR*> queue({ir});
    for (uint i=0; i<queue.size(); i++) {
        KernelIR* node = queue[i];
        bool inner_most = true;
        for (auto& c : node->children) {
            if (c->type == "loop") inner_most = false;
            queue.push_back(c.get());
        }
        if (node->type != "loop" || !inner_most) continue;
        if (!node->has_attr("lvalue")) continue;
        string index = node->get_attr("lvalue");
        if (index.size() == 0) continue;

        // Exactly one accumulating statement, and nothing else that writes.
        KernelIR* store = nullptr;
        string target, rest;
        bool usable = true;
        for (auto& c : node->children) {
            if (c->type == "define") continue;
            string t, r;
            if (c->type.size()
                    || !split_assignment(c->get_attr("code"), t, r)) {
                // Anything that is not a plain assignment (a branch, a call,
                // a nested block) makes the body too unfamiliar to touch.
                if (c->type.size()) { usable = false; break; }
                continue;
            }
            if (t.find('[') == string::npos) continue;
            if (!mentions(r, t.substr(0, t.find('[')))) continue;
            if (store) { usable = false; break; }
            store = c.get(); target = t; rest = r;
        }
        if (!usable || !store) continue;
        // The whole target, brackets included, has to reappear in the value.
        if (rest.find(target) == string::npos) continue;
        // The address must be fixed for the duration of the loop.
        if (mentions(target, index)) continue;
        auto open = target.find('[');
        string index_name = target.substr(open + 1,
                                          target.rfind(']') - open - 1);
        KernelIR* index_define = nullptr;
        for (auto& c : node->children)
            if (c->type == "define" && c->get_attr("lvalue") == index_name)
                index_define = c.get();
        if (index_define && mentions(index_define->get_attr("rvalue"), index))
            continue;

        string acc = "jt_reduce_acc_" + index_name;
        // The index is computed inside the loop, so its definition has to come
        // along; it is loop invariant, which is what was just checked.
        if (index_define) {
            node->push_back(index_define->get_attr("dtype") + " " + index_name
                            + " = " + index_define->get_attr("rvalue") + ";",
                            &node->before);
        }
        node->push_back("auto " + acc + " = " + target + ";", &node->before);
        string value = rest;
        for (size_t at = value.find(target); at != string::npos;
             at = value.find(target, at + acc.size()))
            value = value.substr(0, at) + acc
                  + value.substr(at + target.size());
        store->get_attr("code") = acc + " =" + value;
        node->push_back(target + " = " + acc + ";", &node->after);
    }
}

} // jittor
