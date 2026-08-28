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

        // Collect the accumulating statements. There can be several -- a
        // GroupNorm's mean and mean-of-squares are two reductions fused into
        // one loop -- and each gets its own accumulator.
        vector<KernelIR*> stores;
        vector<string> targets, rests;
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
            stores.push_back(c.get()); targets.push_back(t); rests.push_back(r);
        }
        if (!usable || stores.size() == 0) continue;

        bool ok = true;
        for (uint s=0; s<stores.size() && ok; s++) {
            const string& target = targets[s];
            // The whole target, brackets included, has to reappear in the value.
            if (rests[s].find(target) == string::npos) { ok = false; break; }
            // The address must be fixed for the duration of the loop.
            if (mentions(target, index)) { ok = false; break; }
            string pointer = target.substr(0, target.find('['));
            // Nothing else in the loop may touch that array, or hoisting the
            // location into a local would hide the other access.
            for (auto& c : node->children) {
                if (c.get() == stores[s]) continue;
                const string& code = c->type == "define"
                    ? c->get_attr("rvalue") : c->get_attr("code");
                if (mentions(code, pointer)) { ok = false; break; }
            }
            // Two accumulators for the same location would each miss the
            // other's updates.
            for (uint o=0; o<s; o++)
                if (targets[o] == target) ok = false;
        }
        if (!ok) continue;

        vector<string> index_names(stores.size());
        vector<KernelIR*> index_defines(stores.size(), nullptr);
        for (uint s=0; s<stores.size() && ok; s++) {
            auto open = targets[s].find('[');
            index_names[s] = targets[s].substr(
                open + 1, targets[s].rfind(']') - open - 1);
            for (auto& c : node->children)
                if (c->type == "define"
                        && c->get_attr("lvalue") == index_names[s])
                    index_defines[s] = c.get();
            if (index_defines[s]
                    && mentions(index_defines[s]->get_attr("rvalue"), index))
                ok = false;
        }
        if (!ok) continue;

        for (uint s=0; s<stores.size(); s++) {
            const string& target = targets[s];
            string acc = "jt_reduce_acc_" + index_names[s];
            // The index is computed inside the loop, so its definition has to
            // come along; it is loop invariant, which is what was just checked.
            if (index_defines[s]) {
                node->push_back(index_defines[s]->get_attr("dtype") + " "
                                + index_names[s] + " = "
                                + index_defines[s]->get_attr("rvalue") + ";",
                                &node->before);
            }
            node->push_back("auto " + acc + " = " + target + ";",
                            &node->before);
            string value = rests[s];
            for (size_t at = value.find(target); at != string::npos;
                 at = value.find(target, at + acc.size()))
                value = value.substr(0, at) + acc
                      + value.substr(at + target.size());
            stores[s]->get_attr("code") = acc + " =" + value;
            node->push_back(target + " = " + acc + ";", &node->after);
        }
    }
}

} // jittor
