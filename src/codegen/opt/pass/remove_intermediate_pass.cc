// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "core/var.h"
#include "codegen/opt/pass_manager.h"
#include "codegen/opt/pass/remove_intermediate_pass.h"

namespace jittor {

static bool remove_empty_loop(KernelIR* i) {
    for (int j=0; j<i->children.size(); j++) {
        if (remove_empty_loop(i->children[j].get()))
            j--;
    }
    if (i->type == KernelIRType::loop && i->children.size() == 0) {
        i->erase();
        return true;
    }
    return false;
}

// A kernel that reads back an output it stores declares no pointer
// `__restrict__`. With them, nvcc folds `c ? zp[i] : xp[i]` into one load
// from the selected address, issues it read-only and ahead of the store to
// `zp[i]` -- the read returns what the memory held before the kernel. Dropping
// the qualifier from `zp` alone is not enough; the select still carries `xp`'s.
static void unrestrict(KernelIR* ir) {
    ir->dfs([&](unique_ptr<KernelIR>& c) {
        if (c->type != KernelIRType::define) return;
        auto& dtype = c->get_attr(kir::dtype);
        auto at = dtype.find("__restrict__");
        if (at != string::npos) dtype.erase(at, sizeof("__restrict__")-1);
    });
}

static bool is_name_char(char c) {
    return c == '_' || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// Occurrences of `pat` not preceded by a name character; `amp` counts those
// taken by address (`&pat`), which write through the pointer.
static int count_mentions(const string& text, const string& pat, int* amp=nullptr) {
    int n = 0;
    for (size_t at = text.find(pat); at != string::npos; at = text.find(pat, at + 1)) {
        if (at && is_name_char(text[at-1])) continue;
        n++;
        if (amp) {
            size_t b = at;
            while (b && text[b-1] == ' ') b--;
            if (b && text[b-1] == '&') (*amp)++;
        }
    }
    return n;
}

// An output the kernel stores and then reads back later in the same loop
// body -- a bias gradient summing the GELU backward it also writes out for the
// weight gradient -- goes back to memory for a value it just computed, and the
// load of every iteration waits behind that iteration's store. Keep the value
// in a register instead: `T zd = v; zp[i] = zd;`, and the reads use `zd`.
// It is the assumption remove_intermediate makes for a value never stored: a
// consumer fused into the loop reads the element of the iteration it runs in.
// Only a single plain store with every read after it in the same body is
// rewritten; anything else keeps the memory round trip. Returns whether any
// output is still read back.
static bool forward_stored_outputs(FusedOp* fused, PassManager* pm, KernelIR* ir) {
    unordered_set<Var*> read;
    for (Op* op : fused->ops)
        for (Var* v : op->inputs())
            read.insert(v);
    int reads_back = 0;
    for (auto& vi : fused->vars) {
        if (vi.type != 2 || !read.count(vi.var)) continue;
        reads_back++;
        Op* op = vi.var->input();
        if (!pm->oc->op_exist(op)) continue;
        string name;
        for (uint i=0; i<op->outputs().size(); i++)
            if (op->output(i)==vi.var)
                name = pm->oc->get_name_by_op_output(op, i);
        if (name.empty()) continue;
        string ptr = name + "p[";
        KernelIR* store = nullptr;
        int stores = 0;
        ir->dfs([&](unique_ptr<KernelIR>& c) {
            if (c->type == KernelIRType::none && c->has_attr(kir::code)
                    && startswith(c->attrs[kir::code], ptr)) {
                store = c.get();
                stores++;
            }
        });
        if (stores != 1 || !store->father || store->flist != &store->father->children)
            continue;
        string& code = store->attrs[kir::code];
        size_t close = ptr.size(), depth = 1;
        while (close < code.size() && depth) {
            if (code[close] == '[') depth++;
            if (code[close] == ']') depth--;
            close++;
        }
        size_t eq = close;
        while (eq < code.size() && code[eq] == ' ') eq++;
        if (depth || eq + 1 >= code.size() || code[eq] != '=' || code[eq+1] == '=') continue;
        size_t end = code.find_last_of(';');
        if (end == string::npos || end <= eq + 1) continue;
        auto& siblings = store->father->children;
        uint pos = 0;
        while (siblings[pos].get() != store) pos++;
        int later = 0, amp = 0;
        for (uint k = pos + 1; k < siblings.size(); k++)
            later += count_mentions(siblings[k]->to_string(), ptr, &amp);
        int all = count_mentions(ir->to_string(), ptr);
        if (!later || amp || later + 1 != all) continue;
        string value = code.substr(eq + 1, end - eq - 1);
        code = code.substr(0, close) + " = " + name + "d;";
        // Kept at the stored type: a reader of the tensor sees the rounded
        // value, and a bfloat16 bias gradient summed from the float it was
        // rounded from is not the gradient torch computes. Spelled from the
        // var's dtype, not from the op's type macros: those are named after
        // the op's own template parameters, and `broadcast_to` stores its `z`
        // as `Tx` -- there is no `opN_Tz` to name.
        string type = vi.var->dtype().to_cstring();
        store->father->insert(pos, type + " " + name + "d = " + value + ";");
        unordered_set<string> names = {name};
        for (uint k = pos + 2; k < siblings.size(); k++)
            siblings[k]->remove_intermediate(names);
        if (count_mentions(ir->to_string(), ptr) == 1) reads_back--;
    }
    return reads_back > 0;
}

void RemoveIntermediatePass::run() {
    if (forward_stored_outputs(op, pm, ir))
        unrestrict(ir);
    unordered_set<string> names;
    for (auto& vi : op->vars) {
        // intermediate
        if (vi.type != 1) continue;
        Op* op = vi.var->input();
        if (!pm->oc->op_exist(op)) continue;
        for (uint i=0; i<op->outputs().size(); i++)
            if (op->output(i)==vi.var)
                names.insert(pm->oc->get_name_by_op_output(op, i));
    }
    LOGvvvv << "Remove intermediate:" << names;
    ir->remove_intermediate(names);
    ir->remove_all_unused();
    ir->solve_conflict_define();

    // remove empty loop
    remove_empty_loop(ir);

    ir->remove_all_unused();

}

} // jittor