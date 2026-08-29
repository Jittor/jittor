// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var.h"
#include "op_compiler.h"
#include "opt/pass_manager.h"
#include "opt/pass/cpu_parallel_pass.h"

namespace jittor {

// Jittor's own VectorizePass and UnrollPass are gated on `cc_type == "icc"`
// ("only icc supports pragma"), and nothing else hands the CPU kernels to more
// than one core: of the fused kernels a diffusion UNet compiles, essentially
// none contain a threading construct, and a 64M-element `a+b` reaches 33 GB/s
// where single-threaded numpy already reaches 23.
//
// So mark the outermost loop with OpenMP -- the kernels are already compiled
// with -fopenmp -- but only when the iterations provably do not collide. Every
// index Jittor generates is an affine sum of the loop variables, each with its
// own stride, so if the outermost variable appears as a factor in *every* store
// index then distinct iterations write distinct elements. A reduction over the
// outermost dimension fails that test, because the reduced dimension is absent
// from the output index, and is left alone.
//
// The loop is emitted twice, under complementary guards, rather than once with
// an `if` clause on the pragma. A parallel region costs the inner loop half its
// vector width -- on the fused softmax row-reduce g++ reports "32 byte vectors"
// without the pragma and only "16 byte vectors" with it, because outlining the
// body into the region's function loses what it knew about the pointers. That
// trade is worth making when the threads actually arrive, and pure loss when
// the `if` clause turns them down, which is most calls in a real model. Two
// copies let the serial one keep its full width.

static bool mentions(const string& text, const string& name) {
    for (size_t i = text.find(name); i != string::npos;
         i = text.find(name, i + 1)) {
        bool left = i > 0 && (isalnum(text[i-1]) || text[i-1] == '_');
        size_t end = i + name.size();
        bool right = end < text.size() && (isalnum(text[end]) || text[end] == '_');
        if (!left && !right) return true;
    }
    return false;
}

// True when `index` appears as a factor, "id2 * stride", rather than as the
// zero coefficient the generator writes for a dimension that is held fixed.
static bool scales_with(const string& expr, const string& index) {
    for (size_t i = expr.find(index); i != string::npos;
         i = expr.find(index, i + 1)) {
        bool left = i > 0 && (isalnum(expr[i-1]) || expr[i-1] == '_');
        size_t end = i + index.size();
        bool right = end < expr.size() && (isalnum(expr[end]) || expr[end] == '_');
        if (left || right) continue;
        size_t j = end;
        while (j < expr.size() && (expr[j] == ' ' || expr[j] == '\t')) j++;
        if (j < expr.size() && expr[j] == '*') return true;
    }
    return false;
}

// A plain identifier can be multiplied into the work estimate; anything else
// (a call, an expression) is left out rather than pasted into the guard.
static bool is_identifier(const string& s) {
    if (s.size() == 0 || isdigit(s[0])) return false;
    for (char c : s)
        if (!isalnum(c) && c != '_') return false;
    return true;
}

struct LoopScan {
    bool usable = true;
    // Every store target's index, so the caller can test them together.
    vector<string> store_indices;
    // Trip counts of the nest below the outermost loop, for the work estimate.
    vector<string> inner_bounds;
};

static void scan(KernelIR* node, LoopScan& out,
                 unordered_map<string,string>& defines) {
    for (auto& c : node->children) {
        if (!out.usable) return;
        if (c->type == "define") {
            defines[c->get_attr("lvalue")] = c->get_attr("rvalue");
            continue;
        }
        if (c->type == "loop") {
            if (c->has_attr("rvalue"))
                out.inner_bounds.push_back(c->get_attr("rvalue"));
            scan(c.get(), out, defines);
            // Anything the accumulator pass parked around the loop counts too.
            for (auto* ls : {&c->before, &c->after})
                for (auto& s : *ls) {
                    if (s->type == "define") {
                        defines[s->get_attr("lvalue")] = s->get_attr("rvalue");
                        continue;
                    }
                    const string& code = s->get_attr("code");
                    auto eq = code.find('=');
                    auto br = code.find('[');
                    if (eq == string::npos || br == string::npos || br > eq)
                        continue;
                    out.store_indices.push_back(
                        code.substr(br + 1, code.rfind(']', eq) - br - 1));
                }
            continue;
        }
        if (c->type.size()) { out.usable = false; return; }
        const string& code = c->get_attr("code");
        // A jump would make the loop non-canonical for OpenMP.
        if (mentions(code, "break") || mentions(code, "continue")
                || mentions(code, "return") || mentions(code, "goto")) {
            out.usable = false;
            return;
        }
        auto eq = code.find('=');
        if (eq == string::npos) continue;
        auto br = code.find('[');
        if (br == string::npos || br > eq) continue;
        auto close = code.rfind(']', eq);
        if (close == string::npos || close < br) continue;
        out.store_indices.push_back(code.substr(br + 1, close - br - 1));
    }
}

void CpuParallelPass::run() {
    if (op->flags.get(NodeFlags::_cuda)) return;
    // Uninstrumented accesses would skew what that mode measures, and the
    // reduction accumulators this relies on are skipped there too.
    if (op->get_loop_option("check_cache")) return;

    vector<KernelIR*> bodies({ir});
    for (auto& c : ir->before)
        if (c->type == "func") bodies.push_back(c.get());

    // Collect first: the transform replaces nodes in the list being walked.
    vector<KernelIR*> targets;
    vector<string> guards;
    vector<int> depths;
    for (auto* body : bodies) {
        for (auto& loop : body->children) {
            if (loop->type != "loop") continue;
            if (!loop->has_attr("lvalue") || !loop->has_attr("rvalue")) continue;
            string index = loop->get_attr("lvalue");
            string bound = loop->get_attr("rvalue");
            if (index.size() == 0 || !is_identifier(bound)) continue;
            bool already = false;
            for (auto& b : loop->before)
                if (mentions(b->get_attr("code"), "pragma")) already = true;
            if (already) continue;

            // The outermost loop alone is often not worth threading: on an
            // NCHW tensor it runs over the batch, which is 2. Take the
            // perfectly-nested levels below it as well and collapse them, so
            // the parallelism is the product (batch * channels * ...) rather
            // than the batch. Only a level whose sole child is the next loop
            // may be collapsed -- anything between them, even an index
            // definition, makes the nest imperfect and OpenMP reject it.
            vector<string> indices({index}), bounds({bound});
            KernelIR* level = loop.get();
            while (indices.size() < 3
                    && level->children.size() == 1
                    && level->children[0]->type == "loop"
                    && level->children[0]->has_attr("lvalue")
                    && level->children[0]->has_attr("rvalue")
                    && is_identifier(level->children[0]->get_attr("rvalue"))
                    && level->children[0]->before.size() == 0
                    && level->children[0]->after.size() == 0) {
                level = level->children[0].get();
                indices.push_back(level->get_attr("lvalue"));
                bounds.push_back(level->get_attr("rvalue"));
            }

            LoopScan info;
            unordered_map<string,string> defines;
            scan(loop.get(), info, defines);
            if (!info.usable || info.store_indices.size() == 0) continue;

            // Resolve each store index through the definitions in scope, then
            // require every collapsed variable to scale it: each contributes
            // its own stride to the affine sum, so distinct tuples then address
            // distinct elements.
            bool disjoint = true;
            for (auto& idx : info.store_indices) {
                string expr = idx;
                for (int depth=0; depth<4; depth++) {
                    auto it = defines.find(expr);
                    if (it == defines.end()) break;
                    expr = it->second;
                }
                for (auto& v : indices)
                    if (!scales_with(expr, v)) { disjoint = false; break; }
                if (!disjoint) break;
            }
            // Drop the deepest levels until every one of them passes, rather
            // than giving up on the loop: the outermost alone may well be fine.
            while (!disjoint && indices.size() > 1) {
                indices.pop_back();
                bounds.pop_back();
                disjoint = true;
                for (auto& idx : info.store_indices) {
                    string expr = idx;
                    for (int depth=0; depth<4; depth++) {
                        auto it = defines.find(expr);
                        if (it == defines.end()) break;
                        expr = it->second;
                    }
                    for (auto& v : indices)
                        if (!scales_with(expr, v)) { disjoint = false; break; }
                    if (!disjoint) break;
                }
            }
            if (!disjoint) continue;

            // Total work, not just the trip count of the collapsed levels: a
            // loop of 64 rows over 4096 columns is worth threading and a loop
            // of 4096 scalars is not. Unknown inner bounds are dropped, which
            // only makes this stricter.
            string bound_product = bounds[0];
            for (uint b=1; b<bounds.size(); b++)
                bound_product += " * " + bounds[b];
            string work = bound_product;
            for (auto& b : info.inner_bounds) {
                bool collapsed = false;
                for (auto& used : bounds)
                    if (used == b) collapsed = true;
                if (!collapsed && is_identifier(b)) work += " * " + b;
            }
            bound = bound_product;
            depths.push_back((int)indices.size());
            // ... but the collapsed levels still have to hold enough
            // iterations together to give every core something to do.
            // The cast is (long long) because the index type in scope varies
            // between kernels and the product can overflow a 32-bit one.
            guards.push_back("(" + bound + ") >= 64 && (long long)(" + work
                             + ") >= 65536");
            targets.push_back(loop.get());
        }
    }

    for (uint i=0; i<targets.size(); i++) {
        KernelIR* loop = targets[i];
        KernelIR* father = loop->father;
        uint pos = 0;
        while (pos < father->children.size()
               && father->children[pos].get() != loop) pos++;
        if (pos == father->children.size()) continue;

        auto par = loop->clone(true);
        string pragma = "#pragma omp parallel for";
        if (depths[i] > 1) pragma += " collapse(" + std::to_string(depths[i]) + ")";
        par->push_back(pragma, &par->before);
        auto if_par = std::make_unique<KernelIR>("if (" + guards[i] + ") {}");
        if_par->push_back(move(par), &if_par->children);

        auto ser = loop->clone(true);
        auto if_ser = std::make_unique<KernelIR>("if (!(" + guards[i] + ")) {}");
        if_ser->push_back(move(ser), &if_ser->children);

        vector<unique_ptr<KernelIR>> both;
        both.push_back(move(if_par));
        both.push_back(move(if_ser));
        father->insert(pos, both);
        loop->erase();
    }
}

} // jittor
