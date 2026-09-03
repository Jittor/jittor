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

// A split loop's variable reaches the store as "(id1 + id2) * stride" rather
// than "id1 * stride": the outer steps by the tile size and the inner runs up
// to it, so the pair enumerates the original dimension between them and
// distinct tiles still address distinct elements. Only accepted for a loop that
// really is split (it carries the stride in `rvalue2`), so this does not loosen
// the test for anything else.
static bool scales_with_tile(const string& expr, const string& index) {
    for (size_t i = expr.find(index); i != string::npos;
         i = expr.find(index, i + 1)) {
        bool left = i > 0 && (isalnum(expr[i-1]) || expr[i-1] == '_');
        size_t end = i + index.size();
        bool right = end < expr.size() && (isalnum(expr[end]) || expr[end] == '_');
        if (left || right) continue;
        // Walk to the closing parenthesis of the sum this index sits in, then
        // require that parenthesised group to be multiplied by something.
        if (i == 0 || expr[i-1] != '(') {
            // "(other + index)" -- step back over the other term instead.
            size_t open = expr.rfind('(', i);
            if (open == string::npos) continue;
            bool only_sum = true;
            for (size_t k = open + 1; k < i; k++)
                if (!(isalnum(expr[k]) || expr[k] == '_' || expr[k] == ' '
                      || expr[k] == '+')) { only_sum = false; break; }
            if (!only_sum) continue;
        }
        size_t j = end, depth = 1;
        while (j < expr.size() && depth) {
            if (expr[j] == '(') depth++;
            if (expr[j] == ')') depth--;
            if (depth == 0) break;
            // Anything but a sum of plain names inside the group means this is
            // not the split pattern.
            if (!(isalnum(expr[j]) || expr[j] == '_' || expr[j] == ' '
                  || expr[j] == '+')) return false;
            j++;
        }
        if (j >= expr.size()) continue;
        j++;
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
        if (c->type == KernelIRType::define) {
            defines[c->get_attr(kir::lvalue)] = c->get_attr(kir::rvalue);
            continue;
        }
        if (c->type == KernelIRType::loop) {
            if (c->has_attr(kir::rvalue))
                out.inner_bounds.push_back(c->get_attr(kir::rvalue));
            // A split loop keeps its tile-size define in `inner`, not among the
            // children, and that name is only in scope inside the loop.
            for (auto& s : c->inner)
                if (s->type == KernelIRType::define)
                    defines[s->get_attr(kir::lvalue)] = s->get_attr(kir::rvalue);
            scan(c.get(), out, defines);
            // Anything the accumulator pass parked around the loop counts too.
            for (auto* ls : {&c->before, &c->after})
                for (auto& s : *ls) {
                    if (s->type == KernelIRType::define) {
                        defines[s->get_attr(kir::lvalue)] = s->get_attr(kir::rvalue);
                        continue;
                    }
                    const string& code = s->get_attr(kir::code);
                    auto eq = code.find('=');
                    auto br = code.find('[');
                    if (eq == string::npos || br == string::npos || br > eq)
                        continue;
                    out.store_indices.push_back(
                        code.substr(br + 1, code.rfind(']', eq) - br - 1));
                }
            continue;
        }
        // A comment emits nothing. The asm tuner's `//@begin replace` markers
        // ride along in the parallel copy too; the substitution they drive
        // (ordinary stores to non-temporal ones) is a performance choice, so
        // the worst case if outlining moves the code out of their reach is
        // that it does not happen.
        if (c->type == KernelIRType::comment) continue;
        if (c->type != KernelIRType::none) { out.usable = false; return; }
        const string& code = c->get_attr(kir::code);
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
    if (op->flag(OpFlags::_cuda)) return;
    // Uninstrumented accesses would skew what that mode measures, and the
    // reduction accumulators this relies on are skipped there too.
    if (op->get_loop_option("check_cache")) return;

    vector<KernelIR*> bodies({ir});
    for (auto& c : ir->before)
        if (c->type == KernelIRType::func) bodies.push_back(c.get());

    // Collect first: the transform replaces nodes in the list being walked.
    vector<KernelIR*> targets;
    vector<string> guards;
    vector<string> inner_guards;
    vector<int> depths;
    // Whether that branch has to interchange the two outer loops first.
    vector<bool> interchange;
    for (auto* body : bodies) {
        for (auto& loop : body->children) {
            if (loop->type != KernelIRType::loop) continue;
            if (!loop->has_attr(kir::lvalue) || !loop->has_attr(kir::rvalue)) continue;
            string index = loop->get_attr(kir::lvalue);
            string bound = loop->get_attr(kir::rvalue);
            if (index.size() == 0 || !is_identifier(bound)) continue;
            // A split loop steps by a tile instead of by one, so its trip count
            // is bound/stride and its index reaches the store inside a sum.
            bool split = loop->has_attr(kir::rvalue2)
                         && is_identifier(loop->get_attr(kir::rvalue2));
            string stride = split ? loop->get_attr(kir::rvalue2) : string();
            bool already = false;
            for (auto& b : loop->before)
                if (mentions(b->get_attr(kir::code), "pragma")) already = true;
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
            while (!split && indices.size() < 3
                    && level->children.size() == 1
                    && level->children[0]->type == KernelIRType::loop
                    && level->children[0]->has_attr(kir::lvalue)
                    && level->children[0]->has_attr(kir::rvalue)
                    && is_identifier(level->children[0]->get_attr(kir::rvalue))
                    && level->children[0]->before.size() == 0
                    && level->children[0]->after.size() == 0) {
                level = level->children[0].get();
                indices.push_back(level->get_attr(kir::lvalue));
                bounds.push_back(level->get_attr(kir::rvalue));
            }

            LoopScan info;
            unordered_map<string,string> defines;
            for (auto& s : loop->inner)
                if (s->type == KernelIRType::define)
                    defines[s->get_attr(kir::lvalue)] = s->get_attr(kir::rvalue);
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
                    if (!(split ? scales_with_tile(expr, v)
                                : scales_with(expr, v))) {
                        disjoint = false; break;
                    }
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
                        if (!(split ? scales_with_tile(expr, v)
                                    : scales_with(expr, v))) {
                            disjoint = false; break;
                        }
                    if (!disjoint) break;
                }
            }
            // Not fatal on its own. A reduction's outermost loop runs over a
            // dimension the output index does not contain, so it can never be
            // threaded -- but the loop below it, over a dimension the output
            // does keep, can be. Swapping the two puts the threadable one on
            // the outside, which is the only placement that pays: distributing
            // the inner loop instead costs a thread team (or at best a barrier)
            // on every one of the outer loop's thousands of iterations.
            bool outer_ok = disjoint;

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
                // A bound declared inside the loop is not in scope where the
                // guard sits, and a split loop's inner bound is the tile size,
                // which would double-count the dimension the outer already has.
                if (collapsed || !is_identifier(b) || defines.count(b)) continue;
                work += " * " + b;
            }
            bound = bound_product;
            depths.push_back((int)indices.size());
            // ... but the collapsed levels still have to hold enough
            // iterations together to give every core something to do.
            // The cast is (long long) because the index type in scope varies
            // between kernels and the product can overflow a 32-bit one.
            string parallelism = split
                ? "(" + bound + ") / (" + stride + ")"
                : "(" + bound + ")";
            // Fewer tiles than that and the threads are not worth their setup;
            // a tile is already thousands of elements, so this is not small.
            string least = split ? "32" : "64";
            guards.push_back(outer_ok
                ? parallelism + " >= " + least
                  + " && (long long)(" + work + ") >= 65536"
                : string());
            targets.push_back(loop.get());
            interchange.push_back(!outer_ok);

            // A split loop's tile count is small by construction -- the tile is
            // sized to fit a cache, not to give every core a share -- so the
            // guard above is often false for the whole life of the kernel. The
            // parallelism is in the loop *inside* it, over the other dimension.
            // Offer that as a second branch rather than falling straight back
            // to running the nest on one core: on a ViT step the SGD update
            // was split with stride 4096 over a dimension of at most 3072, so
            // it never once took the parallel branch.
            string inner_guard;
            // Interchange needs a perfect nest: anything else between the two
            // headers would move relative to the loop it sits in.
            bool perfect = loop->children.size() == 1
                           && loop->children[0]->type == KernelIRType::loop;
            {
                for (auto& c : loop->children) {
                    if (c->type != KernelIRType::loop) continue;
                    if (!c->has_attr(kir::lvalue) || !c->has_attr(kir::rvalue)) break;
                    string iv = c->get_attr(kir::lvalue);
                    string ib = c->get_attr(kir::rvalue);
                    if (!is_identifier(ib) || defines.count(ib)) break;
                    bool ok = true;
                    for (auto& idx : info.store_indices) {
                        string expr = idx;
                        for (int d=0; d<4; d++) {
                            auto it = defines.find(expr);
                            if (it == defines.end()) break;
                            expr = it->second;
                        }
                        if (!scales_with(expr, iv)) { ok = false; break; }
                    }
                    if (ok && (outer_ok || perfect))
                        inner_guard = "(" + ib + ") >= 64 && (long long)("
                                      + work + ") >= 65536";
                    break;
                }
            }
            inner_guards.push_back(inner_guard);
            if (!outer_ok && inner_guard.size() == 0) {
                guards.pop_back();
                depths.pop_back();
                targets.pop_back();
                interchange.pop_back();
                inner_guards.pop_back();
            }
        }
    }

    for (uint i=0; i<targets.size(); i++) {
        KernelIR* loop = targets[i];
        KernelIR* father = loop->father;
        uint pos = 0;
        while (pos < father->children.size()
               && father->children[pos].get() != loop) pos++;
        if (pos == father->children.size()) continue;

        unique_ptr<KernelIR> if_par;
        string serial_cond;
        if (guards[i].size()) {
            auto par = loop->clone(true);
            string pragma = "#pragma omp parallel for";
            if (depths[i] > 1)
                pragma += " collapse(" + std::to_string(depths[i]) + ")";
            par->push_back(pragma, &par->before);
            if_par = std::make_unique<KernelIR>("if (" + guards[i] + ") {}");
            if_par->push_back(move(par), &if_par->children);
            serial_cond = "!(" + guards[i] + ")";
        }

        // Second branch: the outermost loop is out -- too few tiles, or it runs
        // over a reduced dimension. For a reduction the fix is to interchange
        // the two headers so the threadable dimension ends up outside, then
        // thread that. Distributing the inner loop where it stands does not
        // work: a thread team per outer iteration made ViT 5.3x slower, and
        // hoisting the region to leave only a barrier per iteration was still
        // 2x slower than not threading at all.
        unique_ptr<KernelIR> if_inner;
        if (inner_guards[i].size()) {
            auto inner_par = loop->clone(true);
            if (interchange[i])
                // `false`: swap the headers, leave the bodies where they are.
                inner_par->swap(*inner_par->children[0], false);
            if (interchange[i])
                inner_par->push_back("#pragma omp parallel for",
                                     &inner_par->before);
            else
                for (auto& c : inner_par->children)
                    if (c->type == KernelIRType::loop) {
                        c->push_back("#pragma omp parallel for", &c->before);
                        break;
                    }
            string cond = serial_cond.size()
                ? serial_cond + " && (" + inner_guards[i] + ")"
                : "(" + inner_guards[i] + ")";
            if_inner = std::make_unique<KernelIR>("if (" + cond + ") {}");
            if_inner->push_back(move(inner_par), &if_inner->children);
            serial_cond = serial_cond.size()
                ? serial_cond + " && !(" + inner_guards[i] + ")"
                : "!(" + inner_guards[i] + ")";
        }

        auto ser = loop->clone(true);
        auto if_ser = std::make_unique<KernelIR>("if (" + serial_cond + ") {}");
        if_ser->push_back(move(ser), &if_ser->children);

        vector<unique_ptr<KernelIR>> both;
        if (if_par) both.push_back(move(if_par));
        if (if_inner) both.push_back(move(if_inner));
        both.push_back(move(if_ser));
        father->insert(pos, both);
        loop->erase();
    }
}

} // jittor
