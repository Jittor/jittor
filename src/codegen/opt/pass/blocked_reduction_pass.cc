// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/var.h"
#include "codegen/op_compiler.h"
#include "codegen/opt/pass_manager.h"
#include "codegen/opt/pass/blocked_reduction_pass.h"
#include "utils/str_utils.h"

namespace jittor {

// A float32 sum that keeps one running total is wrong by an amount that grows
// with the trip count: every add rounds the total against an element that is
// by then much smaller than it, so the error is proportional to `n`. Summing
// sixteen million copies of 0.1 that way lands 15% off. CUDA's tree reduction
// and NumPy's pairwise summation both stay near machine epsilon at the same
// size, so a CPU-versus-CUDA comparison at that scale is measured against a
// CPU reference that is itself wrong.
//
// This rewrites a reduction into the shape those two use:
//
//   * the innermost loop runs in fixed-size blocks, with several partial sums
//     in flight inside a block;
//   * each block result is pushed on a stack that folds two entries whenever
//     they carry the same weight, which is pairwise summation written
//     iteratively;
//   * the stack spans the *whole* reduction nest, not one loop of it, so a
//     `(32, 224, 224)` sum is as accurate as a flat one of the same size.
//
// The stack is what keeps the error flat rather than merely smaller: chaining
// the block results instead would trade a factor of the block size for the
// same linear growth. Spanning the nest is the same argument one level up --
// a per-loop stack would leave the outer dimensions chained, and a `.sum()`
// over an image batch is a nest, not a flat loop.
//
// It is also the faster shape. The loop this replaces is a chain of dependent
// adds, one every few cycles; eight partials are eight independent chains, and
// the machine issues two adds per cycle. Measured on a 16.7M-element float32
// `sum` the kernel goes from 4.5 GB/s to 17 GB/s, so the accuracy here is a
// consequence of the faster shape rather than a payment for it.
//
// Only floating point *additive* reductions are rewritten, and only those:
// ReduceAccumulatorPass marks a loop with `kir::reduce_acc` when every
// accumulating store in it qualifies. `maximum`, `minimum` and the bitwise
// folds are exactly associative and gain nothing; `multiply` is not a
// reassociation anyone asked for; an integer sum is already exact.
//
// The pass runs last, after every pass that reshapes, clones or renames loops,
// and leaves behind one opaque text node -- so nothing downstream has to
// understand the shape, and everything upstream keeps seeing the ordinary
// single-accumulator loop it was written against.

//: partial sums kept in flight inside a block. Eight independent dependency
//: chains cover the latency of one add at two adds per cycle.
static const int WAYS = 8;

//: partials for a loop body big enough that copying it eight times is the
//: expensive part. A fused reduction -- a softmax backward, say -- already
//: does enough arithmetic per element that the accumulator chain is not what
//: limits it, while eight copies of that body cost the JIT half again as long
//: to compile. Two still breaks the chain; the in-block length, and so the
//: accuracy, is unchanged because the block scales with it.
static const int FEW_WAYS = 2;

//: how long a body has to be, in characters of generated C++, to count as
//: big. A plain `sum` is about a hundred; a fused reduction runs to several.
static const size_t BIG_BODY = 400;

//: elements per partial sum. The error inside a block is proportional to it;
//: the number of blocks enters only through the depth of the pairwise stack,
//: which is a logarithm. The block size is this times the number of partials,
//: so accuracy does not move when the partial count does.
static const int PER_WAY = 64;

//: below this many iterations the partials and the fold cost more than the
//: dependency chain they break, so the loop is left serial -- it still hands
//: its total to the same stack, so the accuracy of the nest is unchanged.
static const int LEAST = 2 * WAYS;

//: the pairwise stack never holds more entries than the bit width of the
//: block counter.
static const int DEPTH = 64;

// Trim spaces and a trailing semicolon, so a parsed loop header can be
// compared against the form this pass knows how to rebuild.
static string tidy(const string& s) {
    string out;
    for (char c : s)
        if (c != ' ' && c != '\t' && c != '\n' && c != ';') out += c;
    return out;
}

static bool is_name_or_number(const string& s) {
    if (s.size() == 0) return false;
    for (char c : s)
        if (!isalnum(c) && c != '_') return false;
    return true;
}

// Whole-identifier search: "id3" must not match "id30" or "xid3".
static bool mentions(const string& text, const string& name) {
    if (name.size() == 0) return false;
    for (size_t i = text.find(name); i != string::npos;
         i = text.find(name, i + 1)) {
        bool left = i > 0 && (isalnum(text[i-1]) || text[i-1] == '_');
        size_t end = i + name.size();
        bool right = end < text.size()
                     && (isalnum(text[end]) || text[end] == '_');
        if (!left && !right) return true;
    }
    return false;
}

static string substitute(const string& text, const string& from,
                         const string& to) {
    string out;
    size_t i = 0;
    while (i < text.size()) {
        size_t at = text.find(from, i);
        if (at == string::npos) { out += text.substr(i); break; }
        bool left = at > 0 && (isalnum(text[at-1]) || text[at-1] == '_');
        size_t end = at + from.size();
        bool right = end < text.size()
                     && (isalnum(text[end]) || text[end] == '_');
        out += text.substr(i, at - i);
        out += (left || right) ? from : to;
        i = end;
    }
    return out;
}

// "for (T i = 0; i < n; i++)", with `n` a name or a number and no stride:
// the only header this pass knows how to take apart and put back together.
static bool is_canonical(KernelIR* loop, string& index, string& bound,
                         string& itype) {
    if (loop->type != KernelIRType::loop) return false;
    if (loop->has_attr(kir::rvalue2)) return false;
    // Exactly the three header clauses: a fourth entry is a definition the
    // loop carries in its own scope, and rebuilding the header from its parts
    // would drop it.
    if (loop->inner.size() != 3) return false;
    index = loop->get_attr(kir::lvalue);
    bound = loop->get_attr(kir::rvalue);
    itype = loop->inner[0]->get_attr(kir::dtype);
    if (index.size() == 0 || itype.size() == 0) return false;
    if (!is_name_or_number(bound)) return false;
    if (tidy(loop->inner[0]->get_attr(kir::rvalue)) != "0") return false;
    if (tidy(loop->inner[1]->get_attr(kir::code)) != index + "<" + bound)
        return false;
    if (tidy(loop->inner[2]->get_attr(kir::code)) != index + "++") return false;
    // A pragma belongs to the header this is about to dissolve.
    for (auto& b : loop->before)
        if (b->get_attr(kir::code).find("pragma") != string::npos) return false;
    return true;
}

void BlockedReductionPass::run() {
    if (op->executes_on_accelerator()) return;
    // The instrumented build pairs every array access with a memory_checker
    // call; the accumulators this builds on are not emitted there either.
    if (op->get_loop_option("check_cache")) return;

    // Collect first: the rewrite replaces nodes in the lists being walked.
    vector<KernelIR*> queue({ir});
    for (auto& c : ir->before)
        if (c->type == KernelIRType::func) queue.push_back(c.get());
    vector<KernelIR*> loops;
    for (uint i=0; i<queue.size(); i++) {
        KernelIR* node = queue[i];
        for (auto& c : node->children) queue.push_back(c.get());
        for (auto& c : node->inner) queue.push_back(c.get());
        if (node->type != KernelIRType::loop) continue;
        if (node->get_attr(kir::reduce_acc).size() == 0) continue;
        loops.push_back(node);
    }

    for (auto* inner : loops) {
        string index, bound, itype;
        if (!is_canonical(inner, index, bound, itype)) continue;

        // Straight-line body only: a jump or a nested loop would not survive
        // being run several times per iteration under a different index.
        bool plain = inner->children.size() > 0;
        for (auto& c : inner->children) {
            if (c->type != KernelIRType::none
                    && c->type != KernelIRType::define) plain = false;
            if (c->has_attr(kir::has_bc)) plain = false;
        }
        if (!plain) continue;

        auto accs = split(inner->get_attr(kir::reduce_acc), ",");
        if (accs.size() == 0) continue;

        // The accumulator declarations, and the store that follows them. Both
        // move out of the whole nest below, so everything they name has to
        // stay in scope there -- which is what the climb checks.
        string prologue, epilogue;
        for (auto& b : inner->before) prologue += b->to_string(0);
        for (auto& a : inner->after) epilogue += a->to_string(0);

        // Climb out through the *other* reduced dimensions. A reduced
        // dimension is exactly one whose index does not reach the output
        // address, so a loop qualifies when its index appears nowhere in the
        // accumulator's declarations or its store. Anything else in that
        // loop's body, or anything attached around it, stops the climb: the
        // headers climbed past are rebuilt from their parts, and only a
        // perfect nest can be rebuilt without moving a statement relative to
        // the loop it sits in.
        vector<string> outer_index, outer_bound, outer_itype;
        KernelIR* top = inner;
        while (true) {
            KernelIR* up = top->father;
            if (!up || up->type != KernelIRType::loop) break;
            if (!up->father || up->flist != &up->father->children) break;
            if (up->before.size() || up->after.size()) break;
            if (up->children.size() != 1 || up->children[0].get() != top) break;
            string ui, ub, ut;
            if (!is_canonical(up, ui, ub, ut)) break;
            if (mentions(prologue, ui) || mentions(epilogue, ui)) break;
            outer_index.push_back(ui);
            outer_bound.push_back(ub);
            outer_itype.push_back(ut);
            top = up;
        }
        if (!top->father || top->flist != &top->father->children) continue;

        string body;
        for (auto& c : inner->children) body += c->to_string(0);

        const string p = "jt_blk_" + index + "_";
        const int WAY_COUNT = body.size() > BIG_BODY ? FEW_WAYS : WAYS;
        const int BLOCK = PER_WAY * WAY_COUNT;
        const string blk = S(BLOCK), ways = S(WAY_COUNT);
        vector<string> zero(accs.size());
        for (uint s=0; s<accs.size(); s++)
            zero[s] = "decltype(" + accs[s] + ")(0)";

        // One copy of the body per partial sum, each with its own index. The
        // braces give every copy its own scope, so the definitions the body
        // makes (an offset, a fused subexpression) do not collide.
        auto copy_for = [&](const string& idx_expr, int way) {
            string one = "{ " + itype + " " + index + " = " + idx_expr + ";\n";
            string b = body;
            for (uint s=0; s<accs.size(); s++)
                b = substitute(b, accs[s], p + "a" + S(s) + "_" + S(way));
            return one + b + "}\n";
        };
        // Hand one finished partial to the stack, folding it with every entry
        // of equal weight. The counter's trailing ones say how many those are,
        // which is what makes the stack a balanced tree.
        auto push = [&]() {
            string t = p + "blocks += 1;\n";
            t += "for (long long " + p + "t = " + p + "blocks; !(" + p
               + "t & 1); " + p + "t >>= 1) {\n";
            t += p + "top -= 1;\n";
            for (uint s=0; s<accs.size(); s++)
                t += p + "a" + S(s) + "_0 = (" + p + "stack" + S(s) + "[" + p
                   + "top]) + (" + p + "a" + S(s) + "_0);\n";
            t += "}\n";
            for (uint s=0; s<accs.size(); s++)
                t += p + "stack" + S(s) + "[" + p + "top] = " + p + "a" + S(s)
                   + "_0;\n";
            t += p + "top += 1;\n";
            return t;
        };

        // A brace of its own, the way KernelIR::to_string wraps a loop that
        // carries statements before and after it: the accumulator declared in
        // the prologue must not escape into a scope that already has the name.
        string text = "{\n";
        text += prologue;
        text += "// blocked reduction: " + S((int)accs.size())
              + " accumulation(s), " + ways + " partials per block of " + blk
              + ", pairwise over " + S((int)outer_index.size() + 1)
              + " reduced dimension(s)\n";
        for (uint s=0; s<accs.size(); s++) {
            text += "decltype(" + accs[s] + ") " + p + "stack" + S(s) + "["
                  + S(DEPTH) + "];\n";
            // The partials start at the additive identity, never at the
            // accumulator: the accumulator is read once, by the drain at the
            // bottom, and seeding the partials with it would add whatever it
            // holds once per block.
            text += "decltype(" + accs[s] + ") ";
            for (int w=0; w<WAY_COUNT; w++)
                text += (w ? ", " : "") + p + "a" + S(s) + "_" + S(w)
                      + " = " + zero[s];
            text += ";\n";
        }
        text += "int " + p + "top = 0;\n";
        text += "long long " + p + "blocks = 0;\n";
        // The reduced dimensions above this loop, outermost first.
        for (int d=(int)outer_index.size()-1; d>=0; d--)
            text += "for (" + outer_itype[d] + " " + outer_index[d] + " = 0; "
                  + outer_index[d] + " < (" + outer_bound[d] + "); "
                  + outer_index[d] + "++) {\n";
        text += "if ((" + bound + ") >= " + S(LEAST) + ") {\n";
        text += "for (" + itype + " " + p + "base = 0; " + p + "base < ("
              + bound + "); " + p + "base += " + blk + ") {\n";
        text += itype + " " + p + "end = " + p + "base + " + blk + " < ("
              + bound + ") ? " + p + "base + " + blk + " : (" + bound + ");\n";
        for (uint s=0; s<accs.size(); s++)
            for (int w=0; w<WAY_COUNT; w++)
                text += p + "a" + S(s) + "_" + S(w) + " = " + zero[s] + ";\n";
        text += itype + " " + p + "i = " + p + "base;\n";
        text += "for (; " + p + "i + " + ways + " <= " + p + "end; " + p
              + "i += " + ways + ") {\n";
        for (int w=0; w<WAY_COUNT; w++)
            text += copy_for(p + "i + " + S(w), w);
        text += "}\n";
        text += "for (; " + p + "i < " + p + "end; " + p + "i += 1) {\n";
        text += copy_for(p + "i", 0);
        text += "}\n";
        // Fold this block's partials in a balanced tree, then push.
        for (uint s=0; s<accs.size(); s++)
            for (int width=WAY_COUNT/2; width>=1; width>>=1)
                for (int w=0; w<width; w++) {
                    string a = p + "a" + S(s) + "_";
                    text += a + S(w) + " = (" + a + S(w) + ") + ("
                          + a + S(w+width) + ");\n";
                }
        text += push();
        text += "}\n";
        text += "} else if ((" + bound + ") > 0) {\n";
        for (uint s=0; s<accs.size(); s++)
            text += p + "a" + S(s) + "_0 = " + zero[s] + ";\n";
        text += "for (" + itype + " " + p + "i = 0; " + p + "i < (" + bound
              + "); " + p + "i += 1) {\n";
        text += copy_for(p + "i", 0);
        text += "}\n";
        text += push();
        text += "}\n";
        for (uint d=0; d<outer_index.size(); d++) text += "}\n";
        // Whatever the stack still holds, newest first, onto the value the
        // accumulator came in with.
        text += "while (" + p + "top > 0) {\n";
        text += p + "top -= 1;\n";
        for (uint s=0; s<accs.size(); s++)
            text += accs[s] + " = (" + p + "stack" + S(s) + "[" + p
                  + "top]) + (" + accs[s] + ");\n";
        text += "}\n";
        text += epilogue;
        text += "}\n";

        auto blocked = std::make_unique<KernelIR>();
        blocked->type = KernelIRType::none;
        blocked->attrs[kir::code] = text;
        blocked->attrs[kir::raw] = "1";

        KernelIR* father = top->father;
        uint pos = 0;
        while (pos < father->children.size()
               && father->children[pos].get() != top) pos++;
        if (pos == father->children.size()) continue;
        vector<unique_ptr<KernelIR>> one;
        one.push_back(move(blocked));
        father->insert(pos, one);
        top->erase();
    }
}

} // jittor
