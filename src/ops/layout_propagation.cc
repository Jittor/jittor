// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <algorithm>
#include "ops/layout_propagation.h"
#include "ops/op_register.h"
#include "ops/broadcast_to_op.h"
#include "ops/composite/transpose_op.h"
#include "utils/log.h"

namespace jittor {

DEFINE_FLAG(int, propagate_storage_layout, 1,
    "Let an elementwise op whose operands are one permutation of dense storage "
    "(a channels-last activation read as NCHW) keep that layout: it runs on the "
    "dense source and returns the same permutation as a view, instead of "
    "writing a dense result the next convolution has to convert back. Applies "
    "to calls that record no gradient. 0 disables it.");

DECLARE_FLAG(int, transpose_storage_view);
DECLARE_FLAG(bool, no_grad);

static auto make_transpose = op_constructor<VarPtr, Var*, NanoVector>("transpose");
static auto make_broadcast_to_shape =
    op_constructor<VarPtr, Var*, NanoVector, NanoVector>("broadcast_to");

VarPtr storage_view_transpose(Var* v, NanoVector axes) {
    int saved = transpose_storage_view;
    transpose_storage_view = 1;
    VarPtr result;
    try {
        result = make_transpose(v, axes);
    } catch (...) {
        transpose_storage_view = saved;
        throw;
    }
    transpose_storage_view = saved;
    return result;
}

// Element-wise: NanoVector's own `==` also compares how the values are packed,
// and two permutations built different ways pack the same axes differently.
static bool same_axes(const NanoVector& a, const NanoVector& b) {
    if (a.size() != b.size()) return false;
    for (uint i=0; i<a.size(); i++)
        if (a[i] != b[i]) return false;
    return true;
}

static NanoVector inverse(const NanoVector& axes) {
    NanoVector result;
    result.reserve(axes.size(), axes.size());
    for (uint i=0; i<axes.size(); i++)
        result.set_data(axes[i], i);
    return result;
}

// The permutation `v` is of a dense tensor, and that tensor; false when `v`
// is dense itself or not a permutation of anything dense.
static bool permuted_source(Var* v, NanoVector& axes, VarPtr& source) {
    auto n = v->shape.size();
    if (n < 2) return false;
    Op* op = v->input();
    if (op && !v->is_finished() && op->is_op(op_ids::transpose())) {
        auto* transpose = static_cast<TransposeOp*>(op);
        if (transpose->storage_view && transpose->x->is_contiguous()) {
            axes = transpose->axes;
            source = transpose->x;
            return true;
        }
    }
    if (v->is_contiguous()) return false;
    // Order the axes outermost-first by stride; the storage is dense in that
    // order when each stride is the product of the extents inside it. A
    // unit extent has no meaningful stride, so its position is ambiguous.
    vector<int> order(n);
    for (uint i=0; i<n; i++) {
        if (v->shape[i] <= 1) return false;
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return v->storage_stride(a) > v->storage_stride(b);
    });
    int64 expect = 1;
    for (int k=(int)n-1; k>=0; k--) {
        if (v->storage_stride(order[k]) != expect) return false;
        expect *= v->shape[order[k]];
    }
    NanoVector to_source;
    for (uint i=0; i<n; i++) to_source.push_back(order[i]);
    source = storage_view_transpose(v, to_source);
    if (!source->is_contiguous()) return false;
    axes = inverse(to_source);
    return true;
}

bool storage_layout_operands(const vector<Var*>& inputs, NanoVector& axes,
                             vector<VarPtr>& sources) {
    if (!propagate_storage_layout) return false;
    bool grad = false;
    for (Var* v : inputs) grad |= !v->is_stop_grad();
    if (grad && !no_grad) return false;
    vector<VarPtr> found(inputs.size());
    NanoVector shape;
    bool any = false;
    for (uint i=0; i<inputs.size(); i++) {
        NanoVector a;
        VarPtr s;
        if (!permuted_source(inputs[i], a, s)) continue;
        if (any && !same_axes(a, axes)) return false;
        axes = a;
        shape = s->shape;
        found[i] = move(s);
        any = true;
    }
    if (!any) return false;
    NanoVector back = inverse(axes);
    for (uint i=0; i<inputs.size(); i++) {
        if (found[i]) continue;
        Var* v = inputs[i];
        Op* op = v->input();
        if (v->is_finished() || !op) {
            // Already in memory -- an auto flush can run a broadcast between
            // its creation and its use -- so a view of it in the source's
            // order costs nothing and forces nothing.
            found[i] = storage_view_transpose(v, back);
            continue;
        }
        if (!op->is_op(op_ids::broadcast_to())) return false;
        auto* broadcast = static_cast<BroadcastToOp*>(op);
        if (broadcast->x->num == 1) {
            // A constant: broadcast it to the source's shape directly, so it
            // still fuses into the kernel as one.
            found[i] = make_broadcast_to_shape(broadcast->x, shape, NanoVector());
        } else if (broadcast->expand_is_view) {
            // An expand is a strided view already; permuting it is free.
            found[i] = storage_view_transpose(v, back);
        } else {
            return false;
        }
    }
    sources = move(found);
    return true;
}

} // jittor
