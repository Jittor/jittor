// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include "var.h"
#include "ops/setitem_op.h"
#include "ops/getitem_op.h"
#include "ops/op_register.h"

namespace jittor {

// add dependency b -> a
static inline void add_dependency(Node* a, Node* b) {
    // check dependency is not exist
    for (auto na : a->inputs()) {
        if (na == b) return;
    }
    a->add_inputs({b});
    // set -1 mean this is a control dependency edge
    a->_inputs.back().reverse().index = -1;
}

static void setitem_inplace(SetitemOp* op) {
    // LOGir << "in setitem_inplace";
    auto input = op->inputs().front();
    if (!(input->outputs().size() == 1 && 
        input->liveness.forward.count()<=1 &&
        (op->op == ns_void || op->op == ns_add || op->op == ns_subtract))) {
        return;
    }
    auto input_op = input->input();
    if (input_op) {
        // make sure input op will not use input
        if (!(input_op->type() == OpType::broadcast || 
            input_op->inputs().size() == 0 ||
            input_op->is_op(op_ids::setitem()) ||
            input_op->is_op(op_ids::getitem())))
            // TODO: inplace getitem maybe risky, getitem maybe inplace too
        return;
    }
    auto output = op->outputs().front();
    // return if output is all ready shared
    if (output->allocator || output->is_sharing()) return;
    output->share_with(input);
    
    auto data = op->input(1);
    // if setitem requires type conversion, don't inplace
    if (data->dtype() != input->dtype())
        return;

    input_op = input->input();

    if (input_op && input_op->inputs().size() == 1) {
        input_op = input_op->inputs().front()->input();
    }
    if (input_op && input_op->inputs().size() == 1) {
        input_op = input_op->inputs().front()->input();
    }

    VarSlices vs = op->vs;
    if (!(data->is_finished() == 0 && 
          (data->outputs().size() == 1 || 
           (!input_op 
            || input_op->inputs().size() == 0))))
        return;
    if (data->allocator || data->is_sharing())
        return;
    auto data_op = data->input();
    if (data_op->flag(OpFlags::_custom_flag))
        return;

    auto in_shape = input->shape;
    int64 inplace_size = 1;
    for (int i = vs.n - 1; i > 0; --i) {
        VarSlice s = vs.slices[i];
        if (!(s.is_slice())) return;
        Slice ss = s.slice;
        if (!(ss.start == 0 && (ss.mask&2) && ss.step == 1))
            return;
        inplace_size *= in_shape[i];
    }
    
    VarSlice s = vs.slices[0];
    if (s.is_var() || s.is_str()) return;
    
    int64 size = 0;
    if (s.is_int())
        size = in_shape[0] == 0 ? 0 : s.i * (input->size / in_shape[0]);
    else if (s.is_slice()) {
        Slice ss = s.slice;
        // we also need to check the first dim is continuous
        if (ss.step != 1)
            return;
        size = in_shape[0] == 0 ? 0 : ss.start * (input->size / in_shape[0]);
        inplace_size *= ss.stop - ss.start;
    }
    if (inplace_size > data->num) {
        // if data has been broadcast into input, don't
        // inplace data, because their shapes are not match
        // This would lead partial setitem
        return;
    }
    add_dependency(data->input(), input->node());
    data->share_with(input, size);
    op->ns.set(GetitemOp::_inplace);
    // LOGir << input->shape << input->dtype() << data->shape << data->dtype() << vs << data->input();
    // LOGir << output;
}

static void getitem_inplace(GetitemOp* op) {
    // LOGir << "in getitem_inplace";

    auto in = op->inputs().front();
    auto ou = op->outputs().front();

    // return if out is all ready inplaced
    if (ou->allocator || ou->is_sharing())
        return;

    VarSlices vs = op->vs;
    auto in_shape = in->shape;

    for (int i = vs.n - 1; i > 0; --i) {
        VarSlice s = vs.slices[i];
        if (!(s.is_slice())) return;
        Slice ss = s.slice;
        if (!(ss.start == 0 && (ss.mask&2) && ss.step == 1))
            return;
    }
    
    VarSlice s = vs.slices[0];
    if (s.is_var() || s.is_str()) return;
    
    int64 size = 0;
    if (s.is_int())
        size = in_shape[0] == 0 ? 0 : s.i * (in->size / in_shape[0]);
    else if (s.is_slice()) {
        size = in_shape[0] == 0 ? 0 : s.slice.start * (in->size / in_shape[0]);
        if (s.slice.step != 1) return;
    }
    ASSERT(size>=0 && size<=in->size);
    ou->share_with(in, size);
    op->ns.set(GetitemOp::_inplace);
    // LOGir << "pass getitem_inplace";
    // LOGir << "inplace getitem" << vs << in->shape << ou->shape;
}

static bool slice_bounds(const VarSlice& s, int64 dim_size, int64& start, int64& stop) {
    if (!s.is_slice()) return false;
    Slice ss = s.slice;
    if (ss.mask == 7) {
        start = 0;
        stop = dim_size;
        return true;
    }
    int64 step = (ss.mask & 4) ? 1 : ss.step;
    if (step != 1)
        return false;
    start = (ss.mask & 1) ? 0 : ss.start;
    stop = (ss.mask & 2) ? dim_size : ss.stop;
    if (start < 0) {
        start += dim_size;
        if (start < 0)
            start = 0;
    }
    if (stop < 0)
        stop += dim_size;
    else
        stop = std::min(dim_size, stop);
    if (start < 0 || start > dim_size || stop < 0 || stop > dim_size || stop < start)
        return false;
    return true;
}

static bool full_slice(const VarSlice& s, int64 dim_size) {
    int64 start, stop;
    return slice_bounds(s, dim_size, start, stop) && start == 0 && stop == dim_size;
}

static int64 slice_len(const VarSlice& s, int64 dim_size) {
    int64 start, stop;
    if (!slice_bounds(s, dim_size, start, stop))
        return -1;
    return std::max((int64)0, stop - start);
}

static void getitem_contiguous_inplace(GetitemOp* op) {
    auto in = op->inputs().front();
    auto ou = op->outputs().front();
    if (ou->allocator || ou->is_sharing())
        return;

    VarSlices vs = op->vs;
    auto in_shape = in->shape;
    if (vs.n != (int)in_shape.size())
        return;

    int last_non_full = -1;
    int64 stride = 1;
    int64 offset_elems = 0;
    for (int i = (int)in_shape.size() - 1; i >= 0; --i) {
        auto& s = vs.slices[i];
        if (s.is_var() || s.is_str() || s.is_none() || s.is_ellipsis())
            return;
        if (s.is_int()) {
            int64 v = s.i;
            if (v < 0) v += in_shape[i];
            if (v < 0 || v >= in_shape[i])
                return;
            offset_elems += v * stride;
            if (in_shape[i] != 1 && last_non_full < 0)
                last_non_full = i;
        } else if (s.is_slice()) {
            int64 start, stop;
            if (!slice_bounds(s, in_shape[i], start, stop))
                return;
            offset_elems += start * stride;
            if (!(start == 0 && stop == in_shape[i]) && last_non_full < 0)
                last_non_full = i;
        } else {
            return;
        }
        stride *= in_shape[i];
    }

    if (last_non_full < 0) {
        ou->share_with(in, offset_elems * in->dsize());
        op->ns.set(GetitemOp::_inplace);
        return;
    }

    for (int i = 0; i < last_non_full; ++i) {
        auto& s = vs.slices[i];
        if (s.is_int())
            continue;
        if (!s.is_slice())
            return;
        if (slice_len(s, in_shape[i]) != 1)
            return;
    }
    for (int i = last_non_full + 1; i < (int)in_shape.size(); ++i) {
        if (!full_slice(vs.slices[i], in_shape[i]))
            return;
    }

    int64 offset = offset_elems * in->dsize();
    ASSERT(offset>=0 && offset<=in->size);
    ou->share_with(in, offset);
    op->ns.set(GetitemOp::_inplace);
}

void SetitemOp::graph_optimize() {
    // LOGir << "hello graph_optimize";
    setitem_inplace(this);
    (void*)setitem_inplace;
}

void GetitemOp::graph_optimize() {
    // This optimize is still WIP
    // LOGir << "hello getitem graph_optimize";
    // setitem_grad_opt(this);
    // (void)getitem_inplace;
    getitem_inplace(this);
    (void*)getitem_inplace;
    getitem_contiguous_inplace(this);
    (void*)getitem_contiguous_inplace;
}

}
