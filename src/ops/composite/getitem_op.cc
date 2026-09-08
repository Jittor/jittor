// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include "core/var.h"
#include "core/executor.h"
#include "ops/composite/getitem_op.h"
#include "ops/op_register.h"
#ifndef JIT
#include "utils/stack_vector.h"
#endif

namespace jittor {

#ifndef JIT


static auto make_number = op_constructor<VarPtr, float, Var*>("number");
static auto make_unary = op_constructor<VarPtr, Var*, NanoString>("unary");
static auto make_empty = op_constructor<VarPtr, NanoVector, NanoString>("empty");
static auto make_setitem = op_constructor<VarPtr, Var*, VarSlices&&, Var*, NanoString>("setitem");

GetitemOp::GetitemOp(Var* x, VarSlices&& slices)
    : vs(move(slices)) {
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_has_gopt);
    set_flag(OpFlags::_manual_set_vnbb);
    for (int i=0; i<vs.n; i++)
        if (vs.slices[i].is_var())
            vs.slices[i].var->set_flag(VarFlags::_needed_by_backward);
    create_output(nullptr, x->dtype());
}

GetitemOp::GetitemOp(Var* x, VarSlices&& slices, int _) 
    : vs(move(slices)) {
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_has_gopt);
    set_flag(OpFlags::_custom_flag);
    set_flag(OpFlags::_grads);
    set_flag(OpFlags::_manual_set_vnbb);
    for (int i=0; i<vs.n; i++)
        if (vs.slices[i].is_var())
            vs.slices[i].var->set_flag(VarFlags::_needed_by_backward);
    create_output(nullptr, x->dtype());
    auto out2 = create_output(nullptr, x->dtype());
    out2->share_with(x);
    ns.data = _;
}

void infer_index_slices(Var* in, VarSlices& vs, int& first_oid_of_var, int& var_dim,
    StackVector<>& __restrict__ i_to_vs, 
    StackVector<>& __restrict__ i_to_o,
    StackVector<>& __restrict__ out_shape
) {
    auto in_shape = in->shape;
    auto nin = in_shape.size();
    i_to_vs.n = i_to_o.n = nin;
    out_shape.n = 0;

    int vid = 0;
    first_oid_of_var = -1;
    var_dim = 0;
    for (int i=0; i<nin; i++) {
        auto& s = vs.slices[vid];
        if (vid >= vs.n) {
            // i i i 
            // | | |
            // v v v --> overflow
            // s s
            i_to_vs[i] = -1;
            i_to_o[i] = out_shape.size();
            out_shape.push_back(in_shape[i]);
        } else
        if (s.is_var()) {
            // i --> s ---> o
            //       + ---> o
            // var maybe multiple dims
            if (first_oid_of_var == -1) {
                for (int i=0; i<vs.n; i++)
                    if (vs.slices[i].is_var())
                        var_dim = std::max(var_dim, vs.slices[i].var->shape.size());
                first_oid_of_var = out_shape.size();
                for (int j=0; j<var_dim; j++) {
                    out_shape.push_back(1);
                }
            }
            i_to_vs[i] = vid++;
            i_to_o[i] = -1;
            auto iv = s.var;
            auto iv_shape = iv->shape;
            auto niv = iv_shape.size();
            for (int j=0; j<niv; j++) {
                auto iv_shape_j = iv_shape[niv-j-1];
                auto& out_shape_j = out_shape[first_oid_of_var+var_dim-j-1];
                USER_CHECK(out_shape_j == iv_shape_j || out_shape_j == 1 || iv_shape_j == 1) << "Shape not match " >> out_shape_j >> "!="
                    >> iv_shape_j << "data shape:" << in_shape <<
                    "slice shape:" << iv_shape;
                if (out_shape_j == 1)
                    out_shape_j = iv_shape_j;
            }
        } else
        if (s.is_ellipsis()) {
            auto remain_slice = vs.n-vid-1;
            for (int i=vid+1; i<vs.n; i++)
                if (vs.slices[i].is_none())
                    remain_slice--;
            auto remain_idims = nin-i;
            auto ellipsis_size = remain_idims - remain_slice;
            ASSERT(ellipsis_size>=0) << "NDims not match";
            for (int j=0; j<ellipsis_size; j++) {
                i_to_vs[i+j] = -1;
                i_to_o[i+j] = out_shape.size();
                out_shape.push_back(in_shape[i+j]);
            }
            vid ++;
            i += ellipsis_size-1;
        } else
        if (s.is_none()) {
            i--;
            out_shape.push_back(1);
            vid++;
            continue;
        } else
        if (s.is_int()) {
            i_to_vs[i] = vid++;
            i_to_o[i] = -1;
            auto in_shape_i = in_shape[i];
            auto& v = s.slice.start;
            if (v<0) v += in_shape_i;
            USER_CHECK(v>=0 && v<in_shape_i) << "slice overflow, " << v << "not in [0,">>in_shape_i>>")";
        } else 
        if (s.is_str()) {
            i_to_vs[i] = vid++;
            i_to_o[i] = -1;
        } else {
            // slice
            auto& slice = s.slice;
            auto in_shape_i = in_shape[i];
            auto out_shape_j = in_shape_i;
            if (slice.mask == 7) {
                // slice is a[::]
                // start, stop, step is not filled
                vid++;
                i_to_vs[i] = -1;
                i_to_o[i] = out_shape.size();
                out_shape.push_back(out_shape_j);
            } else {
                i_to_vs[i] = vid++;
                i_to_o[i] = out_shape.size();
                if (in_shape_i > 0) {
                    slice.fill(in_shape_i);
                    if (std::abs(slice.step) <= 1)
                        out_shape_j = (slice.stop - slice.start) * slice.step;
                    else if (slice.step>0)
                        out_shape_j = (slice.stop - slice.start - 1) / slice.step + 1;
                    else
                        out_shape_j = (slice.start - slice.stop - 1) / -slice.step + 1;
                    out_shape_j = std::max((int64)0, out_shape_j);
                }
                out_shape.push_back(out_shape_j);
            }
        }
    }
    while (vid < vs.n) {
        auto& s = vs.slices[vid++];
        if (s.is_none()) {
            out_shape.push_back(1);
        } else
            USER_CHECK(s.is_ellipsis()) << "Too many slices" << vs << "shape:" << in->shape;
    }
}


void GetitemOp::infer_slices(StackVector<>& i_to_vs, StackVector<>& i_to_o,
                            StackVector<>& out_shape) {
    infer_index_slices(inputs().front(), vs, first_oid_of_var, var_dim,
                       i_to_vs, i_to_o, out_shape);
}

void GetitemOp::infer_shape() {
    auto in = inputs().front();
    auto out = outputs().front();
    auto in_shape = in->shape;
    auto nin = in_shape.size();

    StackVector<> i_to_vs(nin);
    StackVector<> i_to_o(nin);
    // shape return to use
    StackVector<> out_shape;
    infer_slices(i_to_vs, i_to_o, out_shape);
    storage_view = outputs().size() == 1 && in->num >= 0;
    vector<int64> storage_steps(out_shape.size(), 0);
    int64 storage_offset = 0;
    for (int i=0; i<nin && storage_view; ++i) {
        const int vid = i_to_vs[i], oid = i_to_o[i];
        if (vid < 0) {
            storage_steps[oid] = in->storage_stride(i);
        } else {
            const auto& slice = vs.slices[vid];
            if (slice.is_int()) storage_offset += slice.i * in->storage_stride(i);
            else if (slice.is_slice() && slice.slice.step > 0) {
                storage_offset += slice.slice.start * in->storage_stride(i);
                storage_steps[oid] = slice.slice.step * in->storage_stride(i);
            } else storage_view = false;
        }
    }
    
    // this will cause save checkpoint failed.
    // if (out_shape.n == 0)
    //     out->set_flag(VarFlags::_is_scalar);
    // optimized shape (each dim is a loop var)
    StackVector<> o_shape;
    int fov = -1;
    for (int i=0; i<nin; i++) {
        auto& vid = i_to_vs[i];
        auto& oid = i_to_o[i];
        auto os = out_shape[oid];
        if (oid>=0) {
            if (vid==-1 && i && i_to_vs[i-1]<0) {
                vid = -2;
                o_shape.back() *= os;
            } else
                o_shape.push_back(os);
            oid = o_shape.size()-1;
        } else {
            auto& s = vs.slices[vid];
            if (s.is_var() && fov == -1) {
                fov = o_shape.size();
                for (int i=0; i<var_dim; i++)
                    o_shape.push_back(out_shape[first_oid_of_var+i]);
            }
        }
    }
    first_oid_of_var = fov;

    out->set_shape(out_shape.to_nano_vector());
    if (storage_view) {
        out->set_storage_strides(NanoVector::make(storage_steps.data(), storage_steps.size()));
        out->share_with(in, storage_offset * in->dsize());
    }
    if (!out_shape.size()) out->set_flag(VarFlags::_is_scalar);

    this->i_to_vs = i_to_vs.to_nano_vector();
    this->i_to_o = i_to_o.to_nano_vector();
    this->o_shape = o_shape.to_nano_vector();
    if (outputs().size() > 1) {
        auto out2 = output(1);
        out2->set_shape(in->shape);
        out2->storage_strides = in->storage_strides;
    }

    LOGV(999) << "\ni_to_vs:" << i_to_vs
        << "\ni_to_o:" << i_to_o
        << "\no_shape:" << o_shape;
}

VarPtr GetitemOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    if (v_index)
        return nullptr;
    auto zeros = make_number(0, v);
    // TODO: maybe add here?
    // need analysis the overlap attr os var slices
    for (int i=0; i<vs.n; i++)
        if (vs.slices[i].is_var()) {
            return make_setitem(zeros, VarSlices(vs, true), dout, ns_add);
        }
    VarPtr value = dout;
    if (value->dtype() != zeros->dtype()) value = make_unary(value, zeros->dtype());
    return make_setitem(zeros, VarSlices(vs, true), value, ns_void);
}

void GetitemOp::grads(Var** dout, VarPtr* dins) {
    VarPtr x = dout[1];
    VarPtr y = dout[0];
    if (!x) {
        auto in = inputs().front();
        // ns.data represents this is the last split var
        if (ns.data)
            x = make_empty(in->shape, in->dtype());
        else
            x = make_number(0, in);
    }
    if (!y) {
        y = make_number(0, outputs().front());
    }
    if (y->dtype() != x->dtype()) y = make_unary(y, x->dtype());
    dins[0] = make_setitem(x, VarSlices(vs, true), y, ns_void);
}

void GetitemOp::jit_prepare(JK& jk) {
    auto in = inputs().front();
    int idim = i_to_vs.size();
    jk << "«Ti:" << in->dtype();
    jk << "«IDIM=" << JK::hex1(i_to_vs.size());
    jk << "«ODIM=" << JK::hex1(o_shape.size());
    if (first_oid_of_var>=0) {
        jk << "«FOV=" << JK::hex1(first_oid_of_var);
        jk << "«VD=" << JK::hex1(var_dim);
    }
    for (int i=0; i<idim; i++) {
        auto iv = i_to_vs[i];
        auto io = i_to_o[i];
        jk << "«IV" << JK::hex1(i) << ':' << JK::shex1(iv);
        jk << "«IO" << JK::hex1(i) << ':' << JK::shex1(io);
        auto& v = vs.slices[iv];
        if (iv>=0 && io==-1) {
            if (v.is_int()) {
                jk << "«VS" << JK::hex1(i) << ":-1";
            } else
            if (v.is_str()) {
                jk << "«VS" << JK::hex1(i) << ":-5";
                jk << "«VSS" << JK::hex1(i) << ":" << v.get_str();
            } else {
                ASSERT(v.is_var());
                auto var = v.var;
                auto vshape = var->shape;
                auto vdim = vshape.size();
                int vsmask = 0;
                for (int j=0; j<vdim; j++) {
                    int k = first_oid_of_var+j+var_dim-vdim;
                    if (vshape[j] == o_shape[k])
                        vsmask |= 1<<(j+var_dim-vdim);
                }
                jk << "«VS" << JK::hex1(i) << '=' << JK::hex(vsmask);
                jk << "«VST" << JK::hex1(i) << ':' << var->dtype();
            }
        } else
        if (iv>=0 && io>=0) {
            ASSERT(v.is_slice());
            jk << "«VS" << JK::hex1(i) << ':';
            if (std::abs(v.slice.step) <= 1)
                jk << JK::shex1(v.slice.step);
            else
                jk << '0';
        }
    }
}

#else // JIT

#pragma GCC diagnostic ignored "-Wunused-variable"

void GetitemOp::jit_run() {
    auto in = inputs().front();
    auto out = outputs().front();
    if (out->num == 0) return;
    if (ns.get(GetitemOp::_inplace) &&
        in->shares_allocation_with(out))
        return;

    @for(i, 0, ODIM, index_t oshape@i = o_shape[@i];)
    @if(ODIM>0,
        index_t ostride@{ODIM-1} = 1;
        @for(i, ODIM-2, -1, -1, index_t ostride@i = ostride@{i+1} * oshape@{i+1};)
    )
    Ti* op = out->ptr<Ti>();

    Ti* ip = in->ptr<Ti>();
    @for(i, 0, IDIM, index_t ishape@i = 
        @if(IV@i==-1,oshape@{IO@i},
        @if(IV@i==-2,1,in->shape[@i]));
    )
    @for(i, 0, IDIM, index_t istride@i = in->storage_stride(@i);)

    
    @for(i, 0, IDIM, 
        @if(IV@i>=0 && IO@i>=0, 
            index_t vstart@i = vs.slices[@{IV@i}].slice.start;
            index_t vstep@i = @if(VS@i==0,vs.slices[@{IV@i}].slice.step;,@{VS@i});
        )
    )
    
    @for(i, 0, IDIM, 
        @if(IV@i>=0 && IO@i<0, 
            @if(VS@i==-1,index_t vi@i = vs.slices[@{IV@i}].slice.start;);
        )
    )
    
    @for(i, 0, IDIM, 
        @if(IV@i>=0 && IO@i<0, 
            @if(VS@i>=0,
                VST@i* vp@i = vs.slices[IV@i].var->ptr<VST@i>();
                // Scalar index tensors have no strides and use vp[0] below.
                @if(VD>0,
                    index_t vs@i@@s@{VD-1} = 1;
                    @for(j,VD-2,-1,-1,index_t vs@i@@s@j = vs@i@@s@{j+1} *
                        @if((VS@i>>(j+1))&1,oshape@{j+1+FOV},1);
                    )
                )
            );
        )
    )
    
    

    // `oid` is the output's linear index, so every iteration writes its own
    // element and the nest collapses. The depth has to be a literal: a
    // `#pragma` line is not run through the template substitution (only the
    // `@if` around it is), and building it with `_Pragma` instead makes
    // KernelIR read it as a function definition and abort. The `JIT_cpu` guard
    // excludes host pragmas when a backend specializes this common body.
    // A gather does not vectorise either way, so the `if`
    // clause costs nothing here.
    @if(@is_def(JIT_cpu) && ODIM>0, index_t o_total = 1 @for(d, 0, ODIM, * oshape@d);)
    @if(@is_def(JIT_cpu) && ODIM==1, #pragma omp parallel for if(o_total >= 65536))
    @if(@is_def(JIT_cpu) && ODIM==2, #pragma omp parallel for collapse(2) if(o_total >= 65536))
    @if(@is_def(JIT_cpu) && ODIM==3, #pragma omp parallel for collapse(3) if(o_total >= 65536))
    @if(@is_def(JIT_cpu) && ODIM>=4, #pragma omp parallel for collapse(4) if(o_total >= 65536))
    @for(d, 0, ODIM, for (index_t i@d=0; i@d < oshape@d; i@d++)) {
        index_t oid = 0 @for(d, 0, ODIM, + i@d * ostride@d);
        @for(d, 0, IDIM, index_t iid@d = 
            @if(IV@d==-1, i@{IO@d},
            @if(IV@d==-2, 0,
            @if(IO@d!=-1, (i@{IO@d}*vstep@d+vstart@d),
            @if(VS@d==-1, vi@d,
            @if(VS@d==-5, VSS@d,
            @if(VS@d>=0,
                index_t(vp@d[0 @for(j,0,VD,@if((VS@d>>j)&1, + i@{j+FOV} * vs@d@@s@j,))])
            , ??? ))))));
        )
        @for(d, 0, IDIM, if (iid@d < 0) iid@d += ishape@d;
        )
        auto iid = 0 @for(d, 0, IDIM,  + iid@d * istride@d);
        op[oid] = ip[iid];
    }
}
#endif // JIT

} // jittor
