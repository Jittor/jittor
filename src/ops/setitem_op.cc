// ***************************************************************
// Copyright (c) 2020 Jittor. All Rights Reserved.
// Authors: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include "var.h"
#include "ops/setitem_op.h"
#include "ops/getitem_op.h"
#ifdef JIT
#include "ops/binary_op_defs.h"
#ifdef JIT_cuda
#include <cuda_runtime.h>
#include <helper_cuda.h>
#endif
#else
#include "ops/op_register.h"
#ifdef HAS_CUDA
#include "misc/cuda_flags.h"
#endif
#endif

namespace jittor {

#ifndef JIT

static auto make_array = get_op_info("array")
    .get_constructor<VarPtr, const void*, NanoVector, NanoString>();
static auto make_number = get_op_info("number")
    .get_constructor<VarPtr, float, Var*>();
static auto make_getitem = get_op_info("getitem")
    .get_constructor<VarPtr, Var*, VarSlices&&>();
static auto make_setitem = get_op_info("setitem")
    .get_constructor<VarPtr, Var*, VarSlices&&, Var*, NanoString>();
static auto make_binary = get_op_info("binary")
    .get_constructor<VarPtr, Var*, Var*, NanoString>();
static auto make_unary = get_op_info("unary")
    .get_constructor<VarPtr, Var*, NanoString>();

SetitemOp::SetitemOp(Var* x, VarSlices&& slices, Var* y, NanoString op)
    : vs(move(slices)), op(op) {
    flags.set(NodeFlags::_cpu);
    flags.set(NodeFlags::_cuda);
    flags.set(NodeFlags::_has_gopt);
    ASSERT(ns == ns_void || ns.is_binary());
    create_output(nullptr, x->dtype());
}

void SetitemOp::infer_shape() {
    auto in = inputs().front();
    auto data = input(1);
    auto out = outputs().front();
    auto in_shape = in->shape;
    auto nin = in_shape.size();

    StackVector<> i_to_vs(nin);
    StackVector<> i_to_o(nin);
    // shape return to use
    StackVector<> out_shape;
    ((GetitemOp*)this)->infer_slices(i_to_vs, i_to_o, out_shape);
    if (!out_shape.size()) out_shape.push_back(1);

    // get broadcast mask of set value
    auto data_shape = data->shape;
    auto data_dim = data_shape.size();
    int bmask = 0;
    int bmask2 = 0;

    ASSERTop(data_dim,<=,out_shape.size()) << "Data dimension not match";
    for (int i=0; i<data_dim; i++) {
        int j = i - data_dim + out_shape.size();
        if (!(data_shape[i]==1 && out_shape[j]!=-1)) {
            ASSERTop(data_shape[i],==,out_shape[j]) << "Data shape not match" << data_shape << out_shape;
            bmask |= 1<<j;
        }
    }

    // optimized shape (each dim is a loop var)
    StackVector<> o_shape;
    int fov = -1;
    for (int i=0; i<nin; i++) {
        auto& vid = i_to_vs[i];
        auto& oid = i_to_o[i];
        auto os = out_shape[oid];
        if (oid>=0) {
            if (vid==-1 && i && i_to_vs[i-1]<0 
                && ((bmask>>oid)&1) == ((bmask>>(oid-1))&1)) 
                // same broadcast condition with prev dim
            {
                vid = -2;
                o_shape.back() *= os;
            } else {
                o_shape.push_back(os);
                // fix bmask2 offset
                bmask2 |= ((bmask>>oid)&1) << (o_shape.size()-1);
            }
            oid = o_shape.size()-1;
        } else {
            auto& s = vs.slices[vid];
            if (s.is_var() && fov == -1) {
                fov = o_shape.size();
                for (int i=0; i<var_dim; i++) {
                    o_shape.push_back(out_shape[first_oid_of_var+i]);
                    // fix bmask2 offset
                    bmask2 |= ((bmask>>(first_oid_of_var+i))&1) << (o_shape.size()-1);
                }
            }
        }
    }
    first_oid_of_var = fov;
    this->bmask = bmask2;

    out->set_shape(in_shape);

    this->i_to_vs = i_to_vs.to_nano_vector();
    this->i_to_o = i_to_o.to_nano_vector();
    this->o_shape = o_shape.to_nano_vector();

    LOGvvvv << "\ni_to_vs:" << i_to_vs
        << "\ni_to_o:" << i_to_o
        << "\no_shape:" << o_shape;
}

VarPtr SetitemOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    if (v_index >= 2)
        return nullptr;
    if (op == ns_void) {
        if (v_index == 0) {
            float32 number = 0;
            VarPtr zero = make_array(&number, 1, ns_float32);
            return make_setitem(dout, VarSlices(vs), zero, ns_void);
        } else {
            return make_getitem(dout, VarSlices(vs));
        }
    }
    if (op == ns_add) {
        if (v_index == 0) {
            return dout;
        } else {
            return make_getitem(dout, VarSlices(vs));
        }
    }
    if (op == ns_subtract) {
        if (v_index == 0) {
            return dout;
        } else {
            return make_unary(make_getitem(dout, VarSlices(vs)), ns_negative);
        }
    }
    if (op == ns_multiply) {
        if (v_index == 0) {
            return make_setitem(dout, VarSlices(vs), input(1), ns_multiply);
        } else {
            return make_binary(
                make_getitem(inputs().front(), VarSlices(vs)),
                make_getitem(dout, VarSlices(vs)), ns_multiply);
        }
    }
    if (op == ns_divide) {
        if (v_index == 0) {
            return make_setitem(dout, VarSlices(vs), input(1), ns_divide);
        } else {
            // dy = -dz*x / y^2
            auto dout2 = make_getitem(dout, VarSlices(vs));
            auto x = make_getitem(inputs().front(), VarSlices(vs));
            auto y = v;
            auto ndz = make_unary(dout2, ns_negative);
            auto ndzx = make_binary(ndz, x, ns_multiply);
            auto y2 = make_binary(y, y, ns_multiply);
            return make_binary(ndzx, y2, ns_divide);
        }
    }
    LOGf << "Setitem grad of op" << op << "is not supported yet";
    return nullptr;
}

void SetitemOp::jit_prepare() {
    auto data = input(1);
    add_jit_define("OP", op);
    add_jit_define("Td", data->dtype());
    add_jit_define("BMASK", JK::hex(bmask));
    // TODO: merge code
    auto in = inputs().front();
    int idim = i_to_vs.size();
    add_jit_define("Ti", in->dtype());
    add_jit_define("IDIM", JK::hex1(i_to_vs.size()));
    add_jit_define("ODIM", JK::hex1(o_shape.size()));
    if (first_oid_of_var>=0) {
        add_jit_define("FOV", JK::hex1(first_oid_of_var));
        add_jit_define("VD", JK::hex1(var_dim));
    }
    for (int i=0; i<idim; i++) {
        auto iv = i_to_vs[i];
        auto io = i_to_o[i];
        add_jit_define("IV", JK::hex1(i), JK::shex1(iv));
        add_jit_define("IO", JK::hex1(i), JK::shex1(io));
        auto& v = vs.slices[iv];
        if (iv>=0 && io==-1) {
            if (v.is_int()) {
                add_jit_define("VS", JK::hex1(i), "-1");
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
                add_jit_define("VS", JK::hex1(i), JK::hex(vsmask));
                add_jit_define("VST", JK::hex1(i), var->dtype());
            }
        } else
        if (iv>=0 && io>=0) {
            ASSERT(v.is_slice());
            if (std::abs(v.slice.step) <= 1)
                add_jit_define("VS", JK::hex1(i), JK::shex1(v.slice.step));
            else
                add_jit_define("VS", JK::hex1(i), "0");
        }
    }
    #ifdef HAS_CUDA
    if (use_cuda) {
        int no = o_shape.size();
        int masks[no];
        int tdims[6];
        cuda_loop_schedule(o_shape, masks, tdims);
        for (int i=0; i<no; i++) {
            add_jit_define("LO", JK::hex1(i), JK::hex(masks[i]));
        }
    }
    #endif
}

void SetitemOp::compile_optimize(string& src) {
    ((GetitemOp*)this)->_compile_optimize(src);
}

#else // JIT

#pragma GCC diagnostic ignored "-Wunused-variable"

void SetitemOp::jit_run() {
    auto in = inputs().front();
    auto data = input(1);
    auto out = outputs().front();
    if (out->num == 0) return;

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
    index_t istride@{IDIM-1} = 1;
    @for(i, IDIM-2, -1, -1, index_t istride@i = istride@{i+1} * ishape@{i+1};)

    
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
                index_t vs@i@@s@{VD-1} = 1;
                VST@i* vp@i = vs.slices[IV@i].var->ptr<VST@i>();
                @for(j,VD-2,-1,-1,index_t vs@i@@s@j = vs@i@@s@{j+1} * 
                    @if((VS@i>>(j+1))&1,oshape@{j+1+FOV},1);
                )
            );
        )
    )

    Td* dp = data->ptr<Td>();
    @if(ODIM>0,
        index_t dstride@{ODIM-1} = 1;
        @for(i, ODIM-2, -1, -1, index_t dstride@i = dstride@{i+1} * @if((BMASK>>(i+1))&1,oshape@{i+1},1);)
    )
    #ifdef JIT_cpu
    if (op != ip)
        std::memcpy(op, ip, out->size);
    #else
    if (op != ip)
        checkCudaErrors(cudaMemcpyAsync(op, ip, out->size, cudaMemcpyDefault, 0));
    #endif

    @for(d, 0, ODIM, for (index_t i@d=0; i@d < oshape@d; i@d++)) {
        index_t did = 0 @for(d, 0, ODIM, @if((BMASK>>d)&1,+ i@d * dstride@d));
        @for(d, 0, IDIM, index_t iid@d = 
            @if(IV@d==-1, i@{IO@d},
            @if(IV@d==-2, 0,
            @if(IO@d!=-1, (i@{IO@d}*vstep@d+vstart@d),
            @if(VS@d==-1, vi@d,
            @if(VS@d>=0,
                index_t(vp@d[0 @for(j,0,VD,@if((VS@d>>j)&1, + i@{j+FOV} * vs@d@@s@j,))])
            , ??? )))));
        )
        auto iid = 0 @for(d, 0, IDIM,  + iid@d * istride@d);

        @if(@strcmp(@OP,void)==0,
            op[iid] = (Ti)dp[did],
            op[iid] = @expand_macro(@OP, Ti, op[iid], dp[did])
        );
    }
}
#endif // JIT

} // jittor