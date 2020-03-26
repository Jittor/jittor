// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include "var.h"
#include "ops/code_op.h"
#include "ops/op_register.h"
#include "misc/cuda_flags.h"

#ifndef JIT

namespace jittor {

static auto make_code = get_op_info("code")
    .get_constructor<VarPtr, NanoVector, NanoString, vector<Var*>&&, string&&, vector<string>&&, string&&, string&&, vector<string>&&, string&&>();
    
CodeOp::CodeOp(NanoVector shape, NanoString dtype, vector<Var*>&& inputs, 
    string&& cpu_src, vector<string>&& cpu_grad_src, string&& cpu_header, 
    string&& cuda_src, vector<string>&& cuda_grad_src, string&& cuda_header)
    : in(inputs), cpu_src(move(cpu_src)), cpu_grad_src(move(cpu_grad_src)), cpu_header(move(cpu_header)),
    cuda_src(move(cuda_src)), cuda_grad_src(move(cuda_grad_src)), cuda_header(move(cuda_header))
{
    flags.set(NodeFlags::_cpu, !!this->cpu_src.size());
    flags.set(NodeFlags::_cuda, !!this->cuda_src.size());
    out = create_output(shape, dtype);
    ASSERTop(inputs.size(),<=,10);
}

VarPtr CodeOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    // Do not have grad to extras input
    string cpu_src = v_index < cpu_grad_src.size() ? cpu_grad_src[v_index] : "";
    string cuda_src = v_index < cuda_grad_src.size() ? cuda_grad_src[v_index] : "";
    if (!cuda_src.size() && !cpu_src.size()) return nullptr;
    auto inputs = clone(in);
    inputs.push_back(out);
    inputs.push_back(dout);
    return make_code(
        in[v_index]->shape,
        in[v_index]->dtype(),
        move(inputs),
        move(cpu_src), {}, clone(cpu_header),
        move(cuda_src), {}, clone(cuda_header)
    );
}

void CodeOp::jit_prepare() {
    add_jit_define("Tout", out->dtype());
    add_jit_define("OUTDIM", JK::hex1(out->shape.size()));
    if (in.size()>=2) {
        auto pout = in.rbegin()[1];
        auto dout = in.rbegin()[0];
        add_jit_define("Tpout", pout->dtype());
        add_jit_define("POUTDIM", JK::hex1(pout->shape.size()));
        add_jit_define("Tdout", dout->dtype());
        add_jit_define("DOUTDIM", JK::hex1(dout->shape.size()));
    }
    add_jit_define("INSIZE", JK::hex1(in.size()));
    for (uint i=0; i<in.size(); i++) {
        add_jit_define("INDIM", JK::hex1(i), JK::hex1(in[i]->shape.size()));
        add_jit_define("Tin", JK::hex1(i), in[i]->dtype());
    }
    if (use_cuda) {
        jk << JK::key << "HEADER" << JK::val << cuda_header;
        ASSERT(cuda_src.size());
        jk << "\nnamespace jittor {\n";
        int i=0;
        // move cuda kernel function into header
        for (; i<cuda_src.size(); i++) {
            if (cuda_src[i] == ' ' || cuda_src[i] == '\t' || cuda_src[i] == '\n') {
                jk << cuda_src[i];
            } else
            if (cuda_src[i] == '_') {
                int presum = 0;
                while (i < cuda_src.size()) {
                    jk << cuda_src[i];
                    if (cuda_src[i] == '{') presum ++;
                    else if (cuda_src[i] == '}') {
                        presum--;
                        if (presum==0)
                            break;
                    }
                    i++;
                }
            } else break;
        }
        jk << "}" << JK::end << JK::key << "CODE" << JK::val;
        for (; i<cuda_src.size(); i++) jk << cuda_src[i];
        jk << JK::end;
    } else {
        add_jit_define("HEADER", cpu_header);
        add_jit_define("CODE", cpu_src);
        ASSERT(cpu_src.size());
    }
}

} // jittor

#else // JIT

#pragma GCC diagnostic ignored "-Wunused-variable"

@for(i, 0, INSIZE,
    @define(in@i@@stride@{INDIM@i-1},1)
)
@define(outstride@{OUTDIM-1},1)
@if(INSIZE>=2,
    @define(poutstride@{POUTDIM-1},1)
    @define(doutstride@{DOUTDIM-1},1)
,)

@define(ARGS_DEF, 
@for(i, 0, INSIZE, @(
    Tin@i* __restrict__ in@i@@p,
    @for(j, 0, INDIM@i, @(index_t in@i@@shape@j,))
))
@for(i, 0, OUTDIM, @(index_t outshape@i,))
Tout* __restrict__ outp
)

@define(ARGS, 
@for(i, 0, INSIZE, @(
    in@i@@p,
    @for(j, 0, INDIM@i, @(in@i@@shape@j,))
))
@for(i, 0, OUTDIM, @(outshape@i,))
outp
)

@define(PRECALC,
@for(i, 0, INSIZE,
    @for(j, INDIM@i-2, -1, -1, auto in@i@@stride@j = in@i@@stride@{j+1} * in@i@@shape@{j+1};)
)
@for(i, OUTDIM-2, -1, -1, auto outstride@i = outstride@{i+1} * outshape@{i+1};)
@if(INSIZE>=2,
    auto* __restrict__ poutp = in@{INSIZE-2}@@p;
    @for(i, 0, POUTDIM, index_t poutshape@i = in@{INSIZE-2}@@shape@i;)
    @for(i, POUTDIM-2, -1, -1, auto poutstride@i = in@{INSIZE-2}@@stride@i;)

    auto* __restrict__ doutp = in@{INSIZE-1}@@p;
    @for(i, 0, DOUTDIM, index_t doutshape@i = in@{INSIZE-1}@@shape@i;)
    @for(i, DOUTDIM-2, -1, -1, auto doutstride@i = in@{INSIZE-1}@@stride@i;)
,)
)


@HEADER

namespace jittor {

void CodeOp::jit_run() {
    // define inputs
    @for(i, 0, INSIZE,
        auto in@i = in[@i];
        auto* __restrict__ in@i@@p = in[@i]->ptr<Tin@i>();
        @for(j, 0, INDIM@i, index_t in@i@@shape@j = in[@i]->shape[@j];)
    )
    // define out
    auto* __restrict__ outp = out->ptr<Tout>();
    @for(i, 0, OUTDIM, index_t outshape@i = out->shape[@i];)

    @PRECALC

    @CODE
}

} // jittor

#endif // JIT
