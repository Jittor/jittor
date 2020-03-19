// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include "var.h"
#include "ops/code_op.h"
#include "ops/op_register.h"

#ifndef JIT

namespace jittor {

static auto make_code = get_op_info("code")
    .get_constructor<VarPtr, NanoVector, NanoString, vector<Var*>&&, string&&, vector<string>&&, string&&>();
    
CodeOp::CodeOp(NanoVector shape, NanoString dtype, vector<Var*>&& inputs, string&& cpu_src, vector<string>&& cpu_grad_src, string&& header)
    : in(inputs), cpu_src(move(cpu_src)), cpu_grad_src(move(cpu_grad_src)), header(move(header)) {
    flags.set(NodeFlags::_cpu);
    out = create_output(shape, dtype);
    ASSERT(this->cpu_src.size());
    ASSERTop(inputs.size(),<=,10);
}

VarPtr CodeOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    // Do not have grad to extras input
    if (cpu_grad_src.size() <= v_index) return nullptr;
    auto src = cpu_grad_src[v_index];
    if (!src.size()) return nullptr;
    auto inputs = clone(in);
    inputs.push_back(out);
    inputs.push_back(dout);
    return make_code(
        in[v_index]->shape,
        in[v_index]->dtype(),
        move(inputs),
        move(src), {}, clone(header)
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
    add_jit_define("HEADER", header);
    add_jit_define("CODE", cpu_src);
}

} // jittor

#else // JIT

@HEADER

namespace jittor {

#pragma GCC diagnostic ignored "-Wunused-variable"
void CodeOp::jit_run() {
    // define inputs
    @for(i, 0, INSIZE,
        auto in@i = in[@i];
        auto* __restrict__ in@i@@p = in[@i]->ptr<Tin@i>();
        @for(j, 0, INDIM@i, index_t in@i@@shape@j = in[@i]->shape[@j];)
        index_t in@i@@stride@{INDIM@i-1} = 1;
        @for(j, INDIM@i-2, -1, -1, auto in@i@@stride@j = in@i@@stride@{j+1} * in@i@@shape@{j+1};)
    )
    // define out
    auto* __restrict__ outp = out->ptr<Tout>();
    @for(i, 0, OUTDIM, index_t outshape@i = out->shape[@i];)
    index_t outstride@{OUTDIM-1} = 1;
    @for(i, OUTDIM-2, -1, -1, auto outstride@i = outstride@{i+1} * outshape@{i+1};)

    @if(INSIZE>=2,
        auto pout = in[@{INSIZE-2}];
        auto* __restrict__ poutp = pout->ptr<Tpout>();
        @for(i, 0, POUTDIM, index_t poutshape@i = pout->shape[@i];)
        index_t poutstride@{POUTDIM-1} = 1;
        @for(i, POUTDIM-2, -1, -1, auto poutstride@i = poutstride@{i+1} * poutshape@{i+1};)

        auto dout = in[@{INSIZE-1}];
        auto* __restrict__ doutp = dout->ptr<Tdout>();
        @for(i, 0, DOUTDIM, index_t doutshape@i = dout->shape[@i];)
        index_t doutstride@{DOUTDIM-1} = 1;
        @for(i, DOUTDIM-2, -1, -1, auto doutstride@i = doutstride@{i+1} * doutshape@{i+1};)
    ,)
    @CODE
}

} // jittor

#endif // JIT
