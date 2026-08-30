// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include "var.h"
#include "ops/code_op.h"
#include "ops/op_register.h"
#include "misc/cuda_flags.h"
#include <mutex>
#include <unordered_map>

#define __inline_static__ inline static

#ifndef JIT

namespace jittor {

static auto make_code = get_op_info("code")
    .get_constructor<VarPtr, NanoVector, NanoString, vector<Var*>&&, string&&, vector<string>&&, string&&, string&&, vector<string>&&, string&&, DataMap&&>();
    
static inline void check_vary_shape(NanoVector v) {
    ASSERT(v.size()) << "Vary shape should not be zero dimension";
    for (int i=0; i<v.size(); i++)
        ASSERT((i == 0) ^ (v[i] >= 0))
            << "Vary shape should only occur in the first dimension:" << v;
}

CodeOp::CodeOp(NanoVector shape, NanoString dtype, vector<Var*>&& inputs, 
    string&& cpu_src, vector<string>&& cpu_grad_src, string&& cpu_header, 
    string&& cuda_src, vector<string>&& cuda_grad_src, string&& cuda_header,
    DataMap&& data)
    : _inputs(inputs), cpu_src(move(cpu_src)), cpu_grad_src(move(cpu_grad_src)), cpu_header(move(cpu_header)),
    cuda_src(move(cuda_src)), cuda_grad_src(move(cuda_grad_src)), cuda_header(move(cuda_header)), 
    data(move(data))
{
    flags.set(NodeFlags::_cpu, !!this->cpu_src.size());
    flags.set(NodeFlags::_cuda, !!this->cuda_src.size());
    _outputs.push_back(create_output(shape, dtype));

    if (_outputs[0]->num < 0) {
        check_vary_shape(_outputs[0]->shape);
    }
    if (this->cuda_grad_src.size() == 0 && this->cpu_grad_src.size() == 0) {
        flags.set(NodeFlags::_manual_set_vnbb);
    }
}


CodeOp::CodeOp(
    vector<NanoVector>&& shapes, vector<NanoString>&& dtypes, vector<Var*>&& inputs, 
    string&& cpu_src, vector<string>&& cpu_grad_src, string&& cpu_header, 
    string&& cuda_src, vector<string>&& cuda_grad_src, string&& cuda_header,
    DataMap&& data)
    : _inputs(inputs), cpu_src(move(cpu_src)), cpu_grad_src(move(cpu_grad_src)), cpu_header(move(cpu_header)),
    cuda_src(move(cuda_src)), cuda_grad_src(move(cuda_grad_src)), cuda_header(move(cuda_header)), 
    data(move(data))
{
    flags.set(NodeFlags::_cpu, !!this->cpu_src.size());
    flags.set(NodeFlags::_cuda, !!this->cuda_src.size());
    CHECKop(shapes.size(),==,dtypes.size()) << "Number of outputs' shapes and dtypes should be the same";
    _outputs.resize(shapes.size());
    CHECKop(_outputs.size(),>,0);
    for (int i=0; i<shapes.size(); i++) {
        _outputs[i] = create_output(shapes[i], dtypes[i]);
        if (_outputs[i]->num < 0) {
            check_vary_shape(_outputs[i]->shape);
        }
    }
    if (this->cuda_grad_src.size() == 0 && this->cpu_grad_src.size() == 0)
        flags.set(NodeFlags::_manual_set_vnbb);
}

CodeOp::CodeOp(
    vector<Var*>&& inputs, vector<Var*>&& outputs, 
    string&& cpu_src, vector<string>&& cpu_grad_src, string&& cpu_header, 
    string&& cuda_src, vector<string>&& cuda_grad_src, string&& cuda_header,
    DataMap&& data)
    : _inputs(inputs), cpu_src(move(cpu_src)), cpu_grad_src(move(cpu_grad_src)), cpu_header(move(cpu_header)),
    cuda_src(move(cuda_src)), cuda_grad_src(move(cuda_grad_src)), cuda_header(move(cuda_header)), 
    data(move(data))
{
    flags.set(NodeFlags::_cpu, !!this->cpu_src.size());
    flags.set(NodeFlags::_cuda, !!this->cuda_src.size());
    _outputs.resize(outputs.size());
    CHECKop(_outputs.size(),>,0);
    for (int i=0; i<outputs.size(); i++) {
        auto o = outputs[i];
        _outputs[i] = create_output(o->shape, o->dtype());
        _outputs[i]->share_with(o);
        /*
            TODO: vary shape not allowed in direct output
        */
    }
    if (this->cuda_grad_src.size() == 0 && this->cpu_grad_src.size() == 0)
        flags.set(NodeFlags::_manual_set_vnbb);
}


VarPtr CodeOp::grad(Var* out, Var* dout, Var* v, int v_index) {
    // Do not have grad to extras input
    string cpu_src = v_index < cpu_grad_src.size() ? cpu_grad_src[v_index] : "";
    string cuda_src = v_index < cuda_grad_src.size() ? cuda_grad_src[v_index] : "";
    if (!cuda_src.size() && !cpu_src.size()) return nullptr;
    auto inputs = clone(_inputs);
    // TODO: remove unused deps
    // dout -> dout
    std::stringstream new_alias;
    new_alias << "\n@alias(dout,in" << JK::dec3(inputs.size()) << ")\n";
    inputs.push_back(dout);
    // _outputs[i] -> poutj
    for (int i=0; i<_outputs.size(); i++) {
        new_alias << "\n@alias(pout" << JK::dec3(i) << ",in" << JK::dec3(inputs.size()) << ")\n";
        if (_outputs[i] == out)
            new_alias << "\n@alias(pout,in" << JK::dec3(inputs.size()) << ")\n";
        inputs.push_back(_outputs[i]);
    }
    auto alias = new_alias.str();
    return make_code(
        _inputs[v_index]->shape,
        _inputs[v_index]->dtype(),
        move(inputs),
        move(cpu_src), {}, alias+cpu_header,
        move(cuda_src), {}, alias+cuda_header,
        DataMap(data)
    );
}


// The part of the key after "«HEADER:" is a pure function of the header and the
// source: it walks the source once, character by character, to lift the CUDA
// kernel definitions into the header. That walk was re-run on EVERY call, and an
// inference kernel's source runs to well over a kilobyte -- a four-figure loop of
// single-character appends in front of a kernel that itself takes about a
// microsecond. The distinct sources are few (one per specialised kernel), so
// remember the result. References into an unordered_map stay valid across
// rehashes, and the lock only guards the map, not the walk's cost.
static const string& code_op_key_tail(const string& header, const string& src) {
    static std::mutex tail_lock;
    static unordered_map<string, string> tails;
    string key;
    key.reserve(header.size() + src.size() + 1);
    key += header;
    key += '\1';
    key += src;
    std::lock_guard<std::mutex> guard(tail_lock);
    auto found = tails.find(key);
    if (found != tails.end())
        return found->second;
    string tail;
    tail.reserve(header.size() + src.size() + 32);
    tail += header;
    tail += "\nnamespace jittor {\n";
    size_t i = 0;
    // move cuda kernel function into header
    for (; i<src.size(); i++) {
        if (src[i] == ' ' || src[i] == '\t' || src[i] == '\n') {
            tail += src[i];
        } else
        if (src[i] == '_') {
            int presum = 0;
            while (i < src.size()) {
                tail += src[i];
                if (src[i] == '{') presum ++;
                else if (src[i] == '}') {
                    presum--;
                    if (presum==0)
                        break;
                }
                i++;
            }
        } else break;
    }
    tail += "}«CODE:";
    for (; i<src.size(); i++) tail += src[i];
    return tails.emplace(std::move(key), std::move(tail)).first->second;
}

void CodeOp::jit_prepare(JK& jk) {

    // forward: in0 in1 in2 -> out0 out1
    // backward: in0 in1 in2 in3(pout0) in4(pout1)
    jk << "«IN_SIZE:" << JK::dec3(_inputs.size());
    for (uint i=0; i<_inputs.size(); i++) {
        //LOGir<<JK::dec3(i);
        jk << "«in" << JK::dec3(i) << "_dim:"
            << JK::hex1(_inputs[i]->shape.size());
        jk << "«in" << JK::dec3(i) << "_type:"
            << _inputs[i]->dtype();
    }
    jk << "«OUT_SIZE:" << JK::dec3(_outputs.size());
    for (uint i=0; i<_outputs.size(); i++) {
        jk << "«out" << JK::dec3(i) << "_dim:"
            << JK::hex1(_outputs[i]->shape.size());
        jk << "«out" << JK::dec3(i) << "_type:"
            << _outputs[i]->dtype();
    }
    string& header = flags.get(NodeFlags::_cuda) ? 
        cuda_header : cpu_header;
    string& src = flags.get(NodeFlags::_cuda) ? 
        cuda_src : cpu_src;

    CHECK(src.size());
    jk << "«HEADER:" << code_op_key_tail(header, src);
}

} // jittor

#else // JIT

#pragma GCC diagnostic ignored "-Wunused-variable"

@for(i, 0, IN_SIZE,
    @define(in@i@@_stride@{in@i@@_dim-1},1)
)
@for(i, 0, OUT_SIZE,
    @define(out@i@@_stride@{out@i@@_dim-1},1)
)

@define(ARGS_DEF, 
@for(i, 0, IN_SIZE, @(
    in@i@@_type* __restrict__ in@i@@_p,
    @for(j, 0, in@i@@_dim, @(index_t in@i@@_shape@j,))
))
@for(i, 0, OUT_SIZE, @(
    out@i@@_type* __restrict__ out@i@@_p,
    @for(j, 0, out@i@@_dim, @(index_t out@i@@_shape@j,))
))
int __tmp
)

@define(ARGS, 
@for(i, 0, IN_SIZE, @(
    in@i@@_p,
    @for(j, 0, in@i@@_dim, @(in@i@@_shape@j,))
))
@for(i, 0, OUT_SIZE, @(
    out@i@@_p,
    @for(j, 0, out@i@@_dim, @(out@i@@_shape@j,))
))
0
)

@define(PRECALC,
@for(i, 0, IN_SIZE,
    @for(j, in@i@@_dim-2, -1, -1, auto in@i@@_stride@j = in@i@@_stride@{j+1} * in@i@@_shape@{j+1};)
)
@for(i, 0, OUT_SIZE,
    @for(j, out@i@@_dim-2, -1, -1, auto out@i@@_stride@j = out@i@@_stride@{j+1} * out@i@@_shape@{j+1};)
)
)

@alias(out, out0)
#undef out

@HEADER

#define out out0

namespace jittor {

void CodeOp::jit_run() {
    // define inputs
    @for(i, 0, IN_SIZE,
        auto in@i = _inputs[@i];
        auto* __restrict__ in@i@@_p = _inputs[@i]->ptr<in@i@@_type>();
        @for(j, 0, in@i@@_dim, index_t in@i@@_shape@j = _inputs[@i]->shape[@j];)
    )
    // define outputs
    @for(i, 0, OUT_SIZE,
        auto out@i = _outputs[@i];
        auto* __restrict__ out@i@@_p = _outputs[@i]->ptr<out@i@@_type>();
        @for(j, 0, out@i@@_dim, index_t out@i@@_shape@j = _outputs[@i]->shape[@j];)
    )

    @PRECALC

    @CODE
}

} // jittor

#endif // JIT
