// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cmath>
#include "core/var.h"
#include "ops/composite/code_op.h"
#include "ops/composite/code_source.h"
#include "ops/op_register.h"
#include "runtime/device.h"
#include <mutex>
#include <sstream>
#include <cstring>
#include <unordered_map>

#define __inline_static__ inline static

#ifndef JIT

namespace jittor {

static auto make_code = op_constructor<VarPtr, NanoVector, NanoString, vector<Var*>&&, string&&, vector<string>&&, string&&, string&&, vector<string>&&, string&&, DataMap&&, string&&>("code");

static auto make_code_multi = op_constructor<vector<VarPtr>, vector<NanoVector>&&, vector<NanoString>&&, vector<Var*>&&, string&&, vector<string>&&, string&&, string&&, vector<string>&&, string&&, DataMap&&, string&&>("code");
    
static inline void check_vary_shape(NanoVector v) {
    USER_CHECK(v.size()) << "Vary shape should not be zero dimension";
    for (int i=0; i<v.size(); i++)
        USER_CHECK((i == 0) ^ (v[i] >= 0))
            << "Vary shape should only occur in the first dimension:" << v;
}

CodeOp::CodeOp(NanoVector shape, NanoString dtype, vector<Var*>&& inputs, 
    string&& cpu_src, vector<string>&& cpu_grad_src, string&& cpu_header, 
    string&& cuda_src, vector<string>&& cuda_grad_src, string&& cuda_header,
    DataMap&& data, string&& backend)
    : _inputs(inputs), cpu_src(move(cpu_src)), cpu_grad_src(move(cpu_grad_src)), cpu_header(move(cpu_header)),
    cuda_src(move(cuda_src)), cuda_grad_src(move(cuda_grad_src)), cuda_header(move(cuda_header)), 
    data(move(data)), backend(move(backend))
{
    set_flag(OpFlags::_cpu, !!this->cpu_src.size());
    set_flag(OpFlags::_cuda, !!this->cuda_src.size());
    _outputs.push_back(create_output(shape, dtype));

    if (_outputs[0]->num < 0) {
        check_vary_shape(_outputs[0]->shape);
    }
    configure_grad();
}


CodeOp::CodeOp(
    vector<NanoVector>&& shapes, vector<NanoString>&& dtypes, vector<Var*>&& inputs, 
    string&& cpu_src, vector<string>&& cpu_grad_src, string&& cpu_header, 
    string&& cuda_src, vector<string>&& cuda_grad_src, string&& cuda_header,
    DataMap&& data, string&& backend)
    : _inputs(inputs), cpu_src(move(cpu_src)), cpu_grad_src(move(cpu_grad_src)), cpu_header(move(cpu_header)),
    cuda_src(move(cuda_src)), cuda_grad_src(move(cuda_grad_src)), cuda_header(move(cuda_header)), 
    data(move(data)), backend(move(backend))
{
    set_flag(OpFlags::_cpu, !!this->cpu_src.size());
    set_flag(OpFlags::_cuda, !!this->cuda_src.size());
    USER_CHECKop(shapes.size(),==,dtypes.size()) << "Number of outputs' shapes and dtypes should be the same";
    _outputs.resize(shapes.size());
    USER_CHECKop(_outputs.size(),>,0);
    for (int i=0; i<shapes.size(); i++) {
        _outputs[i] = create_output(shapes[i], dtypes[i]);
        if (_outputs[i]->num < 0) {
            check_vary_shape(_outputs[i]->shape);
        }
    }
    configure_grad();
}

CodeOp::CodeOp(
    vector<Var*>&& inputs, vector<Var*>&& outputs, 
    string&& cpu_src, vector<string>&& cpu_grad_src, string&& cpu_header, 
    string&& cuda_src, vector<string>&& cuda_grad_src, string&& cuda_header,
    DataMap&& data, string&& backend)
    : _inputs(inputs), cpu_src(move(cpu_src)), cpu_grad_src(move(cpu_grad_src)), cpu_header(move(cpu_header)),
    cuda_src(move(cuda_src)), cuda_grad_src(move(cuda_grad_src)), cuda_header(move(cuda_header)), 
    data(move(data)), backend(move(backend))
{
    set_flag(OpFlags::_cpu, !!this->cpu_src.size());
    set_flag(OpFlags::_cuda, !!this->cuda_src.size());
    _outputs.resize(outputs.size());
    USER_CHECKop(_outputs.size(),>,0);
    for (int i=0; i<outputs.size(); i++) {
        auto o = outputs[i];
        _outputs[i] = create_output(o->shape, o->dtype());
        _outputs[i]->share_with(o);
        /*
            TODO: vary shape not allowed in direct output
        */
    }
    configure_grad();
}

void CodeOp::configure_grad() {
    USER_CHECK(backend.empty() || backend == "cuda" || backend == "acl"
               || backend == "rocm" || backend == "corex")
        << "code backend must be cuda, acl, rocm, or corex";
    if (cuda_grad_src.size() == 0 && cpu_grad_src.size() == 0)
        set_flag(OpFlags::_manual_set_vnbb);
    auto iter = data.find("multi_grad");
    if (iter != data.end() && iter->second != 0) {
        USER_CHECK(cpu_grad_src.size() || cuda_grad_src.size())
            << "multi-output code gradient requires a gradient source";
        set_flag(OpFlags::_grads);
    }
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
        DataMap(data), string(backend)
    );
}

void CodeOp::grads(Var** douts, VarPtr* dins) {
    string cpu_src = cpu_grad_src.size() ? cpu_grad_src[0] : "";
    string cuda_src = cuda_grad_src.size() ? cuda_grad_src[0] : "";
    if (!cpu_src.size() && !cuda_src.size())
        return;

    int output_index = 0;
    auto iter = data.find("multi_grad_output");
    if (iter != data.end()) {
        USER_CHECK(std::isfinite(iter->second) && std::floor(iter->second) == iter->second
                   && iter->second >= 0 && iter->second < _outputs.size())
            << "multi_grad_output must be a valid integer output index";
        output_index = int(iter->second);
    }
    USER_CHECKop(output_index,>=,0);
    USER_CHECKop(output_index,<,_outputs.size());
    if (douts[output_index] == nullptr)
        return;

    int input_count = _inputs.size();
    iter = data.find("multi_grad_input_count");
    if (iter != data.end()) {
        USER_CHECK(std::isfinite(iter->second) && std::floor(iter->second) == iter->second
                   && iter->second > 0 && iter->second <= _inputs.size())
            << "multi_grad_input_count must be an integer within the input list";
        input_count = int(iter->second);
    }
    USER_CHECKop(input_count,>,0);
    USER_CHECKop(input_count,<=,_inputs.size());

    auto inputs = clone(_inputs);
    std::stringstream new_alias;
    new_alias << "\n@alias(dout,in" << JK::dec3(inputs.size()) << ")\n";
    inputs.push_back(douts[output_index]);
    for (int i=0; i<_outputs.size(); i++) {
        new_alias << "\n@alias(pout" << JK::dec3(i) << ",in"
                  << JK::dec3(inputs.size()) << ")\n";
        if (i == output_index)
            new_alias << "\n@alias(pout,in" << JK::dec3(inputs.size()) << ")\n";
        inputs.push_back(_outputs[i]);
    }

    vector<NanoVector> shapes;
    vector<NanoString> dtypes;
    shapes.reserve(input_count);
    dtypes.reserve(input_count);
    for (int i=0; i<input_count; i++) {
        auto input = _inputs[i];
        shapes.push_back(input->shape);
        dtypes.push_back(input->dtype());
    }
    auto alias = new_alias.str();
    DataMap gradient_data(data);
    gradient_data.erase("multi_grad");
    gradient_data.erase("multi_grad_output");
    gradient_data.erase("multi_grad_input_count");
    auto outputs = make_code_multi(
        move(shapes), move(dtypes), move(inputs),
        move(cpu_src), {}, alias+cpu_header,
        move(cuda_src), {}, alias+cuda_header,
        move(gradient_data), string(backend)
    );
    CHECKop(outputs.size(),==,input_count);
    for (int i=0; i<outputs.size(); i++)
        dins[i] = move(outputs[i]);
}

// The part of the key after "«HEADER:" is a pure function of the header and the
// source: it walks the source once, character by character, to lift the CUDA
// kernel definitions into the header. That walk was re-run on EVERY call, and an
// inference kernel's source runs to well over a kilobyte -- a four-figure loop of
// single-character appends in front of a kernel that itself takes about a
// microsecond. The distinct sources are few (one per specialised kernel), so
// remember the result. References into an unordered_map stay valid across
// rehashes, and the lock only guards the map, not the walk's cost.
static string code_op_build_tail(const string& header, const string& src) {
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
    return tail;
}

namespace {
// Eight bytes a step, two independent lanes: a digest to look a source up by,
// several times faster than hashing it with std::hash. The lookup still
// compares the text, so a collision costs a miss, never a wrong kernel.
inline void source_digest(const string& text, uint64& a, uint64& b) {
    const unsigned char* p = (const unsigned char*)text.data();
    size_t n = text.size(), i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64 w;
        memcpy(&w, p + i, 8);
        a = (a ^ w) * 0x9E3779B97F4A7C15ull;
        a ^= a >> 29;
        b = (b + w) * 0xC2B2AE3D27D4EB4Full;
        b = (b << 31) | (b >> 33);
    }
    for (; i < n; i++) {
        a = (a ^ p[i]) * 0x100000001B3ull;
        b = (b + p[i]) * 0x9E3779B97F4A7C15ull;
    }
}

struct InternedSource {
    string header, src, token, tail;
};

struct DigestHash {
    size_t operator()(const pair<uint64, uint64>& d) const { return d.first ^ (d.second * 31); }
};

std::mutex interned_lock;
std::unordered_multimap<pair<uint64, uint64>, InternedSource*, DigestHash> by_digest;
unordered_map<string, InternedSource*> by_token;
} // namespace

const string& code_source_token(const string& header, const string& src) {
    uint64 a = 0x243F6A8885A308D3ull ^ header.size(), b = 0x13198A2E03707344ull ^ src.size();
    source_digest(header, a, b);
    a ^= 0x1; b += 0x1;
    source_digest(src, a, b);
    auto digest = std::make_pair(a, b);
    std::lock_guard<std::mutex> guard(interned_lock);
    auto range = by_digest.equal_range(digest);
    for (auto it = range.first; it != range.second; ++it)
        if (it->second->src == src && it->second->header == header)
            return it->second->token;
    // First sight of this text (or a colliding one): build its tail once.
    auto* entry = new InternedSource{header, src, "", code_op_build_tail(header, src)};
    std::stringstream token;
    token << code_source_prefix << std::hex << a << '_' << b << '_' << std::dec << entry->tail.size();
    entry->token = token.str();
    int dup = 0;
    while (by_token.count(entry->token)) entry->token = token.str() + "_" + std::to_string(++dup);
    by_digest.emplace(digest, entry);
    by_token.emplace(entry->token, entry);
    return entry->token;
}

const string* code_source_tail(const string& token) {
    std::lock_guard<std::mutex> guard(interned_lock);
    auto found = by_token.find(token);
    return found == by_token.end() ? nullptr : &found->second->tail;
}

void CodeOp::jit_prepare(JK& jk) {
    for (auto* output : _outputs)
        USER_CHECK(output->is_contiguous())
            << "CodeOp output buffers require contiguous storage";
    if (!backend.empty()) {
        if (executes_on_accelerator()) {
            auto expected = backend == "cuda" ? BackendId::Cuda
                : backend == "acl" ? BackendId::Acl
                : backend == "rocm" ? BackendId::Rocm : BackendId::Corex;
            USER_CHECK(execution_backend() == expected)
                << "code backend source is for" << backend
                << "but execution backend is" << backend_ops(execution_backend()).name;
        }
        add_jit_define(jk, "source_backend", backend);
    }

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
    const string& header = executes_on_accelerator() ?
        cuda_header : cpu_header;
    const string& src = executes_on_accelerator() ?
        cuda_src : cpu_src;

    USER_CHECK(src.size()) << "code requires source for the selected backend";
    jk << "«HEADER:" << code_source_token(header, src);
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
