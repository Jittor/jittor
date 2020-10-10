// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <limits>

#include "node.h"
#include "op.h"
#include "var.h"
#include "op_compiler.h"
#include "profiler/profiler.h"
#include "mem/allocator.h"
#include "misc/cuda_flags.h"
#include "pybind/py_var_tracer.h"

namespace jittor {

DECLARE_FLAG(string, cache_path);

DEFINE_FLAG(int, try_use_32bit_index, 0,
    "If not overflow, try to use 32 bit type as index type.");

string_view_map<jit_op_entry_t> jit_ops;
string_view_map<string> jit_key_mapper;

int64_t Op::number_of_lived_ops = 0;

Op::Op() {
    flags.set(NodeFlags::_var, 0);
    flags.set(NodeFlags::_cpu, 1);
    number_of_lived_ops++;
}

Op::~Op() {
    number_of_lived_ops--;
}

void Op::forward(Var* input) {
    flags.set(NodeFlags::_forwarded);
    outputs_holder.emplace_back(input);
}

VarPtr Op::duplicate() {
    return nullptr;
}

VarPtr Op::grad(Var* out, Var* dout, Var* v, int v_index) {
    LOGw << "Grad of" << name() << "return zeros";
    return nullptr;
}

void Op::grads(Var** douts, VarPtr* dins) {
    LOGw << "Grads of" << name() << "return zeros";
}

Var* Op::create_output(NanoVector shape, NanoString dtype) {
    VarPtr vp(shape, dtype);
    Var* output = vp.ptr;
    outputs_holder.emplace_back(move(vp));
    return output;
}

void Op::init() {
    infer_shape();
    LOGvvvv << "Create" << this << "and outputs" << outputs();
    for (Var* v : outputs())
        CHECK(v->shape.size()) << "Number of dims should be solved.";
}

bool Op::shape_infered() {
    if (flags.get(NodeFlags::_vary_shape)) return true;
    for (Var* v : outputs())
        if (v->num < 0) return false;
    return true;
}

void Op::compile_optimize(string& src) {}

void Op::infer_shape() {}
void Op::run() {}
void Op::jit_prepare() {}
void Op::graph_optimize() {}

string Op::name_ex() const {
    string a=name();
    if (ns!=ns_void) {
        a += '.';
        a += ns.to_cstring();
    }
    return a;
}

string Op::get_jit_key() {
    jk.clear();
    do_jit_prepare();
    return jk.to_string();
}

vector<pair<string,string>> Op::get_jit_define() {
    return parse_jit_keys(get_jit_key());
}

void Op::do_jit_prepare() {
    memcheck_all_exist();
    jk << name();
    jit_prepare();
    if (jk.empty()) {
        // not a jit op
        bool has_cuda = flags.get(NodeFlags::_cuda);
        bool has_cpu = flags.get(NodeFlags::_cpu);
        CHECK(has_cuda || has_cpu);
        if (has_cuda && has_cpu && !use_cuda)
            flags.set(NodeFlags::_cuda, 0);
    } else {
        // check use int64_t as index_t if array is too big
        int in_id=0, out_id=0;
        bool use_int64_t = false;
        // TODO: fused op do not have inputs,
        //   check use_cuda_op from outputs may not be enough
        bool use_cuda_op = use_cuda;
        for (Var* var : inputs()) {
            if (var->mem_ptr) {
                /* jit key don't include here, because 
                    parallel compiler don't known
                jk << JK::key << "alloc_i" << JK::hex1(in_id)
                    << JK::hex1(var->allocator->flags()) << JK::end;
                */
                use_cuda_op &= var->allocator->is_cuda();
            }
            if (var->num >= std::numeric_limits<int32_t>::max())
                use_int64_t = true;
            in_id ++;
        }
        for (Var* var : outputs()) {
            if (var->mem_ptr) {
                /*
                jk << JK::key << "alloc_o" << JK::hex1(in_id)
                    << JK::hex1(var->allocator->flags()) << JK::end;
                */
                use_cuda_op &= var->allocator->is_cuda();
            }
            if (var->num >= std::numeric_limits<int32_t>::max())
                use_int64_t = true;
            out_id ++;
        }
        add_jit_define("JIT", "1");
        if (use_cuda_op && flags.get(NodeFlags::_cuda)) {
            add_jit_define("JIT_cuda", "1");
            flags.set(NodeFlags::_cpu, 0);
            // TODO: 64bit index in CUDA
            use_int64_t = false;
        } else {
            if (use_cuda==2) {
                if (flags.get(NodeFlags::_cuda))
                    LOGf << "Op" << name() >> "'s vars are not allocated in cuda";
                else
                    LOGf << "Op" << name() << "doesn't have cuda version";
            }
            ASSERT(flags.get(NodeFlags::_cpu))
                << "Op" << name() << "doesn't have cpu version";
            add_jit_define("JIT_cpu", "1");
            flags.set(NodeFlags::_cuda, 0);
        }
        if (try_use_32bit_index) use_int64_t = false;
        add_jit_define("index_t", use_int64_t ? "int64" : "int32");
    }
    jk.finilize();
}

void Op::do_prepare(){
    jk.clear();
    do_jit_prepare();
}

void Op::do_run_after_prepare() {
    if (!jk.empty())
        jit_run();
    else
        run();
}

void Op::do_run() {
    do_prepare();
    do_run_after_prepare();
}

string Op::get_filename_from_jit_key(const string& jit_key, const string& suffix) {
    auto iter = jit_key_mapper.find(jit_key);
    string s = iter==jit_key_mapper.end() ? jit_key : iter->second;
    std::stringstream ss;
    if (s.size() > 100) {
        ss << s.substr(0, 90) << "...hash:"
            << std::hex << std::hash<string>()(s);
    } else {
        ss << s << "_hash:" << 
            std::hex << std::hash<string>()(s);
    }
    s = ss.str();
    for (char& c : s) {
        if (c=='[' || c==']' || c=='<' || c=='>'
            || c=='{' || c=='}' || c=='(' || c==')' || c==',' 
            || c=='\n' || c=='\t' || c==' ' || c=='&' || c=='|'
            || c=='/')
            c = '_';
    }
    string filename = cache_path + "/jit/";
    filename += s;
    filename += "_op";
    filename += suffix;
    return filename;
}

// convert xxx.yyy -> xxx
string Op::op_name_to_file_name(const string& s) {
    auto pos = s.find('.');
    return pos == string::npos ? s : s.substr(0, pos);
}
// convert xxx_xxx -> XxxXxx
string Op::file_name_to_class_name(const string& s) {
    char prev = '_';
    string res;
    res.reserve(s.size());
    for (char c : s) {
        if (c != '_') {
            if (prev == '_')
                res += c-'a'+'A';
            else
                res += c;
        }
        prev = c;
    }
    return res;
}

void Op::jit_run() {
    const char* jit_key = jk.to_cstring();
    auto iter = jit_ops.find(jit_key);
    if (iter != jit_ops.end()) {
        LOGvvv <<  "Jit op key found:" << jit_key << "jit op entry:" << (void*)iter->second;
        Profiler::record_and_run(iter->second, this, jit_key);
        return;
    }
    LOGvv << "Jit op key not found:" << jit_key;
    // compile JIT op
    string prev_jit_key = jit_key;
    auto op_entry = OpCompiler::do_compile(this);
    string new_jit_key = get_jit_key();
    jit_ops[new_jit_key] = jit_ops[prev_jit_key] = op_entry;
    jit_key_mapper[prev_jit_key] = new_jit_key;
    LOGvv << "Get jit op entry:" << (void*)op_entry;
    Profiler::record_and_run(op_entry, this, new_jit_key.c_str());
}

void Op::statistics(uint64_t& in, uint64_t& out, uint64_t& compute) {
    in = out = compute = 0;
    for (Var* var : inputs()) {
        in += var->size;
        compute = std::max(compute, (uint64_t)var->num);
    }
    for (Var* var : outputs()) {
        out += var->size;
        compute = std::max(compute, (uint64_t)var->num);
    }
}

std::ostream& operator<<(std::ostream& os, const Op* op) {
    if (!op) return os << "Op(0)";
    os << "Op(" << (void*)op
        << ':' << op->forward_liveness
        << ':' << op->backward_liveness
        << ':' << op->pending_liveness
        << ":i" << op->_inputs.size()
        << ":o" << op->_outputs.size()
        << ":s" << op->is_finished()
        << "," << op->name_ex();
    if (op->_outputs.size()>1)
        os << "->...";
    else if (op->_outputs.size() == 1) {
        auto v = (Var*)op->_outputs.front().node;
        if (v->name.size())
            os << "->" << v->name;
        else
            os << "->" << (void*)v;
    }
    os << ')';
#ifdef NODE_MEMCHECK
    os << '<' << op->__id() << '>';
    print_node_trace(op, os);
#endif
    return os;
}

} // jittor